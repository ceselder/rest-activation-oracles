"""GRPO trainer for calibrated Activation Oracles.

Simple loop:
1. Load (prompt, question) from WildChat dataset
2. Extract activations, generate G responses
3. Score with Gemini Flash judge
4. Compute advantage = (reward - mean) / std
5. Policy gradient update
"""

import gc
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from grpo_ao.config import GRPOConfig
from grpo_ao.epistemic_status import parse_oracle_output, ORACLE_SYSTEM_PROMPT
from grpo_ao.judge import InformativenessJudge
from grpo_ao.reward import compute_reward

# Import from activation oracle repo
import sys
for path in [
    str(Path(__file__).parent.parent.parent / "activation_oracles"),
    "/root/activation_oracles",
]:
    if Path(path).exists():
        sys.path.insert(0, path)
        break

from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.common import load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import get_introspection_prefix, SPECIAL_TOKEN


class GRPOTrainer:
    """Simple GRPO trainer for calibrated Activation Oracle."""

    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.step = 0

    def setup(self):
        """Initialize model, tokenizer, judge."""
        print(f"Loading model: {self.cfg.model_name}")
        set_seed(self.cfg.seed)

        self.tokenizer = load_tokenizer(self.cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
        )

        # Load pretrained AO LoRA
        if self.cfg.oracle_lora_path:
            print(f"Loading LoRA: {self.cfg.oracle_lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.cfg.oracle_lora_path,
                is_trainable=True,
            )
        else:
            lora_config = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=self.cfg.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()
        self.model.enable_input_require_grads()

        # Find layers for activation extraction/injection
        base_model = self.model.base_model.model
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
            self.layers = base_model.model.layers
        elif hasattr(base_model, 'layers'):
            self.layers = base_model.layers
        else:
            raise RuntimeError(f"Could not find layers in {type(base_model)}")

        self.submodule = self.layers[self.cfg.hook_layer]

        # Judge (Gemini Flash via OpenRouter)
        print(f"Using judge: {self.cfg.judge_model}")
        self.judge = InformativenessJudge(
            model=self.cfg.judge_model,
            temperature=self.cfg.judge_temperature,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
        )

        # Load dataset
        print("Loading WildChat oracle questions dataset...")
        self.dataset = load_dataset("ceselder/wildchat-oracle-questions", split="train")
        print(f"Loaded {len(self.dataset)} examples")

        print("Setup complete!")

    def extract_activations(self, text: str, layer_percent: int) -> tuple[torch.Tensor, list[int], list[int]]:
        """Extract activations from text at specified layer.

        Returns: (activations, input_ids, positions)
        """
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        # Compute layer index from percentage
        num_layers = len(self.layers)
        layer_idx = int(num_layers * layer_percent / 100)

        activations = []
        def capture_hook(module, inp, out):
            if isinstance(out, tuple):
                activations.append(out[0].detach())
            else:
                activations.append(out.detach())

        handle = self.layers[layer_idx].register_forward_hook(capture_hook)

        with self.model.disable_adapter(), torch.no_grad():
            self.model(input_ids)

        handle.remove()

        # Get activations at last few positions (multi-token window)
        acts = activations[0][0]  # [seq_len, hidden]
        window = min(5, acts.shape[0])
        positions = list(range(acts.shape[0] - window, acts.shape[0]))
        steering_vectors = acts[positions]  # [window, hidden]

        return steering_vectors, input_ids[0].tolist(), positions, layer_idx

    def generate_responses(
        self,
        question: str,
        steering_vectors: torch.Tensor,
        layer: int,
        num_responses: int,
    ) -> list[str]:
        """Generate multiple oracle responses with activation injection."""
        num_positions = steering_vectors.shape[0]
        prefix = get_introspection_prefix(layer, num_positions)

        messages = [
            {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
            {"role": "user", "content": prefix + question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        # Find special token positions
        input_ids_list = input_ids[0].tolist()
        special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
        positions = [i for i, t in enumerate(input_ids_list) if t == special_token_id][:num_positions]
        if len(positions) != num_positions:
            positions = list(range(num_positions))

        hook_fn = get_hf_activation_steering_hook(
            vectors=[steering_vectors],
            positions=[positions],
            steering_coefficient=1.0,
            device=self.device,
            dtype=self.dtype,
        )

        responses = []
        for _ in range(num_responses):
            with add_hook(self.submodule, hook_fn):
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=True,
                    temperature=self.cfg.oracle_temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(response)

        return responses

    def compute_grpo_loss(
        self,
        question: str,
        steering_vectors: torch.Tensor,
        layer: int,
        responses: list[str],
        advantages: list[float],
    ) -> torch.Tensor:
        """Compute GRPO policy gradient loss."""
        num_positions = steering_vectors.shape[0]
        prefix = get_introspection_prefix(layer, num_positions)

        total_loss = 0.0

        for response, advantage in zip(responses, advantages):
            messages = [
                {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
                {"role": "user", "content": prefix + question},
                {"role": "assistant", "content": response},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
            ).to(self.device)

            # Create labels (mask prompt)
            labels = input_ids.clone()
            prompt_ids = self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            labels[0, :prompt_ids.shape[1]] = -100

            # Find positions for steering
            input_ids_list = input_ids[0].tolist()
            special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
            positions = [i for i, t in enumerate(input_ids_list) if t == special_token_id][:num_positions]
            if len(positions) != num_positions:
                positions = list(range(num_positions))

            hook_fn = get_hf_activation_steering_hook(
                vectors=[steering_vectors],
                positions=[positions],
                steering_coefficient=1.0,
                device=self.device,
                dtype=self.dtype,
            )

            with add_hook(self.submodule, hook_fn):
                outputs = self.model(input_ids=input_ids, labels=labels)

            # Weight loss by advantage (GRPO: positive advantage = reinforce, negative = penalize)
            total_loss += outputs.loss * advantage

        return total_loss / len(responses)

    def train_step(self, example: dict) -> dict:
        """Single GRPO training step on one example."""
        prompt = example["wildchat_question"]
        questions = example["oracle_questions"]

        # Pick a random question and layer
        question = random.choice(questions)
        layer_percent = random.choice(self.cfg.layer_percents)

        # Extract activations
        steering_vectors, _, _, layer_idx = self.extract_activations(prompt, layer_percent)

        # Generate G responses
        responses = self.generate_responses(
            question, steering_vectors, layer_idx, self.cfg.num_generations
        )

        # Parse and score each response
        parsed = [parse_oracle_output(r) for r in responses]

        # Judge informativeness (async batch)
        judge_results = self.judge.score_batch_sync(
            [prompt] * len(responses),
            [question] * len(responses),
            [p.answer for p in parsed],
        )

        # Compute rewards
        rewards = []
        for p, j in zip(parsed, judge_results):
            r = compute_reward(p, j.informativeness, self.cfg.calibration_lambda)
            rewards.append(r)

        reward_vals = [r.reward for r in rewards]
        mean_reward = sum(reward_vals) / len(reward_vals)
        std_reward = (sum((r - mean_reward)**2 for r in reward_vals) / len(reward_vals)) ** 0.5

        # Compute advantages (group-relative)
        if std_reward > 1e-8:
            advantages = [(r - mean_reward) / std_reward for r in reward_vals]
        else:
            advantages = [0.0] * len(reward_vals)  # No learning if all same

        # GRPO loss
        self.model.train()
        loss = self.compute_grpo_loss(question, steering_vectors, layer_idx, responses, advantages)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Metrics
        metrics = {
            "loss": loss.item(),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_confidence": sum(r.confidence for r in rewards) / len(rewards),
            "mean_informativeness": sum(r.informativeness for r in rewards) / len(rewards),
            "mean_brier": sum(r.brier_score for r in rewards) / len(rewards),
            "advantage_std": (sum(a**2 for a in advantages) / len(advantages)) ** 0.5,
        }

        return metrics, responses, rewards, question, prompt

    def train(self):
        """Main training loop."""
        print("Starting GRPO training")

        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name,
            config=asdict(self.cfg),
        )

        # Shuffle dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        pbar = tqdm(total=self.cfg.num_train_steps, desc="Training")

        for step in range(self.cfg.num_train_steps):
            self.step = step
            idx = indices[step % len(indices)]
            example = self.dataset[idx]

            metrics, responses, rewards, question, prompt = self.train_step(example)

            # Log to wandb
            wandb.log({"step": step, **metrics})

            # Periodic logging
            if step % self.cfg.log_samples_every == 0:
                print(f"\n{'='*70}")
                print(f"Step {step}")
                print(f"{'='*70}")
                print(f"Prompt: {prompt[:100]}...")
                print(f"Question: {question}")
                print(f"\nSample responses:")
                for i, (resp, rew) in enumerate(zip(responses[:3], rewards[:3])):
                    parsed = parse_oracle_output(resp)
                    print(f"  [{i+1}] conf={rew.confidence}/100 info={rew.informativeness:.2f} reward={rew.reward:.3f}")
                    print(f"      {resp[:100]}...")
                print(f"\nMetrics: loss={metrics['loss']:.4f} mean_reward={metrics['mean_reward']:.3f} adv_std={metrics['advantage_std']:.3f}")
                print(f"{'='*70}\n")

            # Checkpoint
            if (step + 1) % self.cfg.checkpoint_every == 0:
                save_path = Path(self.cfg.save_dir) / f"step_{step}"
                save_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")

            pbar.update(1)
            pbar.set_postfix(loss=f"{metrics['loss']:.3f}", reward=f"{metrics['mean_reward']:.3f}")

            # Clear cache periodically
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        pbar.close()
        print("Training complete!")
        wandb.finish()
