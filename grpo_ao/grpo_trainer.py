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
from grpo_ao.calibration_eval import compute_calibration_metrics

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


def format_messages_for_model(
    system_prompt: str,
    user_content: str,
    assistant_content: str | None = None,
    supports_system_role: bool = True,
) -> list[dict]:
    """Format messages, handling models with/without system role support.

    Gemma 2 doesn't support system messages - merge into user.
    Gemma 3, Qwen, etc. do support system messages.
    """
    if supports_system_role:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        # Merge system prompt into user message for Gemma 2 compatibility
        merged_user = f"{system_prompt}\n\n{user_content}"
        messages = [{"role": "user", "content": merged_user}]

    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    return messages


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
        self.tokenizer.padding_side = "left"  # Required for decoder-only generation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
        )

        # Model-specific quirks
        model_name_lower = self.cfg.model_name.lower()
        # Gemma 2 doesn't support system role in chat template
        self.supports_system_role = "gemma-2" not in model_name_lower
        # Gemma 3 (multimodal) needs token_type_ids
        self.needs_token_type_ids = "gemma-3" in model_name_lower
        # Qwen3 has thinking mode that needs to be disabled
        self.is_qwen3 = "qwen3" in model_name_lower or "qwen-3" in model_name_lower
        print(f"Model quirks: supports_system_role={self.supports_system_role}, needs_token_type_ids={self.needs_token_type_ids}, is_qwen3={self.is_qwen3}")

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

        # torch.compile for faster generation
        # NOTE: May not work with steering hooks - they do in-place tensor modifications
        if self.cfg.use_torch_compile:
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("Model compiled!")

        # NOTE: Gradient checkpointing conflicts with steering hooks during backward pass
        # (hooks fire differently during recomputation, causing tensor count mismatch)
        # Disabled for now - Gemma 2 9B should fit with 16 gens on H200

        # Find layers for activation extraction/injection
        # Navigate through PEFT wrapper to find the transformer layers
        base_model = self.model.base_model.model
        self.layers = None

        # Try various model architectures
        candidates = [
            # Gemma 3 VLM: model.language_model.layers (the actual path!)
            lambda m: m.model.language_model.layers if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'layers') else None,
            # Gemma 3 VLM: language_model.model.layers
            lambda m: m.language_model.model.layers if hasattr(m, 'language_model') and hasattr(m.language_model, 'model') and hasattr(m.language_model.model, 'layers') else None,
            # Gemma 3 VLM alternate: language_model.layers
            lambda m: m.language_model.layers if hasattr(m, 'language_model') and hasattr(m.language_model, 'layers') else None,
            # Standard causal LM: model.layers
            lambda m: m.model.layers if hasattr(m, 'model') and hasattr(m.model, 'layers') else None,
            # Direct layers
            lambda m: m.layers if hasattr(m, 'layers') else None,
        ]

        for get_layers in candidates:
            try:
                layers = get_layers(base_model)
                if layers is not None and len(layers) > 0:
                    self.layers = layers
                    break
            except Exception:
                continue

        if self.layers is None:
            # Debug output - use named_children to see actual submodules
            print(f"DEBUG: base_model type = {type(base_model)}")
            print(f"DEBUG: base_model children = {list(base_model.named_children())[:5]}")
            for name, child in base_model.named_children():
                print(f"DEBUG: {name} -> {type(child)}")
                if hasattr(child, 'named_children'):
                    for subname, subchild in list(child.named_children())[:3]:
                        print(f"DEBUG:   {subname} -> {type(subchild)}")
            raise RuntimeError(f"Could not find layers in {type(base_model)}")

        print(f"Found {len(self.layers)} layers")

        self.submodule = self.layers[self.cfg.hook_layer]

        # Judge (Gemini 3 Flash via OpenRouter with CoT)
        print(f"Using judge: {self.cfg.judge_model}")
        self.judge = InformativenessJudge(
            model=self.cfg.judge_model,
            temperature=self.cfg.judge_temperature,
            max_tokens=self.cfg.judge_max_tokens,
            thinking_level=self.cfg.judge_thinking_level,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
        )

        # Load dataset
        print("Loading WildChat oracle questions dataset...")
        self.dataset = load_dataset("ceselder/wildchat-oracle-questions-1k", split="train")
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
            kwargs = {"token_type_ids": torch.zeros_like(input_ids)} if self.needs_token_type_ids else {}
            self.model(input_ids, **kwargs)

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

        messages = format_messages_for_model(ORACLE_SYSTEM_PROMPT, prefix + question, supports_system_role=self.supports_system_role)

        # Disable Qwen3 thinking mode
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
                enable_thinking=False,  # Qwen3 specific
            )
        except TypeError:
            # Fallback for models that don't support enable_thinking
            encoded = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors=None
            )
        # Handle different return types
        if hasattr(encoded, 'input_ids'):
            input_ids_list = encoded.input_ids
        elif isinstance(encoded, list):
            input_ids_list = encoded
        else:
            input_ids_list = list(encoded)
        input_ids = torch.tensor([input_ids_list], device=self.device)

        # Find special token positions
        special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
        positions = [i for i, t in enumerate(input_ids_list) if t == special_token_id][:num_positions]
        if len(positions) != num_positions:
            positions = list(range(num_positions))

        # Batch all generations together for efficiency
        batched_input_ids = input_ids.repeat(num_responses, 1)  # [G, seq_len]

        # Hook needs vectors/positions for each batch item
        hook_fn = get_hf_activation_steering_hook(
            vectors=[steering_vectors] * num_responses,
            positions=[positions] * num_responses,
            steering_coefficient=1.0,
            device=self.device,
            dtype=self.dtype,
        )

        gen_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": True,
            "temperature": self.cfg.oracle_temperature,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.needs_token_type_ids:
            gen_kwargs["token_type_ids"] = torch.zeros_like(batched_input_ids)

        with add_hook(self.submodule, hook_fn):
            outputs = self.model.generate(batched_input_ids, **gen_kwargs)

        responses = []
        for i in range(num_responses):
            response = self.tokenizer.decode(outputs[i][input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(response)

        return responses

    def generate_responses_batched(
        self,
        examples: list[dict],
    ) -> list[list[str]]:
        """Generate responses for multiple examples in one batched call.

        Args:
            examples: List of dicts with keys: question, steering_vectors, layer_idx

        Returns:
            List of response lists, one per example
        """
        G = self.cfg.num_generations
        N = len(examples)

        all_input_ids = []
        all_vectors = []
        all_positions = []
        input_lengths = []

        for ex in examples:
            question = ex["question"]
            steering_vectors = ex["steering_vectors"]
            layer_idx = ex["layer_idx"]
            num_positions = steering_vectors.shape[0]

            prefix = get_introspection_prefix(layer_idx, num_positions)
            messages = format_messages_for_model(ORACLE_SYSTEM_PROMPT, prefix + question, supports_system_role=self.supports_system_role)

            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
                    enable_thinking=False,
                )
            except TypeError:
                encoded = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors=None
                )

            input_ids_list = encoded.input_ids if hasattr(encoded, 'input_ids') else list(encoded)
            input_lengths.append(len(input_ids_list))

            special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
            positions = [i for i, t in enumerate(input_ids_list) if t == special_token_id][:num_positions]
            if len(positions) != num_positions:
                positions = list(range(num_positions))

            # Repeat G times for this example
            for _ in range(G):
                all_input_ids.append(input_ids_list)
                all_vectors.append(steering_vectors)
                all_positions.append(positions)

        # Pad to same length
        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        for ids in all_input_ids:
            padding = [self.tokenizer.pad_token_id] * (max_len - len(ids))
            padded_input_ids.append(ids + padding)

        batched_input_ids = torch.tensor(padded_input_ids, device=self.device)  # [N*G, max_len]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (batched_input_ids != self.tokenizer.pad_token_id).long()

        hook_fn = get_hf_activation_steering_hook(
            vectors=all_vectors,
            positions=all_positions,
            steering_coefficient=1.0,
            device=self.device,
            dtype=self.dtype,
        )

        gen_kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": True,
            "temperature": self.cfg.oracle_temperature,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.needs_token_type_ids:
            gen_kwargs["token_type_ids"] = torch.zeros_like(batched_input_ids)

        with add_hook(self.submodule, hook_fn):
            outputs = self.model.generate(batched_input_ids, **gen_kwargs)

        # Parse responses back into per-example lists
        all_responses = []
        for i in range(N):
            example_responses = []
            for j in range(G):
                idx = i * G + j
                input_len = input_lengths[i]
                # Account for padding: actual content starts after padding
                pad_len = max_len - input_len
                response = self.tokenizer.decode(
                    outputs[idx][max_len:],  # Skip full padded input
                    skip_special_tokens=True
                )
                example_responses.append(response)
            all_responses.append(example_responses)

        return all_responses

    def compute_grpo_loss(
        self,
        question: str,
        steering_vectors: torch.Tensor,
        layer: int,
        responses: list[str],
        advantages: list[float],
    ) -> tuple[torch.Tensor, dict]:
        """Compute GRPO policy gradient loss with KL penalty.

        Returns:
            (loss, metrics_dict) tuple with KL and policy divergence metrics
        """
        num_positions = steering_vectors.shape[0]
        prefix = get_introspection_prefix(layer, num_positions)

        total_loss = 0.0
        total_kl = 0.0
        self._max_log_ratio = 0.0
        self._clipfrac = 0.0

        for response, advantage in zip(responses, advantages):
            messages = format_messages_for_model(ORACLE_SYSTEM_PROMPT, prefix + question, response, supports_system_role=self.supports_system_role)

            try:
                full_encoded = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, return_tensors=None,
                    enable_thinking=False,
                )
            except TypeError:
                full_encoded = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, return_tensors=None
                )
            full_ids_list = full_encoded.input_ids if hasattr(full_encoded, 'input_ids') else list(full_encoded)
            input_ids = torch.tensor([full_ids_list], device=self.device)

            # Create labels (mask prompt)
            try:
                prompt_encoded = self.tokenizer.apply_chat_template(
                    messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors=None,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_encoded = self.tokenizer.apply_chat_template(
                    messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors=None
                )
            prompt_ids_list = prompt_encoded.input_ids if hasattr(prompt_encoded, 'input_ids') else list(prompt_encoded)
            labels = input_ids.clone()
            labels[0, :len(prompt_ids_list)] = -100

            # Find positions for steering
            special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
            positions = [i for i, t in enumerate(full_ids_list) if t == special_token_id][:num_positions]
            if len(positions) != num_positions:
                positions = list(range(num_positions))

            hook_fn = get_hf_activation_steering_hook(
                vectors=[steering_vectors],
                positions=[positions],
                steering_coefficient=1.0,
                device=self.device,
                dtype=self.dtype,
            )

            # Forward pass with current policy (adapter enabled)
            model_kwargs = {"token_type_ids": torch.zeros_like(input_ids)} if self.needs_token_type_ids else {}
            with add_hook(self.submodule, hook_fn):
                outputs = self.model(input_ids=input_ids, labels=labels, **model_kwargs)
                # Get log probs for KL computation
                logits = outputs.logits[:, :-1, :]  # [1, seq_len-1, vocab]
                target_ids = input_ids[:, 1:]  # [1, seq_len-1]
                log_probs = F.log_softmax(logits, dim=-1)
                # Gather log probs for actual tokens
                current_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]

            # KL penalty: compute reference log probs (adapter disabled)
            if self.cfg.kl_penalty > 0:
                with torch.no_grad(), self.model.disable_adapter(), add_hook(self.submodule, hook_fn):
                    ref_outputs = self.model(input_ids=input_ids, **model_kwargs)
                    ref_logits = ref_outputs.logits[:, :-1, :]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_log_probs_gathered = ref_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

                # KL divergence on response tokens only (where labels != -100)
                response_mask = (labels[:, 1:] != -100).float()  # [1, seq_len-1]
                log_ratio = current_log_probs - ref_log_probs_gathered  # [1, seq_len-1]
                # Mean KL per response token
                kl = (log_ratio * response_mask).sum() / response_mask.sum().clamp(min=1)
                total_kl += kl.item()

                # Track policy divergence metrics (like PPO's clipfrac)
                # log_ratio = log(π/π_ref), so ratio = exp(log_ratio)
                # Large |log_ratio| means policy diverged significantly
                masked_log_ratio = log_ratio * response_mask
                max_log_ratio = masked_log_ratio.abs().max().item()
                # "Clip fraction": tokens where |log(π/π_ref)| > 0.2 (~20% prob change)
                clip_threshold = 0.2
                clipped = ((masked_log_ratio.abs() > clip_threshold) * response_mask).sum()
                clipfrac = (clipped / response_mask.sum().clamp(min=1)).item()
                total_max_log_ratio = max(getattr(self, '_max_log_ratio', 0.0), max_log_ratio)
                self._max_log_ratio = total_max_log_ratio
                self._clipfrac = getattr(self, '_clipfrac', 0.0) + clipfrac

            # Dr. GRPO length bias fix: multiply by response length to get sum of token losses
            loss = outputs.loss
            if self.cfg.fix_length_bias:
                response_len = (labels[0] != -100).sum().item()
                if response_len > 0:
                    loss = loss * response_len  # Convert mean -> sum

            # Weight loss by advantage (GRPO: positive advantage = reinforce, negative = penalize)
            total_loss += loss * advantage

        # Add KL penalty to loss
        if self.cfg.kl_penalty > 0:
            total_loss += self.cfg.kl_penalty * total_kl

        mean_kl = total_kl / len(responses)
        mean_clipfrac = self._clipfrac / len(responses)

        loss_metrics = {
            "mean_kl": mean_kl,
            "max_log_ratio": self._max_log_ratio,  # Max |log(π/π_ref)| - large = unstable
            "clipfrac": mean_clipfrac,  # Fraction of tokens with large policy shift
        }

        # Dr. GRPO: divide by (max_length * num_generations) not just num_generations
        if self.cfg.fix_length_bias:
            return total_loss / (self.cfg.max_new_tokens * len(responses)), loss_metrics
        else:
            return total_loss / len(responses), loss_metrics

    def sample_unrelated_question(self, exclude_idx: int) -> str:
        """Sample a question from a different example (for 'unrelated' probes)."""
        # Pick a random different example
        other_idx = exclude_idx
        while other_idx == exclude_idx:
            other_idx = random.randint(0, len(self.dataset) - 1)
        other_example = self.dataset[other_idx]
        return random.choice(other_example["oracle_questions"])

    def train_step_batched(self, examples_with_idx: list[tuple[dict, int]]) -> dict:
        """GRPO training step with batched generation across multiple examples.

        Args:
            examples_with_idx: List of (example, dataset_idx) tuples

        Returns:
            Aggregated metrics, last example's responses/rewards for logging
        """
        N = len(examples_with_idx)
        G = self.cfg.num_generations

        # Step 1: Prepare all examples (extract activations, select questions)
        prepared = []
        for example, example_idx in examples_with_idx:
            prompt = example["wildchat_question"]
            questions = example["oracle_questions"]

            # Dataset already has 50% relevant + 50% unrelated questions mixed in
            question = random.choice(questions)

            layer_percent = random.choice(self.cfg.layer_percents)
            steering_vectors, _, _, layer_idx = self.extract_activations(prompt, layer_percent)

            prepared.append({
                "prompt": prompt,
                "question": question,
                "steering_vectors": steering_vectors,
                "layer_idx": layer_idx,
            })

        # Step 2: Batch generate all N*G responses
        all_responses = self.generate_responses_batched(prepared)  # List of N lists of G responses

        # Step 3: Parse and score all responses
        all_parsed = []
        all_prompts = []
        all_questions = []
        all_answers = []
        for i, ex in enumerate(prepared):
            for response in all_responses[i]:
                parsed = parse_oracle_output(response)
                all_parsed.append(parsed)
                all_prompts.append(ex["prompt"])
                all_questions.append(ex["question"])
                all_answers.append(parsed.answer)

        judge_results = self.judge.score_batch_sync(all_prompts, all_questions, all_answers)

        # Step 4: Compute rewards and advantages per example, accumulate loss
        all_metrics = []
        total_scaled_loss = 0.0

        self.model.train()
        for i, ex in enumerate(prepared):
            responses = all_responses[i]
            parsed = all_parsed[i*G : (i+1)*G]
            judge_res = judge_results[i*G : (i+1)*G]

            rewards = []
            for p, j in zip(parsed, judge_res):
                r = compute_reward(p, j.informativeness, self.cfg.calibration_lambda)
                rewards.append(r)

            reward_vals = [r.reward for r in rewards]
            mean_reward = sum(reward_vals) / len(reward_vals)
            std_reward = (sum((r - mean_reward)**2 for r in reward_vals) / len(reward_vals)) ** 0.5

            # Dr. GRPO: center, don't scale
            if self.cfg.scale_rewards == "none":
                advantages = [r - mean_reward for r in reward_vals]
            elif self.cfg.scale_rewards == "group" and std_reward > 1e-8:
                advantages = [(r - mean_reward) / std_reward for r in reward_vals]
            else:
                advantages = [0.0] * len(reward_vals)

            # Compute loss for this example
            loss, loss_metrics = self.compute_grpo_loss(
                ex["question"], ex["steering_vectors"], ex["layer_idx"],
                responses, advantages
            )

            # Scale for gradient accumulation (N examples per batch, gradient_accumulation_steps batches)
            scale = self.cfg.gradient_accumulation_steps * N
            scaled_loss = loss / scale
            scaled_loss.backward()

            all_metrics.append({
                "loss": loss.item(),
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_confidence": sum(r.confidence for r in rewards) / len(rewards),
                "mean_informativeness": sum(r.informativeness for r in rewards) / len(rewards),
                "mean_brier": sum(r.brier_score for r in rewards) / len(rewards),
                "advantage_std": (sum(a**2 for a in advantages) / len(advantages)) ** 0.5,
                **loss_metrics,  # mean_kl, max_log_ratio, clipfrac
            })

        # Aggregate metrics across examples
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        # Return last example's details for logging
        last_ex = prepared[-1]
        last_responses = all_responses[-1]
        last_rewards = []
        for p, j in zip(all_parsed[-G:], judge_results[-G:]):
            last_rewards.append(compute_reward(p, j.informativeness, self.cfg.calibration_lambda))

        return avg_metrics, last_responses, last_rewards, last_ex["question"], last_ex["prompt"]

    def train_step(self, example: dict, example_idx: int) -> dict:
        """Single GRPO forward/backward on one example (no optimizer step).

        Args:
            example: Dataset example with wildchat_question and oracle_questions
            example_idx: Index of this example in the dataset (for sampling unrelated questions)

        Returns:
            metrics, responses, rewards, question, prompt
        """
        prompt = example["wildchat_question"]
        questions = example["oracle_questions"]

        # Dataset already has 50% relevant + 50% unrelated questions mixed in
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

        # Compute advantages (Dr. GRPO: don't scale by std to avoid difficulty bias)
        if self.cfg.scale_rewards == "none":
            # Dr. GRPO: just center, don't scale
            advantages = [r - mean_reward for r in reward_vals]
        elif self.cfg.scale_rewards == "group" and std_reward > 1e-8:
            # Original GRPO: scale by group std
            advantages = [(r - mean_reward) / std_reward for r in reward_vals]
        else:
            advantages = [0.0] * len(reward_vals)  # No learning if all same

        # GRPO loss
        self.model.train()
        loss, loss_metrics = self.compute_grpo_loss(question, steering_vectors, layer_idx, responses, advantages)

        # Scale loss for gradient accumulation and backward
        # Optimizer step is handled by the main training loop
        scaled_loss = loss / self.cfg.gradient_accumulation_steps
        scaled_loss.backward()

        # Metrics
        metrics = {
            "loss": loss.item(),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_confidence": sum(r.confidence for r in rewards) / len(rewards),
            "mean_informativeness": sum(r.informativeness for r in rewards) / len(rewards),
            "mean_brier": sum(r.brier_score for r in rewards) / len(rewards),
            "advantage_std": (sum(a**2 for a in advantages) / len(advantages)) ** 0.5,
            **loss_metrics,  # mean_kl, max_log_ratio, clipfrac
        }

        return metrics, responses, rewards, question, prompt

    def run_holdout_eval(self, step: int) -> dict:
        """Run evaluation on classification benchmarks (holdout tasks).

        Uses the same datasets as sft.py: geometry_of_truth, sst2, etc.
        """
        from nl_probes.utils.dataset_utils import (
            get_introspection_prefix,
            materialize_missing_steering_vectors,
            SPECIAL_TOKEN,
        )
        from nl_probes.dataset_classes.classification import (
            ClassificationDatasetConfig,
            ClassificationDatasetLoader,
        )
        from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig

        CLASSIFICATION_DATASETS = {
            "geometry_of_truth": 100,
            "sst2": 100,
            "ag_news": 100,
            "tense": 100,
            "singular_plural": 100,
        }

        if self.cfg.eval_datasets:
            CLASSIFICATION_DATASETS = {k: v for k, v in CLASSIFICATION_DATASETS.items()
                                       if k in self.cfg.eval_datasets}

        self.model.eval()
        all_confidences = []
        all_correct = []
        eval_results = {}

        for ds_name, num_test in CLASSIFICATION_DATASETS.items():
            print(f"  Evaluating {ds_name}...")

            try:
                config = DatasetLoaderConfig(
                    dataset_name="",
                    num_train=0,
                    num_test=num_test,
                    splits=["test"],
                    model_name=self.cfg.model_name,
                    layer_percents=self.cfg.layer_percents,
                    save_acts=False,
                    batch_size=0,
                    seed=42,
                    custom_dataset_params=ClassificationDatasetConfig(
                        classification_dataset_name=ds_name,
                        max_window_size=5,
                        min_end_offset=-1,
                        max_end_offset=-5,
                        num_qa_per_sample=1,
                    ),
                )
                loader = ClassificationDatasetLoader(dataset_config=config)
                eval_data = loader.load_dataset("test")
            except Exception as e:
                print(f"    Failed to load {ds_name}: {e}")
                continue

            if not eval_data:
                continue

            # Materialize steering vectors
            eval_data = materialize_missing_steering_vectors(eval_data, self.tokenizer, self.model)

            ds_confidences = []
            ds_correct = []

            for dp in eval_data[:num_test]:
                num_positions = len(dp.positions)
                prefix = get_introspection_prefix(dp.layer, num_positions)
                question = self.tokenizer.decode(dp.input_ids, skip_special_tokens=True)
                if "Answer with" in question:
                    question = question[question.find("Answer with"):]

                messages = format_messages_for_model(ORACLE_SYSTEM_PROMPT, prefix + question, supports_system_role=self.supports_system_role)

                try:
                    encoded = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
                        enable_thinking=False,  # Disable Qwen3 thinking mode
                    )
                except TypeError:
                    encoded = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors=None
                    )
                input_ids_list = encoded.input_ids if hasattr(encoded, 'input_ids') else list(encoded)
                input_ids = torch.tensor([input_ids_list], device=self.device)

                special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
                positions = [i for i, t in enumerate(input_ids_list) if t == special_token_id][:num_positions]
                if len(positions) != num_positions:
                    positions = list(range(num_positions))

                steering_vectors = dp.steering_vectors
                if steering_vectors is None:
                    continue

                hook_fn = get_hf_activation_steering_hook(
                    vectors=[steering_vectors],
                    positions=[positions],
                    steering_coefficient=1.0,
                    device=self.device,
                    dtype=self.dtype,
                )

                eval_gen_kwargs = {
                    "max_new_tokens": 30,
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
                if self.needs_token_type_ids:
                    eval_gen_kwargs["token_type_ids"] = torch.zeros_like(input_ids)

                with torch.no_grad(), add_hook(self.submodule, hook_fn):
                    output_ids = self.model.generate(input_ids, **eval_gen_kwargs)

                response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                parsed = parse_oracle_output(response)

                target = dp.target_output.strip().lower()
                answer = parsed.answer.strip().lower().rstrip(".!?,;:")
                is_correct = answer == target

                ds_confidences.append(parsed.confidence_normalized)
                ds_correct.append(is_correct)

            if ds_confidences:
                metrics = compute_calibration_metrics(ds_confidences, ds_correct)
                eval_results[ds_name] = {
                    "accuracy": metrics.accuracy,
                    "ece": metrics.ece,
                    "brier": metrics.brier,
                    "mean_confidence": metrics.mean_confidence,
                    "n_samples": metrics.n_samples,
                }
                print(f"    {ds_name}: acc={metrics.accuracy:.3f} ece={metrics.ece:.3f} brier={metrics.brier:.3f}")
                all_confidences.extend(ds_confidences)
                all_correct.extend(ds_correct)

        # Overall metrics
        if all_confidences:
            overall = compute_calibration_metrics(all_confidences, all_correct)
            eval_results["_overall"] = {
                "accuracy": overall.accuracy,
                "ece": overall.ece,
                "brier": overall.brier,
                "mean_confidence": overall.mean_confidence,
                "n_samples": overall.n_samples,
            }
            print(f"  OVERALL: acc={overall.accuracy:.3f} ece={overall.ece:.3f} brier={overall.brier:.3f}")

            # Log to wandb
            wandb.log({
                f"eval/step": step,
                f"eval/accuracy": overall.accuracy,
                f"eval/ece": overall.ece,
                f"eval/brier": overall.brier,
                f"eval/mean_confidence": overall.mean_confidence,
            })

        self.model.train()
        return eval_results

    def train(self):
        """Main training loop."""
        print("Starting GRPO training")
        N = self.cfg.examples_per_batch
        G = self.cfg.num_generations
        print(f"Batch size: {N} examples × {G} generations = {N*G} rollouts per batch")
        print(f"Gradient accumulation: {self.cfg.gradient_accumulation_steps} batches")
        print(f"Effective batch size: {N * self.cfg.gradient_accumulation_steps} examples per optimizer step")

        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name,
            config=asdict(self.cfg),
        )

        # Shuffle dataset
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        # Total micro-steps = train_steps * gradient_accumulation_steps
        total_micro_steps = self.cfg.num_train_steps * self.cfg.gradient_accumulation_steps
        pbar = tqdm(total=self.cfg.num_train_steps, desc="Training")

        accumulated_metrics = []
        example_counter = 0  # Track position in shuffled indices

        for micro_step in range(total_micro_steps):
            # Determine if this is the last accumulation step (time to do optimizer step)
            is_last_accum = (micro_step + 1) % self.cfg.gradient_accumulation_steps == 0

            if N > 1:
                # Batched: sample N examples and process together
                examples_with_idx = []
                for _ in range(N):
                    idx = indices[example_counter % len(indices)]
                    example_counter += 1
                    examples_with_idx.append((self.dataset[idx], idx))

                metrics, responses, rewards, question, prompt = self.train_step_batched(examples_with_idx)
            else:
                # Single example (original behavior)
                idx = indices[example_counter % len(indices)]
                example_counter += 1
                example = self.dataset[idx]
                metrics, responses, rewards, question, prompt = self.train_step(example, idx)

            accumulated_metrics.append(metrics)

            # On optimizer step, aggregate metrics and log
            if is_last_accum:
                step = micro_step // self.cfg.gradient_accumulation_steps
                self.step = step

                # Clip gradients and step optimizer - capture grad norm for logging
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Average metrics across accumulated steps
                avg_metrics = {}
                for key in accumulated_metrics[0].keys():
                    avg_metrics[key] = sum(m[key] for m in accumulated_metrics) / len(accumulated_metrics)
                accumulated_metrics = []

                # Add grad norm to metrics
                avg_metrics["grad_norm"] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm

                # Use avg_metrics for logging
                metrics = avg_metrics

                # Log to wandb
                wandb.log({"step": step, **metrics})

                # Periodic logging
                if step % self.cfg.log_samples_every == 0:
                    print(f"\n{'='*70}")
                    print(f"Step {step}")
                    print(f"{'='*70}")
                    print(f"Prompt: {prompt[:100]}...")
                    print(f"Question: {question}")
                    print(f"\nSample AO responses:")
                    sample_table = []
                    for i, (resp, rew) in enumerate(zip(responses[:3], rewards[:3])):
                        parsed = parse_oracle_output(resp)
                        print(f"  [{i+1}] conf={rew.confidence}/100 info={rew.informativeness:.2f} reward={rew.reward:.3f}")
                        # Show full response (up to 300 chars)
                        print(f"      {resp[:300]}")
                        sample_table.append([rew.confidence, rew.informativeness, rew.reward, resp[:200]])
                    print(f"\nMetrics: loss={metrics['loss']:.4f} mean_reward={metrics['mean_reward']:.3f} adv_std={metrics['advantage_std']:.3f}")
                    if 'grad_norm' in metrics:
                        print(f"Grad norm: {metrics['grad_norm']:.4f} | Clipfrac: {metrics.get('clipfrac', 0):.3f} | Max log ratio: {metrics.get('max_log_ratio', 0):.3f}")
                    print(f"{'='*70}\n")

                    # Log samples to wandb for easy viewing
                    wandb.log({
                        "samples/question": question,
                        "samples/prompt_preview": prompt[:200],
                        "samples/responses": wandb.Table(
                            columns=["confidence", "informativeness", "reward", "response"],
                            data=sample_table
                        ),
                    }, step=step)

                # Checkpoint
                if (step + 1) % self.cfg.checkpoint_every == 0:
                    save_path = Path(self.cfg.save_dir) / f"step_{step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")

                    # Run holdout eval
                    if self.cfg.eval_at_checkpoints:
                        print(f"\nRunning holdout evaluation at step {step}...")
                        try:
                            eval_results = self.run_holdout_eval(step)
                        except Exception as e:
                            print(f"Eval failed: {e}")

                pbar.update(1)
                pbar.set_postfix(loss=f"{metrics['loss']:.3f}", reward=f"{metrics['mean_reward']:.3f}")

                # Clear cache periodically
                if step % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        pbar.close()

        # Save final checkpoint
        final_path = Path(self.cfg.save_dir) / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_path)
        print(f"Saved final checkpoint to {final_path}")

        # Push to HuggingFace Hub
        if self.cfg.push_to_hub:
            print(f"Pushing to HuggingFace Hub: {self.cfg.hub_repo_id}")
            try:
                self.model.push_to_hub(
                    self.cfg.hub_repo_id,
                    commit_message=f"GRPO trained for {self.cfg.num_train_steps} steps",
                )
                self.tokenizer.push_to_hub(self.cfg.hub_repo_id)
                print(f"Successfully pushed to {self.cfg.hub_repo_id}")
            except Exception as e:
                print(f"Failed to push to hub: {e}")

        print("Training complete!")
        wandb.finish()
