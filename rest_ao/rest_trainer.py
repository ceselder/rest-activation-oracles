"""ReST (Reinforced Self-Training) trainer for calibrated Activation Oracles.

The training loop:
1. GROW: Sample oracle responses for (activation, question) pairs
2. SCORE: Have judge score informativeness, compute reward
3. IMPROVE: Weighted SFT on high-reward samples
4. Repeat for N rounds
"""

import gc
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rest_ao.config import RESTConfig
from rest_ao.data_pipeline import PromptQuestionPair, load_diverse_prompts, create_prompt_question_pairs
from rest_ao.epistemic_status import OracleOutput, parse_oracle_output, ORACLE_SYSTEM_PROMPT
from rest_ao.judge import InformativenessJudge, LocalJudge
from rest_ao.question_generation import QuestionGenerator
from rest_ao.reward import (
    RewardResult,
    compute_batch_rewards,
    filter_by_reward,
    normalize_rewards_to_weights,
)

# Import from activation oracle repo
import sys
# Try multiple paths for flexibility
for path in [
    str(Path(__file__).parent.parent.parent / "activation-oracle-rest"),
    str(Path(__file__).parent.parent.parent / "activation_oracles"),
    "/root/activation_oracles",
]:
    if Path(path).exists():
        sys.path.insert(0, path)
        break

from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.common import load_model, load_tokenizer, set_seed, layer_percent_to_layer
from nl_probes.utils.dataset_utils import get_introspection_prefix, SPECIAL_TOKEN


class RESTTrainer:
    """Trainer for calibrated Activation Oracle using ReST."""

    def __init__(self, cfg: RESTConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)

        # Will be initialized in setup()
        self.model = None
        self.tokenizer = None
        self.submodule = None
        self.judge = None
        self.question_generator = None

    def setup(self):
        """Initialize model, tokenizer, and other components."""
        print(f"Loading model: {self.cfg.model_name}")
        set_seed(self.cfg.seed)

        self.tokenizer = load_tokenizer(self.cfg.model_name)
        # Load model with eager attention (no flash attention)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            attn_implementation="eager",
        )

        # Setup LoRA
        if self.cfg.oracle_lora_path:
            print(f"Loading LoRA from: {self.cfg.oracle_lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.cfg.oracle_lora_path,
                is_trainable=True,
            )
        else:
            print("Initializing new LoRA adapter")
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

        # Get the hook submodule - for PEFT models, access through base_model.model
        # Hook at layer 1 for activation injection
        base_model = self.model.base_model.model
        self.submodule = base_model.model.layers[self.cfg.hook_layer]

        # Store layer accessor for activation extraction
        self._get_layer = lambda l: base_model.model.layers[l]

        # Question generator (uses base model without LoRA) - BATCHED
        self.question_generator = QuestionGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=self.cfg.question_temperature,
            device=str(self.device),
            batch_size=self.cfg.question_batch_size,
        )

        # Judge - use local for efficiency since we have the model loaded
        self.judge = LocalJudge(
            model=self.model,
            tokenizer=self.tokenizer,
            device=str(self.device),
        )

        print("Setup complete!")

    def _extract_activations(
        self,
        context_input_ids: list[int],
        context_positions: list[int],
        layer: int,
    ) -> torch.Tensor:
        """Extract activations from the model for given positions.

        Returns:
            Tensor of shape [num_positions, hidden_dim]
        """
        input_ids = torch.tensor([context_input_ids], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Use direct hook approach for PEFT compatibility
        activations = []

        def capture_hook(module, inp, out):
            if isinstance(out, tuple):
                activations.append(out[0].detach())
            else:
                activations.append(out.detach())

        target_layer = self._get_layer(layer)
        handle = target_layer.register_forward_hook(capture_hook)

        with self.model.disable_adapter():
            with torch.no_grad():
                self.model(**inputs)

        handle.remove()

        acts = activations[0][0]  # [L, D] from [1, L, D]
        acts = acts[context_positions, :]  # [num_positions, D]

        return acts.detach()

    def _generate_oracle_response(
        self,
        pair: PromptQuestionPair,
        activations: torch.Tensor,
    ) -> str:
        """Generate a single oracle response with epistemic status.

        Args:
            pair: The prompt-question pair
            activations: Extracted activations [num_positions, D]

        Returns:
            Raw oracle response string
        """
        num_positions = activations.shape[0]
        prefix = get_introspection_prefix(pair.layer, num_positions)

        # Build the prompt with system instructions
        messages = [
            {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
            {"role": "user", "content": prefix + pair.question},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        # Find positions of the special tokens for activation injection
        input_ids_list = input_ids[0].tolist()
        special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
        positions = [i for i, tok in enumerate(input_ids_list) if tok == special_token_id]
        positions = positions[:num_positions]

        if len(positions) != num_positions:
            # Fallback: just use first positions
            positions = list(range(num_positions))

        # Create steering hook
        hook_fn = get_hf_activation_steering_hook(
            vectors=[activations],
            positions=[positions],
            steering_coefficient=1.0,
            device=self.device,
            dtype=self.dtype,
        )

        # Generate with activation injection
        with add_hook(self.submodule, hook_fn):
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.oracle_temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def grow_phase(
        self,
        pairs: list[PromptQuestionPair],
    ) -> list[tuple[PromptQuestionPair, list[str]]]:
        """GROW phase: Generate multiple oracle responses per pair.

        Args:
            pairs: List of prompt-question pairs

        Returns:
            List of (pair, responses) tuples
        """
        print(f"GROW phase: Generating {self.cfg.samples_per_question} responses per question")
        results = []

        for pair in tqdm(pairs, desc="Generating responses"):
            # Extract activations
            acts = self._extract_activations(
                pair.context_input_ids,
                pair.context_positions,
                pair.layer,
            )

            # Generate multiple responses
            responses = []
            for _ in range(self.cfg.samples_per_question):
                resp = self._generate_oracle_response(pair, acts)
                responses.append(resp)

            results.append((pair, responses))

        return results

    def score_phase(
        self,
        grow_results: list[tuple[PromptQuestionPair, list[str]]],
    ) -> list[tuple[PromptQuestionPair, str, RewardResult]]:
        """SCORE phase: Have judge score responses, compute rewards.

        Args:
            grow_results: Output from grow phase

        Returns:
            List of (pair, response, reward) tuples
        """
        print("SCORE phase: Computing informativeness and rewards")

        # Flatten for batch judging
        all_prompts = []
        all_questions = []
        all_answers = []
        all_pairs = []
        all_raw_responses = []

        for pair, responses in grow_results:
            for resp in responses:
                parsed = parse_oracle_output(resp)
                all_prompts.append(pair.prompt)
                all_questions.append(pair.question)
                all_answers.append(parsed.answer)
                all_pairs.append(pair)
                all_raw_responses.append(resp)

        # Batch judge
        print(f"Judging {len(all_answers)} responses...")
        judge_results = self.judge.score_batch_sync(
            all_prompts,
            all_questions,
            all_answers,
        )

        # Compute rewards
        parsed_outputs = [parse_oracle_output(r) for r in all_raw_responses]
        informativeness_scores = [j.informativeness for j in judge_results]

        rewards = compute_batch_rewards(
            parsed_outputs,
            informativeness_scores,
            self.cfg.calibration_lambda,
        )

        # Zip back together
        results = list(zip(all_pairs, all_raw_responses, rewards))

        return results

    def improve_phase(
        self,
        scored_samples: list[tuple[PromptQuestionPair, str, RewardResult]],
        round_num: int,
    ) -> dict:
        """IMPROVE phase: Train on high-reward samples.

        Args:
            scored_samples: Output from score phase
            round_num: Current ReST round number

        Returns:
            Dict of training metrics
        """
        print("IMPROVE phase: Training on filtered samples")

        # Filter bottom X%
        samples = [(p, r) for p, r, _ in scored_samples]
        rewards = [rw for _, _, rw in scored_samples]

        filtered_samples, filtered_rewards = filter_by_reward(
            samples,
            rewards,
            self.cfg.filter_bottom_percent,
        )

        print(f"Kept {len(filtered_samples)}/{len(samples)} samples after filtering")

        # Compute training weights
        weights = normalize_rewards_to_weights(filtered_rewards)

        # Prepare optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
        )

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Simple training loop
        random.shuffle(list(zip(filtered_samples, weights, filtered_rewards)))

        for i in tqdm(range(0, len(filtered_samples), self.cfg.batch_size), desc="Training"):
            batch = filtered_samples[i:i + self.cfg.batch_size]
            batch_weights = weights[i:i + self.cfg.batch_size]

            batch_loss = 0.0

            for (pair, response), weight in zip(batch, batch_weights):
                # Create training input
                acts = self._extract_activations(
                    pair.context_input_ids,
                    pair.context_positions,
                    pair.layer,
                )

                num_positions = acts.shape[0]
                prefix = get_introspection_prefix(pair.layer, num_positions)

                messages = [
                    {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
                    {"role": "user", "content": prefix + pair.question},
                    {"role": "assistant", "content": response},
                ]

                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                ).to(self.device)

                # Create labels (mask prompt, keep response)
                labels = input_ids.clone()
                # Find where assistant starts
                user_end = self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).shape[1]
                labels[0, :user_end] = -100

                # Find positions for activation injection
                input_ids_list = input_ids[0].tolist()
                special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
                positions = [j for j, tok in enumerate(input_ids_list) if tok == special_token_id]
                positions = positions[:num_positions]

                if len(positions) != num_positions:
                    positions = list(range(num_positions))

                # Create hook
                hook_fn = get_hf_activation_steering_hook(
                    vectors=[acts],
                    positions=[positions],
                    steering_coefficient=1.0,
                    device=self.device,
                    dtype=self.dtype,
                )

                # Forward with hook
                with add_hook(self.submodule, hook_fn):
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                    )
                    loss = outputs.loss * weight

                loss.backward()
                batch_loss += loss.item()

            # Step optimizer
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

            total_loss += batch_loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Compute metrics
        metrics = {
            "train/loss": avg_loss,
            "train/num_samples": len(filtered_samples),
            "train/mean_reward": sum(r.reward for r in filtered_rewards) / len(filtered_rewards),
            "train/mean_informativeness": sum(r.informativeness for r in filtered_rewards) / len(filtered_rewards),
            "train/mean_confidence": sum(r.confidence for r in filtered_rewards) / len(filtered_rewards),
            "train/mean_brier": sum(r.brier_score for r in filtered_rewards) / len(filtered_rewards),
            "train/confidence_std": torch.tensor([r.confidence for r in filtered_rewards]).std().item(),
            "train/parse_success_rate": sum(r.parse_success for r in filtered_rewards) / len(filtered_rewards),
        }

        return metrics

    def run_benchmark_eval(self, round_num: int) -> dict:
        """Evaluate on classification benchmarks and return metrics for wandb."""
        from rest_ao.calibration_eval import compute_calibration_metrics

        # Quick eval on 2 datasets with 50 samples each (fast but informative)
        eval_datasets = ["sst2", "geometry_of_truth"]
        eval_samples = 50

        all_confidences = []
        all_correct = []
        metrics = {}

        self.model.eval()

        for ds_name in eval_datasets:
            try:
                # Import here to avoid circular deps
                from nl_probes.dataset_classes.classification import (
                    ClassificationDatasetConfig,
                    ClassificationDatasetLoader,
                )
                from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig

                config = DatasetLoaderConfig(
                    dataset_name="",
                    num_train=0,
                    num_test=eval_samples,
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

                confidences = []
                correct = []

                for dp in eval_data[:eval_samples]:
                    # Quick eval - generate oracle response
                    num_pos = len(dp.positions)
                    prefix = get_introspection_prefix(dp.layer, num_pos)
                    question = self.tokenizer.decode(dp.input_ids, skip_special_tokens=True)
                    if "Answer with" in question:
                        question = question[question.find("Answer with"):]

                    messages = [
                        {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
                        {"role": "user", "content": prefix + question},
                    ]

                    encoded = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                    )
                    input_ids = encoded["input_ids"].to(self.device)

                    # Get steering vectors
                    if dp.steering_vectors is not None:
                        sv = dp.steering_vectors.to(self.device)
                    else:
                        continue

                    input_ids_list = input_ids[0].tolist()
                    special_token_id = self.tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
                    positions = [j for j, t in enumerate(input_ids_list) if t == special_token_id][:num_pos]

                    hook_fn = get_hf_activation_steering_hook(
                        vectors=[sv], positions=[positions],
                        steering_coefficient=1.0, device=self.device, dtype=self.dtype
                    )

                    with torch.no_grad(), add_hook(self.submodule, hook_fn):
                        out = self.model.generate(
                            input_ids, max_new_tokens=30, do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id
                        )

                    response = self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
                    parsed = parse_oracle_output(response)

                    target = dp.target_output.strip().lower()
                    answer = parsed.answer.strip().lower().rstrip(".!?,;:")

                    confidences.append(parsed.confidence)
                    correct.append(answer == target)

                if confidences:
                    ds_metrics = compute_calibration_metrics(confidences, correct)
                    metrics[f"eval/{ds_name}_accuracy"] = ds_metrics.accuracy
                    metrics[f"eval/{ds_name}_ece"] = ds_metrics.ece
                    metrics[f"eval/{ds_name}_brier"] = ds_metrics.brier
                    all_confidences.extend(confidences)
                    all_correct.extend(correct)

                    print(f"  {ds_name}: acc={ds_metrics.accuracy:.3f}, ece={ds_metrics.ece:.4f}")

            except Exception as e:
                print(f"  Eval failed for {ds_name}: {e}")

        # Overall metrics
        if all_confidences:
            overall = compute_calibration_metrics(all_confidences, all_correct)
            metrics["eval/overall_accuracy"] = overall.accuracy
            metrics["eval/overall_ece"] = overall.ece
            metrics["eval/overall_brier"] = overall.brier

        self.model.train()
        return metrics

    def train(self):
        """Run the full ReST training loop."""
        print("Starting ReST training")

        # Initialize wandb
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name,
            config=asdict(self.cfg),
        )

        # Load prompts
        print(f"Loading {self.cfg.num_prompts} diverse prompts...")
        prompts = load_diverse_prompts(
            num_prompts=self.cfg.num_prompts,
            seed=self.cfg.seed,
        )

        for round_num in range(self.cfg.num_rest_rounds):
            print(f"\n{'='*60}")
            print(f"ReST Round {round_num + 1}/{self.cfg.num_rest_rounds}")
            print(f"{'='*60}\n")

            # Generate fresh questions each round (BATCHED)
            print("Generating questions...")
            pairs = create_prompt_question_pairs(
                prompts[:1000],  # Use subset per round
                self.question_generator,
                self.tokenizer,
                self.cfg.layer_percents,
                self.cfg.model_name,
                questions_per_prompt=self.cfg.questions_per_prompt,
            )

            # GROW
            grow_results = self.grow_phase(pairs)

            # SCORE
            scored = self.score_phase(grow_results)

            # IMPROVE
            metrics = self.improve_phase(scored, round_num)

            # Log training metrics
            wandb.log({
                "round": round_num,
                **metrics,
            })

            print(f"\nRound {round_num + 1} metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            # Run benchmark evaluation and log to wandb
            eval_metrics = self.run_benchmark_eval(round_num)
            if eval_metrics:
                wandb.log({
                    "round": round_num,
                    **eval_metrics,
                })

            # Save checkpoint
            save_path = Path(self.cfg.save_dir) / f"round_{round_num}"
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")

            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()

        print("\nTraining complete!")
        wandb.finish()
