#!/usr/bin/env python3
"""Observe a big rollout without training - just vibes check.

Run this to see what the oracle outputs, how the judge scores them,
and whether the rewards make sense before committing to training.
"""

import argparse
import os
import random
from pathlib import Path

# Load .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from grpo_ao.config import GRPOConfig
from grpo_ao.epistemic_status import parse_oracle_output, ORACLE_SYSTEM_PROMPT
from grpo_ao.judge import InformativenessJudge
from grpo_ao.reward import compute_reward

import sys
for path in [
    str(Path(__file__).parent.parent / "activation_oracles"),
    "/root/activation_oracles",
]:
    if Path(path).exists():
        sys.path.insert(0, path)
        break

from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.common import load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import get_introspection_prefix, SPECIAL_TOKEN


def format_messages(user_content: str, use_system_prompt: bool = True):
    """Format as chat messages with system prompt for epistemic format."""
    if use_system_prompt:
        # Merge system prompt into user message for compatibility
        merged = f"{ORACLE_SYSTEM_PROMPT}\n\n{user_content}"
        return [{"role": "user", "content": merged}]
    else:
        return [{"role": "user", "content": user_content}]


def main():
    parser = argparse.ArgumentParser(description="Observe rollout vibes")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Number of examples to sample")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Generations per example")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--oracle_lora", type=str,
                        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_judge", action="store_true",
                        help="Skip judging (faster, just see raw outputs)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = load_tokenizer(args.model)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )

    # Model quirks
    model_lower = args.model.lower()
    needs_token_type_ids = "gemma-3" in model_lower
    is_qwen3 = "qwen3" in model_lower or "qwen-3" in model_lower

    if args.oracle_lora:
        print(f"Loading LoRA: {args.oracle_lora}")
        model = PeftModel.from_pretrained(model, args.oracle_lora, is_trainable=False)

    # Find layers
    base_model = model.base_model.model if hasattr(model, 'base_model') else model
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    elif hasattr(base_model, 'layers'):
        layers = base_model.layers
    else:
        raise RuntimeError("Could not find layers")

    print(f"Found {len(layers)} layers")

    # Judge (optional)
    judge = None
    if not args.no_judge:
        print("Initializing judge (GLM-4.7-flash)...")
        judge = InformativenessJudge(
            model="z-ai/glm-4.7-flash",
            temperature=0.5,
            max_tokens=5000,
        )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("ceselder/wildchat-oracle-questions-1k", split="train")
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    model.eval()
    cfg = GRPOConfig()

    print("\n" + "=" * 80)
    print("ROLLOUT OBSERVATION")
    print("=" * 80)

    all_results = []

    for ex_idx in range(args.num_examples):
        idx = indices[ex_idx]
        example = dataset[idx]
        prompt = example["wildchat_question"]
        questions = example["oracle_questions"]
        question = random.choice(questions)

        print(f"\n{'='*80}")
        print(f"EXAMPLE {ex_idx + 1}/{args.num_examples}")
        print(f"{'='*80}")
        print(f"\nPROMPT (first 300 chars):\n{prompt[:300]}...")
        print(f"\nQUESTION: {question}")

        # Extract activations
        layer_percent = random.choice([50])  # Middle layer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        layer_idx = int(len(layers) * layer_percent / 100)

        activations = []
        def capture_hook(module, inp, out):
            if isinstance(out, tuple):
                activations.append(out[0].detach())
            else:
                activations.append(out.detach())

        handle = layers[layer_idx].register_forward_hook(capture_hook)
        with torch.no_grad():
            if hasattr(model, 'disable_adapter'):
                with model.disable_adapter():
                    kwargs = {"token_type_ids": torch.zeros_like(input_ids)} if needs_token_type_ids else {}
                    model(input_ids, **kwargs)
            else:
                kwargs = {"token_type_ids": torch.zeros_like(input_ids)} if needs_token_type_ids else {}
                model(input_ids, **kwargs)
        handle.remove()

        acts = activations[0][0]
        window = min(5, acts.shape[0])
        positions = list(range(acts.shape[0] - window, acts.shape[0]))
        steering_vectors = acts[positions]
        num_positions = steering_vectors.shape[0]

        # Generate responses
        prefix = get_introspection_prefix(layer_idx, num_positions)
        messages = format_messages(prefix + question)

        try:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
                enable_thinking=False,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors=None
            )

        input_ids_list = encoded.input_ids if hasattr(encoded, 'input_ids') else list(encoded)
        gen_input_ids = torch.tensor([input_ids_list], device=device)

        # Find special token positions
        special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
        hook_positions = [i for i, t in enumerate(input_ids_list) if t == special_token_id][:num_positions]
        if len(hook_positions) != num_positions:
            hook_positions = list(range(num_positions))

        # Batch for all generations
        batched_input = gen_input_ids.repeat(args.num_generations, 1)

        hook_fn = get_hf_activation_steering_hook(
            vectors=[steering_vectors] * args.num_generations,
            positions=[hook_positions] * args.num_generations,
            steering_coefficient=1.0,
            device=device,
            dtype=dtype,
        )

        gen_kwargs = {
            "max_new_tokens": 300,
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": 0.95,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if needs_token_type_ids:
            gen_kwargs["token_type_ids"] = torch.zeros_like(batched_input)

        print(f"\nGenerating {args.num_generations} responses...")
        # Inject at layer 1 (oracle injection layer), NOT extraction layer
        with torch.no_grad(), add_hook(layers[1], hook_fn):
            output_ids = model.generate(batched_input, **gen_kwargs)

        responses = []
        for i in range(args.num_generations):
            resp = tokenizer.decode(output_ids[i][gen_input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(resp)

        # Parse and optionally judge
        parsed = [parse_oracle_output(r) for r in responses]

        if judge:
            print("Judging responses...")
            judge_results = judge.score_batch_sync(
                [prompt] * len(responses),
                [question] * len(responses),
                [p.answer for p in parsed],
            )
        else:
            # Fake judge results
            from grpo_ao.judge import JudgeResult
            judge_results = [JudgeResult(informativeness=0.5, raw_response="(skipped)", parse_success=True) for _ in responses]

        # Compute rewards
        rewards = [compute_reward(p, j.informativeness, cfg.calibration_lambda)
                   for p, j in zip(parsed, judge_results)]

        # Display
        print(f"\n{'─'*80}")
        print("RESPONSES:")
        print(f"{'─'*80}")

        for i, (resp, p, j, r) in enumerate(zip(responses, parsed, judge_results, rewards)):
            print(f"\n[{i+1}] Conf={p.confidence}/10 | Info={r.informativeness:.2f} | Brier={r.brier_score:.3f} | Reward={r.reward:.3f}")
            print(f"    Parse OK: {p.parse_success}")
            print(f"    Response: {resp[:200]}{'...' if len(resp) > 200 else ''}")
            if judge and j.raw_response != "(skipped)":
                print(f"    Judge: {j.raw_response[:150]}{'...' if len(j.raw_response) > 150 else ''}")

        # Stats for this example
        reward_vals = [r.reward for r in rewards]
        conf_vals = [r.confidence for r in rewards if r.confidence >= 0]
        info_vals = [r.informativeness for r in rewards]
        brier_vals = [r.brier_score for r in rewards]

        print(f"\n{'─'*80}")
        print("STATS FOR THIS EXAMPLE:")
        print(f"  Mean reward: {sum(reward_vals)/len(reward_vals):.3f}")
        print(f"  Reward std:  {(sum((r - sum(reward_vals)/len(reward_vals))**2 for r in reward_vals) / len(reward_vals)) ** 0.5:.3f}")
        if conf_vals:
            print(f"  Mean conf:   {sum(conf_vals)/len(conf_vals):.1f}/10")
        print(f"  Mean info:   {sum(info_vals)/len(info_vals):.3f}")
        print(f"  Mean brier:  {sum(brier_vals)/len(brier_vals):.3f}")
        print(f"  Parse fails: {sum(1 for p in parsed if not p.parse_success)}/{len(parsed)}")

        all_results.extend(rewards)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    all_rewards = [r.reward for r in all_results]
    all_conf = [r.confidence for r in all_results if r.confidence >= 0]
    all_info = [r.informativeness for r in all_results]
    all_brier = [r.brier_score for r in all_results]
    parse_fails = sum(1 for r in all_results if not r.parse_success)

    print(f"Total samples: {len(all_results)}")
    print(f"Parse failures: {parse_fails} ({100*parse_fails/len(all_results):.1f}%)")
    print(f"\nReward:  mean={sum(all_rewards)/len(all_rewards):.3f}, min={min(all_rewards):.3f}, max={max(all_rewards):.3f}")
    if all_conf:
        print(f"Conf:    mean={sum(all_conf)/len(all_conf):.1f}/10, min={min(all_conf)}, max={max(all_conf)}")
    print(f"Info:    mean={sum(all_info)/len(all_info):.3f}, min={min(all_info):.3f}, max={max(all_info):.3f}")
    print(f"Brier:   mean={sum(all_brier)/len(all_brier):.3f}")

    # Confidence distribution
    if all_conf:
        print("\nConfidence distribution:")
        for bucket in range(0, 11, 2):
            count = sum(1 for c in all_conf if bucket <= c < bucket + 2)
            bar = "█" * count
            print(f"  {bucket}-{bucket+1}: {bar} ({count})")

    print("\n" + "=" * 80)
    print("Done! Check vibes above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
