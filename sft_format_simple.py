#!/usr/bin/env python3
"""Simple SFT to teach the oracle the [epistemic status: X] format.

Uses synthetic data - no activation steering needed, just format learning.
"""

import argparse
import json
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
import wandb
from peft import PeftModel
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import sys
for path in [
    str(Path(__file__).parent.parent / "activation_oracles"),
    "/root/activation_oracles",
]:
    if Path(path).exists():
        sys.path.insert(0, path)
        break

from nl_probes.utils.common import load_tokenizer, set_seed


def load_dataset(path: str) -> list[dict]:
    """Load JSONL dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_messages(question: str, response: str):
    """Format as chat messages - NO system prompt (matches oracle training)."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]


def main():
    parser = argparse.ArgumentParser(description="SFT for epistemic status format")
    parser.add_argument("--dataset", type=str, default="datasets/sft_format_dataset.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--oracle_lora", type=str,
                        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B")
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--save_path", type=str, default="checkpoints/sft_format")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="grpo-activation-oracle")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if not Path(args.dataset).exists():
        print(f"Dataset not found! Run: python generate_sft_dataset.py --output {args.dataset}")
        return
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")

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
    is_qwen3 = "qwen3" in model_lower or "qwen-3" in model_lower

    # Load oracle LoRA
    if args.oracle_lora:
        print(f"Loading LoRA: {args.oracle_lora}")
        model = PeftModel.from_pretrained(model, args.oracle_lora, is_trainable=True)

    model.enable_input_require_grads()
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Wandb
    wandb.init(
        project=args.wandb_project,
        name=f"sft_format_{args.model.split('/')[-1]}",
        config=vars(args),
    )

    print(f"\nTraining for {args.num_steps} steps...")
    pbar = tqdm(range(args.num_steps), desc="SFT")

    for step in pbar:
        # Sample random example
        ex = random.choice(dataset)

        # Format messages - NO system prompt (matches oracle training)
        messages = format_messages(
            ex["question"],
            ex["formatted_response"],
        )

        # Encode full conversation
        try:
            full_encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors=None,
                enable_thinking=False if is_qwen3 else None,
            )
        except TypeError:
            full_encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors=None
            )

        full_ids = full_encoded.input_ids if hasattr(full_encoded, 'input_ids') else list(full_encoded)
        input_ids = torch.tensor([full_ids], device=device)

        # Encode prompt only (to create labels mask)
        try:
            prompt_encoded = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors=None,
                enable_thinking=False if is_qwen3 else None,
            )
        except TypeError:
            prompt_encoded = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors=None
            )

        prompt_ids = prompt_encoded.input_ids if hasattr(prompt_encoded, 'input_ids') else list(prompt_encoded)

        # Create labels - mask out the prompt
        labels = input_ids.clone()
        labels[0, :len(prompt_ids)] = -100

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"sft/loss": loss.item(), "sft/step": step})
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % 50 == 0:
            torch.cuda.empty_cache()

    # Save
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nSaved to {save_path}")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    main()
