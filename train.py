#!/usr/bin/env python3
"""Main entry point for GRPO Activation Oracle training.

Usage:
    python train.py  # Use defaults (Qwen3-8B)
    python train.py --model google/gemma-3-27b-it --no_initial_eval
"""

import argparse
import os
from pathlib import Path

# Load .env file if it exists
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Set environment variables before imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from grpo_ao.config import GRPOConfig
from grpo_ao.grpo_trainer import GRPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train calibrated Activation Oracle with GRPO")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model to train (default: Qwen/Qwen3-8B)")
    parser.add_argument("--oracle_lora_path", type=str,
                        default="adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B",
                        help="Path to pretrained AO checkpoint")

    # Data
    parser.add_argument("--num_prompts", type=int, default=100,
                        help="Number of prompts to use")
    parser.add_argument("--questions_per_prompt", type=int, default=20,
                        help="Questions generated per prompt")

    # GRPO
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Completions per (activation, question) pair")
    parser.add_argument("--num_train_steps", type=int, default=1000,
                        help="Total training steps")
    parser.add_argument("--kl_penalty", type=float, default=0.05,
                        help="KL divergence penalty (beta)")
    parser.add_argument("--calibration_lambda", type=float, default=0.5,
                        help="Weight for calibration penalty")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)

    # Evaluation
    parser.add_argument("--no_initial_eval", action="store_true",
                        help="Skip evaluation before training starts")
    parser.add_argument("--eval_datasets", type=str, nargs="+",
                        default=["sst2", "geometry_of_truth"],
                        help="Datasets for evaluation")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="grpo-activation-oracle")
    parser.add_argument("--wandb_run_name", type=str, default="")

    # Misc
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    # Build config
    cfg = GRPOConfig(
        model_name=args.model,
        oracle_lora_path=args.oracle_lora_path,
        num_prompts=args.num_prompts,
        questions_per_prompt=args.questions_per_prompt,
        num_generations=args.num_generations,
        num_train_steps=args.num_train_steps,
        kl_penalty=args.kl_penalty,
        calibration_lambda=args.calibration_lambda,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        run_initial_eval=not args.no_initial_eval,
        eval_datasets=args.eval_datasets,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_dir=args.save_dir,
    )

    print("=" * 60)
    print("GRPO Activation Oracle Training")
    print("=" * 60)
    print(f"Model: {cfg.model_name}")
    print(f"Generations per prompt: {cfg.num_generations}")
    print(f"Training steps: {cfg.num_train_steps}")
    print(f"KL penalty (β): {cfg.kl_penalty}")
    print(f"Calibration λ: {cfg.calibration_lambda}")
    print(f"Initial eval: {cfg.run_initial_eval}")
    print("=" * 60)

    # Create trainer and run
    trainer = GRPOTrainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
