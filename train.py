#!/usr/bin/env python3
"""Main entry point for ReST Activation Oracle training.

Usage:
    python train.py  # Use defaults
    python train.py --model Qwen/Qwen3-1.7B-Base --num_prompts 1000
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

from rest_ao.config import RESTConfig
from rest_ao.rest_trainer import RESTTrainer


def main():
    parser = argparse.ArgumentParser(description="Train calibrated Activation Oracle with ReST")

    # Model
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        help="Model to train (default: google/gemma-3-27b-it)")
    parser.add_argument("--oracle_lora_path", type=str,
                        default="adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it",
                        help="Path to pretrained AO checkpoint")

    # Data
    parser.add_argument("--num_prompts", type=int, default=100,
                        help="Number of prompts to use")
    parser.add_argument("--questions_per_prompt", type=int, default=20,
                        help="Questions generated per prompt")

    # ReST
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of ReST rounds")
    parser.add_argument("--samples_per_question", type=int, default=5,
                        help="Oracle responses per question")
    parser.add_argument("--calibration_lambda", type=float, default=0.5,
                        help="Weight for calibration penalty")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="rest-activation-oracle")
    parser.add_argument("--wandb_run_name", type=str, default="")

    # Misc
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    # Build config
    cfg = RESTConfig(
        model_name=args.model,
        oracle_lora_path=args.oracle_lora_path,
        num_prompts=args.num_prompts,
        questions_per_prompt=args.questions_per_prompt,
        num_rest_rounds=args.num_rounds,
        samples_per_question=args.samples_per_question,
        calibration_lambda=args.calibration_lambda,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_dir=args.save_dir,
    )

    print("=" * 60)
    print("ReST Activation Oracle Training")
    print("=" * 60)
    print(f"Model: {cfg.model_name}")
    print(f"ReST rounds: {cfg.num_rest_rounds}")
    print(f"Prompts: {cfg.num_prompts}")
    print(f"Questions per prompt: {cfg.questions_per_prompt}")
    print(f"Samples per question: {cfg.samples_per_question}")
    print(f"Calibration λ: {cfg.calibration_lambda}")
    print("=" * 60)

    # Create trainer and run
    trainer = RESTTrainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
