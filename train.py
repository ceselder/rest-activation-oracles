#!/usr/bin/env python3
"""GRPO Activation Oracle training.

Usage:
    python train.py
    python train.py --num_train_steps 500 --num_generations 4
"""

import argparse
import os
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

from grpo_ao.config import GRPOConfig
from grpo_ao.grpo_trainer import GRPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train calibrated Activation Oracle with GRPO")

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--oracle_lora_path", type=str,
                        default="adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B")
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl_penalty", type=float, default=0.05)
    parser.add_argument("--calibration_lambda", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="grpo-activation-oracle")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cfg = GRPOConfig(
        model_name=args.model,
        oracle_lora_path=args.oracle_lora_path,
        num_train_steps=args.num_train_steps,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        kl_penalty=args.kl_penalty,
        calibration_lambda=args.calibration_lambda,
        log_samples_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        wandb_project=args.wandb_project,
        seed=args.seed,
    )

    print("=" * 60)
    print("GRPO Activation Oracle Training")
    print("=" * 60)
    print(f"Model: {cfg.model_name}")
    print(f"Steps: {cfg.num_train_steps}")
    print(f"Generations per step: {cfg.num_generations}")
    print(f"LR: {cfg.learning_rate}")
    print(f"Calibration λ: {cfg.calibration_lambda}")
    print("=" * 60)

    trainer = GRPOTrainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
