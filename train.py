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

    # Defaults from GRPOConfig
    defaults = GRPOConfig()
    parser.add_argument("--model", type=str, default=defaults.model_name)
    parser.add_argument("--oracle_lora_path", type=str, default=defaults.oracle_lora_path)
    parser.add_argument("--num_train_steps", type=int, default=defaults.num_train_steps)
    parser.add_argument("--num_generations", type=int, default=defaults.num_generations)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--kl_penalty", type=float, default=defaults.kl_penalty)
    parser.add_argument("--calibration_lambda", type=float, default=defaults.calibration_lambda)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="grpo-activation-oracle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_at_checkpoints", action="store_true", default=True,
                        help="Run holdout evals at each checkpoint")
    parser.add_argument("--no_eval", action="store_true",
                        help="Disable holdout evals")
    parser.add_argument("--push_to_hub", action="store_true", default=True,
                        help="Push final model to HuggingFace Hub")
    parser.add_argument("--no_push", action="store_true",
                        help="Disable pushing to HuggingFace Hub")
    parser.add_argument("--hub_repo_id", type=str, default="ceselder/grpo-activation-oracle-qwen3-8b",
                        help="HuggingFace Hub repo ID")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (use if hooks break)")

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
        eval_at_checkpoints=not args.no_eval,
        push_to_hub=not args.no_push,
        hub_repo_id=args.hub_repo_id,
        use_torch_compile=not args.no_compile,
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
