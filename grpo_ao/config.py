"""Configuration for GRPO Activation Oracle training."""

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """GRPO training configuration."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    oracle_lora_path: str | None = "adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B"
    hook_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])

    # LoRA (if no pretrained checkpoint)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # GRPO
    num_train_steps: int = 1000
    num_generations: int = 8  # G responses per (prompt, question)
    kl_penalty: float = 0.05
    calibration_lambda: float = 0.5
    oracle_temperature: float = 0.9

    # Training
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    max_new_tokens: int = 80

    # Logging
    log_samples_every: int = 10
    checkpoint_every: int = 100
    wandb_project: str = "grpo-activation-oracle"
    wandb_run_name: str = ""

    # Paths
    save_dir: str = "checkpoints"

    # Judge (Gemini Flash Lite via OpenRouter)
    judge_model: str = "google/gemini-2.0-flash-lite-001"
    judge_temperature: float = 0.0

    # Misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    def __post_init__(self):
        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]
            self.wandb_run_name = f"grpo_{model_short}_g{self.num_generations}"


JUDGE_PROMPT = """Rate how informative and correct this answer is on a scale from 0.0 to 1.0.

ORIGINAL TEXT:
{prompt}

QUESTION: {question}

ANSWER: {answer}

Scoring guidelines:
- 0.0 = Completely wrong or nonsensical
- 0.2 = Technically not wrong but extremely vague (e.g., "it's about science")
- 0.4 = Partially correct with minimal detail
- 0.6 = Correct with reasonable detail
- 0.8 = Correct with good specificity and relevant details
- 1.0 = Correct, highly specific, captures nuance and key information

Consider both correctness AND level of detail. A vague answer that is technically true should score low.

Return only a single decimal number between 0.0 and 1.0."""
