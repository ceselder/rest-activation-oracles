"""Configuration for GRPO Activation Oracle training."""

from dataclasses import dataclass, field


@dataclass
class GRPOConfig:
    """GRPO training configuration."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    oracle_lora_path: str | None = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
    hook_layer: int = 1
    layer_percents: list[int] = field(default_factory=lambda: [50])  # Only middle layer

    # LoRA (if no pretrained checkpoint)
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # GRPO
    num_train_steps: int = 1000
    num_generations: int = 8  # Reduced - no gradient checkpointing due to hook conflict
    examples_per_batch: int = 2  # 8×2=16 rollouts per batch (fits in 140GB without checkpointing)
    kl_penalty: float = 0.04
    calibration_lambda: float = 0.75  # Brier score weight
    oracle_temperature: float = 1.0
    # Dr. GRPO: "none" = don't scale by std (recommended), "group" = original GRPO
    scale_rewards: str = "none"
    # Dr. GRPO length bias fix: normalize by global constant (max_tokens * G) not response length
    fix_length_bias: bool = True

    # Training
    learning_rate: float = 3e-6
    max_grad_norm: float = 1.0
    max_new_tokens: int = 300  # Allow longer AO responses
    gradient_accumulation_steps: int = 4  # Compensate for smaller per-step batch

    # Logging
    log_samples_every: int = 10
    checkpoint_every: int = 100
    wandb_project: str = "grpo-activation-oracle"
    wandb_run_name: str = ""

    # Paths
    save_dir: str = "checkpoints"
    push_to_hub: bool = True
    hub_repo_id: str = "ceselder/grpo-activation-oracle-qwen3-8b"

    # Judge (GLM-4.7 Flash via OpenRouter - fast)
    judge_model: str = "z-ai/glm-4.7-flash"
    judge_max_tokens: int = 5000  # Full CoT
    judge_thinking_level: str = "low"  # minimal/low/medium/high
    judge_temperature: float = 0.5

    # Evaluation
    eval_at_checkpoints: bool = True
    eval_datasets: list[str] | None = None  # None = all 9 datasets
    eval_batch_size: int = 4

    # Misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_torch_compile: bool = True  # torch.compile for faster generation (experimental)

    def __post_init__(self):
        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]
            self.wandb_run_name = f"grpo_{model_short}_g{self.num_generations}"


JUDGE_PROMPT = """You are evaluating an Activation Oracle (AO) - a model that reads the internal activations (hidden states) of another model while it processes text, and answers questions about what the model is "thinking".

The AO has been shown the activations from a model processing this TEXT:
{prompt}

The AO was asked this QUESTION about the model's internal state:
{question}

The AO gave this ANSWER:
{answer}

Rate how informative and correct this answer is. Score from 0 to 100:
- 0-20: Confidently wrong, nonsensical, or completely unrelated
- 20-40: Wrong or vague hedge that could apply to anything
- 40-60: Partially correct but missing key details or too generic
- 60-80: Correct with relevant specific details
- 80-100: Excellent - specific, insightful, captures nuances about model's processing

Think through your reasoning, then output your final score as an integer on the last line."""
