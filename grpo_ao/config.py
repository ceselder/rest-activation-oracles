"""Configuration for GRPO Activation Oracle training."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GRPOConfig:
    """Configuration for GRPO training of calibrated Activation Oracles."""

    # Model settings
    model_name: str = "Qwen/Qwen3-8B"
    oracle_lora_path: str | None = "adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B"
    hook_layer: int = 1  # Layer to inject activations (paper uses layer 1)
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])  # Extract at multiple depths

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Data settings
    num_prompts: int = 100
    questions_per_prompt: int = 20
    question_temperature: float = 1.5
    question_batch_size: int = 16
    grow_batch_size: int = 8
    judge_batch_size: int = 32

    # GRPO settings
    num_generations: int = 8  # G completions per (activation, question)
    kl_penalty: float = 0.05  # β for KL divergence penalty
    calibration_lambda: float = 0.5  # λ in reward formula
    oracle_temperature: float = 0.9  # For diverse samples

    # Training settings
    num_train_steps: int = 1000
    checkpoint_every: int = 200
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6  # Lower than SFT for RL stability
    max_grad_norm: float = 1.0

    # Generation settings
    max_new_tokens: int = 80

    # Logging
    log_samples_every: int = 50  # Log sample generations every N samples
    log_samples_count: int = 3   # Number of samples to show each time

    # Evaluation
    run_initial_eval: bool = True  # Run eval before training starts
    eval_datasets: list[str] = field(default_factory=lambda: ["sst2", "geometry_of_truth"])
    eval_samples_per_dataset: int = 50

    # Logging
    wandb_project: str = "grpo-activation-oracle"
    wandb_run_name: str = ""

    # Paths
    save_dir: str = "checkpoints"
    cache_dir: str = "cache"

    # Misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Judge settings (Gemini Flash Lite via OpenRouter - cheap and fast)
    judge_model: str = "google/gemini-2.5-flash-lite"
    judge_temperature: float = 0.0
    use_external_judge: bool = True  # Use OpenRouter instead of local model

    def __post_init__(self):
        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]
            self.wandb_run_name = f"grpo_{model_short}_g{self.num_generations}"


# Question generation templates
QUESTION_TEMPLATES = [
    """Generate 20 questions about this text. Each question must end with a question mark.

RULES:
- First question MUST be: "What is the model thinking about?"
- Mix yes/no questions with open-ended questions
- Questions should probe: topic, sentiment, user intent, author traits, tone
- Keep questions short and clear
- Every line must be a question ending with ?

Example good questions:
1. What is the model thinking about?
2. Is this text about technology?
3. What is the main topic?
4. Does the user seem frustrated?
5. Is this a request for help?
6. What can you infer about the author?
7. Is the tone formal or casual?
8. Is there code in this text?

TEXT:
{prompt}

Generate 20 questions (numbered 1-20), each ending with ?""",

    """Write 20 short questions about this text.

IMPORTANT:
- Question 1 must be: "What is the model thinking about?"
- Each question must end with a ?
- Mix of yes/no and open-ended
- Probe: themes, sentiment, intent, author background

TEXT:
{prompt}

Questions (1-20):""",

    """Create 20 questions to probe what this text is about.

Format: numbered list, each ending with ?
First question: "What is the model thinking about?"

Topics to cover:
- Overall theme/content (open-ended)
- Specific topic checks (yes/no)
- User sentiment and mood
- Author expertise level
- Writing style and tone

TEXT:
{prompt}

List 20 questions:""",
]


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
