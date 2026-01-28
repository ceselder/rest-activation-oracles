"""Configuration for ReST Activation Oracle training."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RESTConfig:
    """Configuration for Reinforced Self-Training of calibrated Activation Oracles."""

    # Model settings
    model_name: str = "google/gemma-3-27b-it"
    oracle_lora_path: str | None = "adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it"
    hook_layer: int = 1  # Layer to inject activations (paper uses layer 1)
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])  # Extract at multiple depths

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Data settings
    num_prompts: int = 100  # Fewer prompts for faster feedback loops
    questions_per_prompt: int = 20  # More questions per prompt
    question_temperature: float = 1.5  # High diversity, still coherent
    question_batch_size: int = 8  # Start small, adaptive batching will adjust
    grow_batch_size: int = 4  # Smaller for 27B model
    judge_batch_size: int = 16  # Judge is external API, can be larger

    # ReST settings
    num_rest_rounds: int = 10  # Total rounds to run
    checkpoint_every: int = 2  # Save checkpoint every N rounds
    samples_per_question: int = 3  # Oracle responses sampled per question
    oracle_temperature: float = 1.2  # Higher for diverse samples (was 0.7)
    filter_bottom_percent: float = 0.2  # Remove bottom 20% by reward
    calibration_lambda: float = 0.5  # λ in reward formula

    # Training settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    epochs_per_round: int = 1

    # Generation settings
    max_new_tokens: int = 80  # Reduced from 150 for speed (epistemic status + short answer)

    # Logging
    wandb_project: str = "rest-activation-oracle"
    wandb_run_name: str = ""

    # Paths
    save_dir: str = "checkpoints"
    cache_dir: str = "cache"

    # Misc
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Judge settings (OpenRouter for Qwen-32B or similar)
    judge_model: str = "qwen/qwen-2.5-72b-instruct"  # Via OpenRouter
    judge_temperature: float = 0.0

    def __post_init__(self):
        if not self.wandb_run_name:
            model_short = self.model_name.split("/")[-1]
            self.wandb_run_name = f"rest_{model_short}_r{self.num_rest_rounds}"


# Question generation templates for Activation Oracle training
# Keep it simple - just ask for a numbered list of questions
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
