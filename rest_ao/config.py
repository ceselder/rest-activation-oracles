"""Configuration for ReST Activation Oracle training."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RESTConfig:
    """Configuration for Reinforced Self-Training of calibrated Activation Oracles."""

    # Model settings
    model_name: str = "Qwen/Qwen3-8B"
    oracle_lora_path: str | None = "adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B"
    hook_layer: int = 1  # Layer to inject activations (paper uses layer 1)
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Data settings
    num_prompts: int = 10_000  # Number of prompts to use
    questions_per_prompt: int = 10  # Questions generated per prompt
    question_temperature: float = 1.75  # Very high temp for maximum question diversity
    question_batch_size: int = 1024  # Batch size for question generation (A100 80GB)
    grow_batch_size: int = 1024  # Batch size for GROW phase (A100 80GB)
    judge_batch_size: int = 1024  # Batch size for SCORE phase (A100 80GB)

    # ReST settings
    num_rest_rounds: int = 5
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


# Question generation templates - varied phrasing, 2-4 yes/no + 2-4 open-ended, mixed difficulty
QUESTION_TEMPLATES = [
    """Read this text and create questions of varying types and difficulty:
- 2-3 binary questions (must be answerable with exactly "yes" or "no")
- 3-4 open-ended questions (some easy, some requiring careful thought)
Make each question distinct and interesting.

TEXT:
{prompt}""",

    """Imagine you're quizzing someone on this passage. Write:
- A couple quick yes/no checks
- Several deeper questions that need real explanation
- At least one tricky question that requires reading between the lines

TEXT:
{prompt}""",

    """Create a diverse question set for this text:
- 2-4 polar questions (yes/no only)
- 2-4 exploratory questions ranging from straightforward to challenging
Vary the phrasing and focus of each question.

TEXT:
{prompt}""",

    """What would you ask to test understanding? Generate:
- Some simple verification questions (answerable yes or no)
- Some interpretive questions requiring elaboration
- Mix easy and difficult ones

TEXT:
{prompt}""",

    """Formulate questions about this content:
- Binary questions for quick fact-checking (2-3)
- Open questions probing comprehension at different depths (3-4)
Each question should stand alone and not repeat others.

TEXT:
{prompt}""",

    """Design a question battery for this text:
- Include yes/no items to verify basic facts
- Include wh-questions (what/why/how/who) of varying complexity
- Range from surface-level to requiring inference

TEXT:
{prompt}""",

    """Craft questions to explore this passage:
- A few closed-form queries (yes or no answers only)
- Several open-ended prompts, some simple, some requiring synthesis
Write each in a different style.

TEXT:
{prompt}""",

    """Generate a mix of question types:
- 2-4 that can be answered with just "yes" or "no"
- 2-4 that need fuller responses, varying in difficulty
Avoid repetitive phrasing.

TEXT:
{prompt}""",

    """Build questions for this text at multiple levels:
- Factual checks (yes/no format)
- Analytical questions requiring thought
- At least one that's genuinely hard
Make them sound natural and varied.

TEXT:
{prompt}""",

    """Survey questions for this content:
- Quick binary items (yes/no)
- Deeper probes needing explanation
- Include both easy wins and real challenges
Each question should feel fresh.

TEXT:
{prompt}""",
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
