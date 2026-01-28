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
    question_temperature: float = 1.1  # High temp for diverse questions

    # ReST settings
    num_rest_rounds: int = 5
    samples_per_question: int = 5  # Oracle responses sampled per question
    oracle_temperature: float = 0.7
    filter_bottom_percent: float = 0.2  # Remove bottom 20% by reward
    calibration_lambda: float = 0.5  # λ in reward formula

    # Training settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    epochs_per_round: int = 1

    # Generation settings
    max_new_tokens: int = 150

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


# Question generation templates (10 varied templates as specified in plan)
QUESTION_TEMPLATES = [
    """Generate questions about this text.
- 3 yes/no questions (answer must be exactly "yes" or "no")
- 5 open-ended questions requiring detailed answers
- 2 questions requiring highly specific, detailed answers

TEXT:
{prompt}""",

    """What would someone want to know about this text? Generate:
- 3 binary questions with clear yes/no answers
- 5 interpretive questions about meaning or intent
- 2 questions about specific details or facts

TEXT:
{prompt}""",

    """Generate questions to test comprehension of this text:
- 3 true/false style questions (respond yes or no)
- 5 "what/why/how" questions
- 2 questions requiring nuanced, multi-part answers

TEXT:
{prompt}""",

    """Create questions about this text at different difficulty levels:
- 3 simple yes/no questions
- 5 medium-difficulty open questions
- 2 challenging questions requiring detailed analysis

TEXT:
{prompt}""",

    """Generate diverse questions:
- 3 questions answerable with yes or no only
- 5 questions about the main ideas or themes
- 2 questions about subtle details or implications

TEXT:
{prompt}""",

    """What could we ask about this text?
- 3 polar questions (yes/no answers)
- 5 wh-questions (what, who, where, when, why, how)
- 2 complex questions needing paragraph-length answers

TEXT:
{prompt}""",

    """Generate questions to probe understanding:
- 3 yes/no questions about factual content
- 5 open questions about context or interpretation
- 2 deep questions requiring inference and detail

TEXT:
{prompt}""",

    """Create a question set:
- 3 binary choice questions (yes/no format)
- 5 exploratory questions
- 2 questions that require synthesizing multiple parts of the text

TEXT:
{prompt}""",

    """Generate questions at varying specificity:
- 3 yes/no questions
- 5 questions with moderate detail expected
- 2 questions expecting comprehensive, specific answers

TEXT:
{prompt}""",

    """What should we ask about this text?
- 3 closed questions (yes or no only)
- 5 open questions about content and meaning
- 2 detailed questions requiring careful analysis

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
