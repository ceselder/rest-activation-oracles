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
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])  # Extract at multiple depths

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # Data settings
    num_prompts: int = 10_000  # Number of prompts to use
    questions_per_prompt: int = 10  # Questions generated per prompt
    question_temperature: float = 1.75  # Very high temp for maximum question diversity
    question_batch_size: int = 32  # Batch size for question generation (variable prompt lengths)
    grow_batch_size: int = 32  # Batch size for GROW phase
    judge_batch_size: int = 32  # Batch size for SCORE phase

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
# AO sees only activations (not raw text) and must answer questions about the content
# Questions should test factual recall, not require external knowledge
# All templates require English output regardless of input language
_AO_CONTEXT = """You are generating questions to train an Activation Oracle - a model that answers questions about text using only internal activations (not the raw text).

Good questions:
- Test factual content recall (who, what, where, when)
- Have clear yes/no or short factual answers
- Can be answered from the text alone

Bad questions:
- Require external knowledge or opinions
- Are vague or philosophical
- Are not actually questions (statements)

"""

QUESTION_TEMPLATES = [
    _AO_CONTEXT + """Generate questions about this text:
- 2-3 yes/no questions (e.g., "Is X mentioned?", "Does the text say Y?")
- 3-4 factual questions (e.g., "What is X?", "Who did Y?", "Where is Z?")
IMPORTANT: English only. Must be answerable from the text.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Quiz questions for this text:
- 2-3 yes/no verification questions
- 2-3 "what/who/where/when" factual questions
Keep questions simple and directly answerable from the text.
IMPORTANT: English only.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Create factual recall questions:
- Yes/no: "Is X true?", "Does Y happen?", "Is Z mentioned?"
- Open: "What is X?", "Who does Y?", "How many Z?"
IMPORTANT: English only. No opinions or external knowledge.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Write questions testing text comprehension:
- Binary checks (yes/no answers only)
- Short-answer factual questions
Questions must be answerable using only the text content.
IMPORTANT: English only.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Generate reading comprehension questions:
- 2-3 true/false style (answerable yes or no)
- 2-3 factual retrieval (who, what, where, when, how many)
IMPORTANT: English only. Clear, unambiguous questions.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Create questions about the content:
- Yes/no questions checking specific facts
- Wh-questions (what/who/where/when) with factual answers
No philosophical or opinion questions.
IMPORTANT: English only.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Write fact-checking questions:
- "Is it true that X?" (yes/no)
- "What is the X mentioned?" (factual)
- "Who/where/when is Y?" (factual)
IMPORTANT: English only. Text-answerable only.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Generate simple comprehension questions:
- 2-4 yes/no questions about facts in the text
- 2-4 short-answer questions about details
IMPORTANT: English only.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Create questions a reader could answer:
- Verification questions (yes/no)
- Detail questions (specific facts from text)
Avoid vague or subjective questions.
IMPORTANT: English only.

TEXT:
{prompt}""",

    _AO_CONTEXT + """Write clear factual questions:
- Yes/no: test if specific things are mentioned/true
- Open: ask for specific details (names, numbers, places)
IMPORTANT: English only. No external knowledge needed.

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
