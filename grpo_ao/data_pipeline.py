"""Data pipeline for ReST training - BATCHED.

Handles:
1. Loading diverse text prompts (lmsys-chat-1m)
2. Generating questions with diverse templates (BATCHED)
3. Extracting activations
4. Creating training samples
"""

import random
from dataclasses import dataclass, field
from typing import Iterator

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from grpo_ao.question_generation import QuestionGenerator, GeneratedQuestions


@dataclass
class PromptQuestionPair:
    """A prompt with its associated question and activation info."""

    prompt: str
    question: str
    layer: int
    context_input_ids: list[int]
    context_positions: list[int]
    template_idx: int


@dataclass
class OracleSample:
    """A sample for oracle training/generation."""

    prompt: str
    question: str
    oracle_response: str  # Full response with epistemic status
    informativeness: float | None = None
    reward: float | None = None


def load_diverse_prompts(
    num_prompts: int = 10000,
    max_length: int = 4096,
    seed: int = 42,
) -> list[str]:
    """Load diverse ENGLISH user questions from WildChat."""
    random.seed(seed)
    prompts = []

    print(f"Loading {num_prompts} English prompts from WildChat...")

    # Load more than needed so we can shuffle and sample
    load_target = num_prompts * 3

    ds = load_dataset(
        "allenai/WildChat-1M",
        split="train",
        streaming=True,
    )

    for item in tqdm(ds, desc="WildChat (English)", total=load_target):
        if len(prompts) >= load_target:
            break
        # Filter for English only
        if item.get("language") != "English":
            continue
        conversation = item.get("conversation", [])
        for turn in conversation:
            if turn.get("role") == "user":
                content = turn.get("content", "").strip()
                if 20 < len(content) <= max_length:
                    prompts.append(content)
                    break  # Only first user message per convo

    # Shuffle and sample
    random.shuffle(prompts)
    prompts = prompts[:num_prompts]
    print(f"Loaded {len(prompts)} English prompts from WildChat")

    return prompts


def create_prompt_question_pairs(
    prompts: list[str],
    question_generator: QuestionGenerator,
    tokenizer: PreTrainedTokenizer,
    layer_percents: list[int],
    model_name: str,
    questions_per_prompt: int = 10,
) -> list[PromptQuestionPair]:
    """Create prompt-question pairs with activation positions - BATCHED.

    Args:
        prompts: List of text prompts
        question_generator: Generator for questions (uses batched generation)
        tokenizer: Tokenizer for the model
        layer_percents: Layers to sample from (e.g., [25, 50, 75])
        model_name: Model name for layer calculation
        questions_per_prompt: Target questions per prompt

    Returns:
        List of PromptQuestionPair objects
    """
    from nl_probes.utils.common import layer_percent_to_layer

    layers = [layer_percent_to_layer(model_name, p) for p in layer_percents]

    # BATCHED question generation - this is the key speedup
    print(f"Generating questions for {len(prompts)} prompts (batched)...")
    all_gen_results = question_generator.generate_questions_batch(prompts)

    pairs = []
    for prompt, gen_result in zip(prompts, all_gen_results):
        questions = gen_result.questions[:questions_per_prompt]

        if not questions:
            continue

        # Tokenize the prompt for activation extraction
        context_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Use positions after some initial context
        min_offset = max(1, len(context_ids) // 4)
        context_positions = list(range(min_offset, len(context_ids)))

        if not context_positions:
            context_positions = list(range(len(context_ids)))

        for question in questions:
            layer = random.choice(layers)

            pairs.append(PromptQuestionPair(
                prompt=prompt,
                question=question,
                layer=layer,
                context_input_ids=context_ids,
                context_positions=context_positions,
                template_idx=gen_result.template_idx,
            ))

    print(f"Created {len(pairs)} prompt-question pairs")
    return pairs
