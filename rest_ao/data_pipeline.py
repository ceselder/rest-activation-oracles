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

from rest_ao.question_generation import QuestionGenerator, GeneratedQuestions


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
    max_length: int = 1024,
    seed: int = 42,
) -> list[str]:
    """Load diverse prompts from FineWeb (pretraining) + LMSYS Chat-1M (conversational).

    Following the paper: equal mix of both datasets.
    """
    random.seed(seed)
    prompts = []
    half = num_prompts // 2

    # Load FineWeb (pretraining data) - not gated
    print(f"Loading {half} prompts from FineWeb...")
    try:
        fineweb = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        for item in tqdm(fineweb, desc="FineWeb", total=half * 2):
            if len(prompts) >= half:
                break
            text = item.get("text", "")
            # Take a reasonable chunk
            if 50 < len(text) <= max_length:
                prompts.append(text)
    except Exception as e:
        print(f"FineWeb failed: {e}")

    # Load LMSYS Chat-1M (conversational) - gated but we have token
    print(f"Loading {half} prompts from LMSYS Chat-1M...")
    lmsys_prompts = []
    try:
        lmsys = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            streaming=True,
        )
        for item in tqdm(lmsys, desc="LMSYS", total=half * 2):
            if len(lmsys_prompts) >= half:
                break
            conversation = item.get("conversation", [])
            for turn in conversation:
                if turn.get("role") == "user":
                    content = turn.get("content", "")
                    if 20 < len(content) <= max_length:
                        lmsys_prompts.append(content)
                        if len(lmsys_prompts) >= half:
                            break
        prompts.extend(lmsys_prompts)
    except Exception as e:
        print(f"LMSYS failed: {e}, using only FineWeb")

    random.shuffle(prompts)
    prompts = prompts[:num_prompts]
    print(f"Loaded {len(prompts)} prompts ({len(prompts) - len(lmsys_prompts)} FineWeb, {len(lmsys_prompts)} LMSYS)")

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
