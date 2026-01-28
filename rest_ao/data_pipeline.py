"""Data pipeline for ReST training.

Handles:
1. Loading diverse text prompts (RedPajama/The Pile subsets)
2. Generating questions with diverse templates
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
    """Load diverse user prompts from lmsys-chat-1m.

    Uses real user prompts from the LMSYS Chatbot Arena - actual prompts
    people sent to various LLMs. This is more representative of real use
    cases than pretraining data.

    Args:
        num_prompts: Total number of prompts to load
        max_length: Maximum character length per prompt
        seed: Random seed

    Returns:
        List of user prompts
    """
    random.seed(seed)
    prompts = []

    print(f"Loading {num_prompts} prompts from lmsys-chat-1m...")

    try:
        ds = load_dataset(
            "lmsys/lmsys-chat-1m",
            split="train",
            streaming=True,
        )

        for i, item in enumerate(tqdm(ds, desc="Loading prompts", total=num_prompts * 2)):
            if len(prompts) >= num_prompts:
                break

            # Extract user messages from conversation
            conversation = item.get("conversation", [])
            for turn in conversation:
                if turn.get("role") == "user":
                    content = turn.get("content", "")
                    # Filter by length
                    if 20 < len(content) <= max_length:
                        prompts.append(content)
                        if len(prompts) >= num_prompts:
                            break

    except Exception as e:
        print(f"Error loading lmsys-chat-1m: {e}")
        print("Falling back to simple prompts...")
        # Fallback to some basic prompts if dataset fails
        prompts = [
            "Explain how machine learning works.",
            "What is the capital of France?",
            "Write a short story about a robot.",
        ] * (num_prompts // 3)

    random.shuffle(prompts)
    prompts = prompts[:num_prompts]
    print(f"Loaded {len(prompts)} prompts")

    return prompts


def create_prompt_question_pairs(
    prompts: list[str],
    question_generator: QuestionGenerator,
    tokenizer: PreTrainedTokenizer,
    layer_percents: list[int],
    model_name: str,
    questions_per_prompt: int = 10,
) -> Iterator[PromptQuestionPair]:
    """Create prompt-question pairs with activation positions.

    Args:
        prompts: List of text prompts
        question_generator: Generator for questions
        tokenizer: Tokenizer for the model
        layer_percents: Layers to sample from (e.g., [25, 50, 75])
        model_name: Model name for layer calculation
        questions_per_prompt: Target questions per prompt

    Yields:
        PromptQuestionPair objects
    """
    from nl_probes.utils.common import layer_percent_to_layer

    layers = [layer_percent_to_layer(model_name, p) for p in layer_percents]

    for prompt in tqdm(prompts, desc="Generating questions"):
        # Generate questions
        gen_result = question_generator.generate_questions(prompt)
        questions = gen_result.questions[:questions_per_prompt]

        if not questions:
            continue

        # Tokenize the prompt for activation extraction
        context_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Use all positions after some initial context
        min_offset = max(1, len(context_ids) // 4)
        context_positions = list(range(min_offset, len(context_ids)))

        if not context_positions:
            context_positions = list(range(len(context_ids)))

        for question in questions:
            layer = random.choice(layers)

            yield PromptQuestionPair(
                prompt=prompt,
                question=question,
                layer=layer,
                context_input_ids=context_ids,
                context_positions=context_positions,
                template_idx=gen_result.template_idx,
            )
