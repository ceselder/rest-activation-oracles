"""Question generation from prompts using diverse templates."""

import random
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from rest_ao.config import QUESTION_TEMPLATES


@dataclass
class GeneratedQuestions:
    """Container for questions generated from a prompt."""

    prompt: str
    questions: list[str]
    template_idx: int


def parse_questions(raw_output: str) -> list[str]:
    """Parse individual questions from model output.

    Handles various formats:
    - Numbered lists (1., 2., etc.)
    - Bulleted lists (-, *, etc.)
    - Line-separated questions
    """
    lines = raw_output.strip().split("\n")
    questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common prefixes
        # Numbered: "1.", "1)", "1:"
        import re
        line = re.sub(r"^\d+[\.\)\:]?\s*", "", line)
        # Bulleted: "-", "*", "•"
        line = re.sub(r"^[\-\*\•]\s*", "", line)

        line = line.strip()

        # Skip if it's a category header (e.g., "Yes/No Questions:")
        if line.endswith(":") and len(line.split()) <= 5:
            continue

        # Skip if too short to be a real question
        if len(line) < 10:
            continue

        # Keep if it looks like a question or statement
        questions.append(line)

    return questions


class QuestionGenerator:
    """Generate diverse questions about prompts using LLM."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        temperature: float = 1.1,
        max_new_tokens: int = 500,
        device: str = "cuda",
    ):
        """Initialize question generator.

        Args:
            model: The LLM to use for generation
            tokenizer: Corresponding tokenizer
            temperature: High temperature for diversity (1.0-1.2 recommended)
            max_new_tokens: Max tokens for question generation
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device

    def generate_questions(
        self,
        prompt: str,
        template_idx: int | None = None,
    ) -> GeneratedQuestions:
        """Generate questions for a single prompt.

        Args:
            prompt: The text to generate questions about
            template_idx: Specific template to use, or None for random

        Returns:
            GeneratedQuestions with parsed questions
        """
        if template_idx is None:
            template_idx = random.randint(0, len(QUESTION_TEMPLATES) - 1)

        template = QUESTION_TEMPLATES[template_idx]
        full_prompt = template.format(prompt=prompt)

        messages = [{"role": "user", "content": full_prompt}]

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        questions = parse_questions(response)

        return GeneratedQuestions(
            prompt=prompt,
            questions=questions,
            template_idx=template_idx,
        )

    def generate_questions_batch(
        self,
        prompts: list[str],
        rotate_templates: bool = True,
    ) -> list[GeneratedQuestions]:
        """Generate questions for multiple prompts.

        Args:
            prompts: List of text prompts
            rotate_templates: Whether to cycle through templates

        Returns:
            List of GeneratedQuestions
        """
        results = []
        for i, prompt in enumerate(prompts):
            if rotate_templates:
                template_idx = i % len(QUESTION_TEMPLATES)
            else:
                template_idx = None

            result = self.generate_questions(prompt, template_idx)
            results.append(result)

        return results
