"""Question generation from prompts using diverse templates - BATCHED."""

import random
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from grpo_ao.config import QUESTION_TEMPLATES


@dataclass
class GeneratedQuestions:
    """Container for questions generated from a prompt."""

    prompt: str
    questions: list[str]
    template_idx: int


def parse_questions(raw_output: str) -> list[str]:
    """Parse individual questions from model output."""
    import re
    lines = raw_output.strip().split("\n")
    questions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove common prefixes
        line = re.sub(r"^\d+[\.\)\:]?\s*", "", line)
        line = re.sub(r"^[\-\*\•]\s*", "", line)
        line = line.strip()

        # Skip headers
        if line.endswith(":") and len(line.split()) <= 5:
            continue

        # Skip if too short
        if len(line) < 10:
            continue

        questions.append(line)

    return questions


class QuestionGenerator:
    """Generate diverse questions about prompts using LLM - BATCHED."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        temperature: float = 1.1,
        max_new_tokens: int = 500,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.batch_size = batch_size

        # Ensure pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _generate_batch(
        self,
        batch_prompts: list[str],
        batch_template_idxs: list[int],
    ) -> list[GeneratedQuestions]:
        """Generate questions for a single batch."""
        # Build batch inputs
        batch_messages = []
        for i, prompt in enumerate(batch_prompts):
            template = QUESTION_TEMPLATES[batch_template_idxs[i]]
            full_prompt = template.format(prompt=prompt)
            batch_messages.append([{"role": "user", "content": full_prompt}])

        # Tokenize batch with padding
        batch_input_ids = []
        max_len = 0

        for messages in batch_messages:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"][0]
            batch_input_ids.append(input_ids)
            max_len = max(max_len, len(input_ids))

        # Left-pad to max length
        padded_input_ids = []
        padded_attention_masks = []

        for input_ids in batch_input_ids:
            pad_len = max_len - len(input_ids)
            padded = torch.cat([
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
                input_ids
            ])
            attn_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(input_ids), dtype=torch.long)
            ])
            padded_input_ids.append(padded)
            padded_attention_masks.append(attn_mask)

        input_ids = torch.stack(padded_input_ids).to(self.device)
        attention_mask = torch.stack(padded_attention_masks).to(self.device)

        # Batched generation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode each output
        results = []
        for i, (prompt, template_idx) in enumerate(zip(batch_prompts, batch_template_idxs)):
            generated = outputs[i][input_ids.shape[1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            questions = parse_questions(response)
            results.append(GeneratedQuestions(
                prompt=prompt,
                questions=questions,
                template_idx=template_idx,
            ))

        return results

    def generate_questions_batch(
        self,
        prompts: list[str],
        rotate_templates: bool = True,
    ) -> list[GeneratedQuestions]:
        """Generate questions for multiple prompts with ADAPTIVE BATCHED inference."""
        results = []
        current_batch_size = self.batch_size

        batch_start = 0
        pbar = tqdm(total=len(prompts), desc="Generating questions")

        while batch_start < len(prompts):
            batch_prompts = prompts[batch_start:batch_start + current_batch_size]
            batch_template_idxs = []

            for i, prompt in enumerate(batch_prompts):
                global_idx = batch_start + i
                if rotate_templates:
                    template_idx = global_idx % len(QUESTION_TEMPLATES)
                else:
                    template_idx = random.randint(0, len(QUESTION_TEMPLATES) - 1)
                batch_template_idxs.append(template_idx)

            try:
                batch_results = self._generate_batch(batch_prompts, batch_template_idxs)
                results.extend(batch_results)
                pbar.update(len(batch_prompts))
                batch_start += current_batch_size
                # Try to increase batch size after success (up to original)
                if current_batch_size < self.batch_size:
                    current_batch_size = min(current_batch_size * 2, self.batch_size)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"\n  OOM! Reducing batch size to {current_batch_size}")
                else:
                    # Skip this prompt if even batch_size=1 fails
                    print(f"\n  OOM even at batch_size=1, skipping prompt")
                    results.append(GeneratedQuestions(
                        prompt=batch_prompts[0],
                        questions=[],
                        template_idx=batch_template_idxs[0],
                    ))
                    batch_start += 1
                    pbar.update(1)

        pbar.close()
        return results

    def generate_questions(
        self,
        prompt: str,
        template_idx: int | None = None,
    ) -> GeneratedQuestions:
        """Generate questions for a single prompt (calls batched version)."""
        if template_idx is None:
            template_idx = random.randint(0, len(QUESTION_TEMPLATES) - 1)

        # Use batch of 1
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

        return GeneratedQuestions(
            prompt=prompt,
            questions=parse_questions(response),
            template_idx=template_idx,
        )
