"""Judge module for informativeness scoring.

Uses an external LLM (via OpenRouter) to score oracle answers on a 0-1 scale
for informativeness and correctness.
"""

import os
import re
import asyncio
from dataclasses import dataclass
from typing import Sequence

from openai import AsyncOpenAI

from grpo_ao.config import JUDGE_PROMPT


@dataclass
class JudgeResult:
    """Result from judging a single oracle answer."""

    informativeness: float
    raw_response: str
    parse_success: bool


def _parse_score(response: str) -> float | None:
    """Parse a 0-1 score from judge response."""
    response = response.strip()

    # Try direct float parse first
    try:
        score = float(response)
        if 0.0 <= score <= 1.0:
            return score
    except ValueError:
        pass

    # Try to find a decimal number in the response
    match = re.search(r"([01]\.?\d*)", response)
    if match:
        try:
            score = float(match.group(1))
            if 0.0 <= score <= 1.0:
                return score
        except ValueError:
            pass

    return None


class InformativenessJudge:
    """Judge that scores oracle answers for informativeness."""

    def __init__(
        self,
        model: str = "qwen/qwen-2.5-72b-instruct",
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_concurrent: int = 50,
    ):
        """Initialize the judge.

        Args:
            model: Model to use via OpenRouter
            temperature: Generation temperature (0.0 for deterministic)
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            base_url: API base URL
            max_concurrent: Max concurrent API calls
        """
        self.model = model
        self.temperature = temperature
        self.max_concurrent = max_concurrent

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def _score_single(
        self,
        prompt: str,
        question: str,
        answer: str,
    ) -> JudgeResult:
        """Score a single oracle answer."""
        judge_prompt = JUDGE_PROMPT.format(
            prompt=prompt,
            question=question,
            answer=answer,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=self.temperature,
                max_tokens=10,
            )
            raw = response.choices[0].message.content or ""
            score = _parse_score(raw)

            if score is not None:
                return JudgeResult(
                    informativeness=score,
                    raw_response=raw,
                    parse_success=True,
                )
            else:
                # Default to 0.3 for unparseable responses (slight penalty)
                return JudgeResult(
                    informativeness=0.3,
                    raw_response=raw,
                    parse_success=False,
                )
        except Exception as e:
            return JudgeResult(
                informativeness=0.3,
                raw_response=f"Error: {e}",
                parse_success=False,
            )

    async def score_batch(
        self,
        prompts: Sequence[str],
        questions: Sequence[str],
        answers: Sequence[str],
    ) -> list[JudgeResult]:
        """Score a batch of oracle answers.

        Args:
            prompts: Original text prompts
            questions: Questions asked
            answers: Oracle's answers (without epistemic status prefix)

        Returns:
            List of JudgeResult objects
        """
        assert len(prompts) == len(questions) == len(answers)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded_score(p, q, a):
            async with semaphore:
                return await self._score_single(p, q, a)

        tasks = [
            bounded_score(p, q, a)
            for p, q, a in zip(prompts, questions, answers)
        ]

        return await asyncio.gather(*tasks)

    def score_batch_sync(
        self,
        prompts: Sequence[str],
        questions: Sequence[str],
        answers: Sequence[str],
    ) -> list[JudgeResult]:
        """Synchronous wrapper for score_batch."""
        return asyncio.run(self.score_batch(prompts, questions, answers))


class LocalJudge:
    """Judge using a local model (same model as oracle, for efficiency) - BATCHED."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 16,
    ):
        """Initialize with a loaded model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def score_batch_sync(
        self,
        prompts: Sequence[str],
        questions: Sequence[str],
        answers: Sequence[str],
    ) -> list[JudgeResult]:
        """Score using the local model - BATCHED."""
        import torch
        from tqdm import tqdm

        results = []

        for batch_start in tqdm(range(0, len(prompts), self.batch_size), desc="Judging"):
            batch_prompts = prompts[batch_start:batch_start + self.batch_size]
            batch_questions = questions[batch_start:batch_start + self.batch_size]
            batch_answers = answers[batch_start:batch_start + self.batch_size]

            # Build batch inputs
            batch_input_ids = []
            max_len = 0

            for prompt, question, answer in zip(batch_prompts, batch_questions, batch_answers):
                judge_prompt = JUDGE_PROMPT.format(
                    prompt=prompt,
                    question=question,
                    answer=answer,
                )

                messages = [{"role": "user", "content": judge_prompt}]
                encoded = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
                )
                input_ids = encoded["input_ids"][0]
                batch_input_ids.append(input_ids)
                max_len = max(max_len, len(input_ids))

            # Left-pad
            padded_input_ids = []
            attention_masks = []

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
                attention_masks.append(attn_mask)

            input_ids_tensor = torch.stack(padded_input_ids).to(self.device)
            attention_mask = torch.stack(attention_masks).to(self.device)

            # Batched generation
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode and parse
            for i in range(len(batch_prompts)):
                generated = outputs[i][input_ids_tensor.shape[1]:]
                response = self.tokenizer.decode(generated, skip_special_tokens=True)

                score = _parse_score(response)
                if score is not None:
                    results.append(JudgeResult(
                        informativeness=score,
                        raw_response=response,
                        parse_success=True,
                    ))
                else:
                    results.append(JudgeResult(
                        informativeness=0.3,
                        raw_response=response,
                        parse_success=False,
                    ))

        return results
