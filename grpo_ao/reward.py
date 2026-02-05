"""Reward computation for GRPO training.

The reward function balances informativeness and calibration:
    reward = informativeness - λ × (certainty/10 - informativeness)²

Where:
- informativeness: 0-1 score from judge for how detailed and correct the answer is
- certainty: parsed certainty (0-10, normalized to 0-1 for Brier)
- λ: calibration weight (default 0.5)

This prevents exploitation through:
1. Vague but "correct" answers (low informativeness score)
2. Overconfident wrong answers (high Brier penalty)
"""

from dataclasses import dataclass

from grpo_ao.epistemic_status import OracleOutput


@dataclass
class RewardResult:
    """Result of reward computation for a single sample."""

    reward: float
    informativeness: float
    confidence: int  # 0-10 raw value (certainty)
    confidence_normalized: float  # 0-1 for Brier computation
    brier_score: float
    parse_success: bool

    @property
    def calibration_error(self) -> float:
        """Absolute calibration error (confidence - informativeness)."""
        return abs(self.confidence_normalized - self.informativeness)


FORMAT_PENALTY = -0.5  # Penalty for malformed output (no certainty tag)


def compute_reward(
    oracle_output: OracleOutput,
    informativeness: float,
    calibration_lambda: float = 0.5,
) -> RewardResult:
    """Compute reward for a single oracle response.

    Args:
        oracle_output: Parsed oracle output with certainty (0-10, or -1 if malformed)
        informativeness: Judge's 0-1 score for answer quality
        calibration_lambda: Weight for calibration penalty (default 0.5)

    Returns:
        RewardResult with reward and component scores
    """
    # Malformed output (no certainty tag) gets penalty
    if not oracle_output.parse_success:
        return RewardResult(
            reward=FORMAT_PENALTY,
            informativeness=informativeness,
            confidence=-1,
            confidence_normalized=0.0,
            brier_score=1.0,
            parse_success=False,
        )

    confidence = oracle_output.confidence  # 0-100
    confidence_norm = oracle_output.confidence_normalized  # 0-1
    brier = (confidence_norm - informativeness) ** 2

    reward = informativeness - calibration_lambda * brier

    return RewardResult(
        reward=reward,
        informativeness=informativeness,
        confidence=confidence,
        confidence_normalized=confidence_norm,
        brier_score=brier,
        parse_success=True,
    )


def compute_batch_rewards(
    oracle_outputs: list[OracleOutput],
    informativeness_scores: list[float],
    calibration_lambda: float = 0.5,
) -> list[RewardResult]:
    """Compute rewards for a batch of oracle responses.

    Args:
        oracle_outputs: List of parsed oracle outputs
        informativeness_scores: List of judge scores
        calibration_lambda: Weight for calibration penalty

    Returns:
        List of RewardResult objects
    """
    assert len(oracle_outputs) == len(informativeness_scores)

    return [
        compute_reward(out, info, calibration_lambda)
        for out, info in zip(oracle_outputs, informativeness_scores)
    ]


def filter_by_reward(
    samples: list,
    rewards: list[RewardResult],
    filter_bottom_percent: float = 0.2,
) -> tuple[list, list[RewardResult]]:
    """Filter out the bottom X% of samples by reward.

    Args:
        samples: List of samples (any type)
        rewards: Corresponding reward results
        filter_bottom_percent: Fraction to remove (e.g., 0.2 = remove bottom 20%)

    Returns:
        Tuple of (filtered_samples, filtered_rewards)
    """
    assert len(samples) == len(rewards)

    if not samples:
        return [], []

    # Sort by reward
    paired = list(zip(samples, rewards))
    paired.sort(key=lambda x: x[1].reward)

    # Remove bottom X%
    n_remove = int(len(paired) * filter_bottom_percent)
    filtered = paired[n_remove:]

    if not filtered:
        # Don't remove everything
        filtered = paired

    samples_out = [p[0] for p in filtered]
    rewards_out = [p[1] for p in filtered]

    return samples_out, rewards_out


def normalize_rewards_to_weights(
    rewards: list[RewardResult],
    min_weight: float = 0.1,
) -> list[float]:
    """Convert rewards to training weights normalized to [min_weight, 1.0].

    Args:
        rewards: List of reward results
        min_weight: Minimum weight to assign

    Returns:
        List of weights in [min_weight, 1.0]
    """
    if not rewards:
        return []

    raw = [r.reward for r in rewards]
    min_r = min(raw)
    max_r = max(raw)

    if max_r == min_r:
        return [1.0] * len(rewards)

    # Normalize to [0, 1], then scale to [min_weight, 1.0]
    normalized = [(r - min_r) / (max_r - min_r) for r in raw]
    weights = [min_weight + (1.0 - min_weight) * n for n in normalized]

    return weights
