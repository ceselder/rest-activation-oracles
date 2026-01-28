"""Calibration evaluation for ReST-trained Activation Oracle.

Measures:
1. Standard accuracy on oracle tasks
2. ECE (Expected Calibration Error)
3. Brier score
4. Reliability diagrams
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from grpo_ao.epistemic_status import parse_oracle_output


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a set of predictions."""

    accuracy: float
    mean_confidence: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier: float
    n_samples: int

    # Per-bin data for reliability diagram
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]


def compute_calibration_metrics(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute calibration metrics.

    Args:
        confidences: List of confidence values (0-1)
        correct: List of whether each prediction was correct
        n_bins: Number of bins for ECE calculation

    Returns:
        CalibrationMetrics with all computed values
    """
    confidences = np.array(confidences)
    correct = np.array(correct, dtype=float)

    n = len(confidences)
    if n == 0:
        return CalibrationMetrics(
            accuracy=0, mean_confidence=0, ece=0, mce=0, brier=0,
            n_samples=0, bin_accuracies=[], bin_confidences=[], bin_counts=[]
        )

    # Basic metrics
    accuracy = correct.mean()
    mean_confidence = confidences.mean()
    brier = ((confidences - correct) ** 2).mean()

    # Bin-based metrics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        bin_count = mask.sum()
        bin_counts.append(int(bin_count))

        if bin_count > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_accuracies.append(float(bin_acc))
            bin_confidences.append(float(bin_conf))

            # ECE contribution
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
            mce = max(mce, abs(bin_acc - bin_conf))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append((low + high) / 2)

    return CalibrationMetrics(
        accuracy=float(accuracy),
        mean_confidence=float(mean_confidence),
        ece=float(ece),
        mce=float(mce),
        brier=float(brier),
        n_samples=n,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


def print_calibration_report(metrics: CalibrationMetrics, name: str = ""):
    """Print a formatted calibration report."""
    print(f"\n{'='*60}")
    print(f"Calibration Report{f': {name}' if name else ''}")
    print(f"{'='*60}")
    print(f"Samples:          {metrics.n_samples}")
    print(f"Accuracy:         {metrics.accuracy:.3f}")
    print(f"Mean Confidence:  {metrics.mean_confidence:.3f}")
    print(f"ECE:              {metrics.ece:.4f}")
    print(f"MCE:              {metrics.mce:.4f}")
    print(f"Brier Score:      {metrics.brier:.4f}")
    print(f"\nReliability Diagram Data:")
    print(f"{'Bin':<6} {'Conf':<8} {'Acc':<8} {'Count':<8} {'Gap':<8}")
    print("-" * 40)
    for i, (conf, acc, count) in enumerate(zip(
        metrics.bin_confidences,
        metrics.bin_accuracies,
        metrics.bin_counts
    )):
        gap = abs(conf - acc) if count > 0 else 0
        print(f"{i:<6} {conf:<8.3f} {acc:<8.3f} {count:<8} {gap:<8.3f}")


def plot_reliability_diagram(
    metrics: CalibrationMetrics,
    save_path: str | None = None,
):
    """Plot reliability diagram.

    Perfect calibration = diagonal line.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Actual calibration
    confs = metrics.bin_confidences
    accs = metrics.bin_accuracies
    counts = metrics.bin_counts

    # Only plot bins with samples
    valid = [i for i, c in enumerate(counts) if c > 0]
    if valid:
        ax.bar(
            [confs[i] for i in valid],
            [accs[i] for i in valid],
            width=0.1,
            alpha=0.7,
            edgecolor='black',
            label='Model'
        )

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Reliability Diagram (ECE={metrics.ece:.3f})')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reliability diagram to {save_path}")
    else:
        plt.show()

    plt.close()


class CalibrationEvaluator:
    """Evaluator for calibrated activation oracles."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_on_dataset(
        self,
        prompts: list[str],
        questions: list[str],
        ground_truths: list[str],
        judge_fn,  # Function that returns True/False for correctness
    ) -> CalibrationMetrics:
        """Evaluate calibration on a dataset.

        Args:
            prompts: Original texts
            questions: Questions about the texts
            ground_truths: Correct answers
            judge_fn: Function(response, ground_truth) -> bool

        Returns:
            CalibrationMetrics
        """
        confidences = []
        correct = []

        for prompt, question, gt in tqdm(
            zip(prompts, questions, ground_truths),
            total=len(prompts),
            desc="Evaluating"
        ):
            # Generate oracle response (implement based on your setup)
            response = self._generate_response(prompt, question)
            parsed = parse_oracle_output(response)

            confidences.append(parsed.confidence_normalized)
            correct.append(judge_fn(parsed.answer, gt))

        return compute_calibration_metrics(confidences, correct)

    def _generate_response(self, prompt: str, question: str) -> str:
        """Generate oracle response for a prompt/question pair.

        Override this in subclass or extend with your specific setup.
        """
        raise NotImplementedError("Implement _generate_response for your setup")


if __name__ == "__main__":
    # Demo with synthetic data
    print("Demo: Calibration metrics on synthetic data\n")

    # Well-calibrated model
    np.random.seed(42)
    n = 1000
    confidences_good = np.random.beta(2, 2, n)  # Centered around 0.5
    correct_good = np.random.random(n) < confidences_good

    metrics_good = compute_calibration_metrics(
        confidences_good.tolist(),
        correct_good.tolist()
    )
    print_calibration_report(metrics_good, "Well-calibrated model")

    # Overconfident model
    confidences_bad = np.clip(confidences_good + 0.3, 0, 1)
    correct_bad = correct_good  # Same accuracy, but overconfident

    metrics_bad = compute_calibration_metrics(
        confidences_bad.tolist(),
        correct_bad.tolist()
    )
    print_calibration_report(metrics_bad, "Overconfident model")
