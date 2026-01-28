#!/usr/bin/env python3
"""Run evaluation on ReST Activation Oracle using classification benchmarks from sft.py.

Uses the same classification datasets as the original AO paper:
- geometry_of_truth, relations, sst2, md_gender, snli, ag_news, ner, tense, etc.

Measures calibration (ECE, Brier) by comparing oracle confidence to correctness.

Usage:
    python evaluate.py --checkpoint checkpoints/round_2
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add paths for activation oracle imports
for path in [
    str(Path(__file__).parent.parent / "activation-oracle-rest"),
    str(Path(__file__).parent.parent / "activation_oracles"),
    "/root/activation_oracles",
]:
    if Path(path).exists():
        sys.path.insert(0, path)

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    construct_batch,
    get_introspection_prefix,
    materialize_missing_steering_vectors,
    SPECIAL_TOKEN,
)
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig

from grpo_ao.epistemic_status import parse_oracle_output, ORACLE_SYSTEM_PROMPT
from grpo_ao.calibration_eval import (
    compute_calibration_metrics,
    print_calibration_report,
    plot_reliability_diagram,
    CalibrationMetrics,
)

# Classification datasets from sft.py
CLASSIFICATION_DATASETS = {
    "geometry_of_truth": {"num_test": 250},
    "relations": {"num_test": 250},
    "sst2": {"num_test": 250},
    "md_gender": {"num_test": 250},
    "snli": {"num_test": 250},
    "ag_news": {"num_test": 250},
    "ner": {"num_test": 250},
    "tense": {"num_test": 250},
    "singular_plural": {"num_test": 250},
}


def load_classification_dataset(
    dataset_name: str,
    model_name: str,
    layer_percents: list[int],
    num_test: int = 250,
) -> list[TrainingDataPoint]:
    """Load a classification dataset for evaluation."""
    config = DatasetLoaderConfig(
        dataset_name="",
        num_train=0,
        num_test=num_test,
        splits=["test"],
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=False,
        batch_size=0,
        seed=42,
        custom_dataset_params=ClassificationDatasetConfig(
            classification_dataset_name=dataset_name,
            max_window_size=5,  # Multi-token for richer activations
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=1,
        ),
    )

    loader = ClassificationDatasetLoader(dataset_config=config)
    return loader.load_dataset("test")


def evaluate_on_dataset(
    model,
    tokenizer,
    submodule,
    device: torch.device,
    dtype: torch.dtype,
    eval_data: list[TrainingDataPoint],
    batch_size: int = 8,
) -> tuple[list[float], list[bool], list[dict]]:
    """Evaluate oracle on a dataset with batched inference.

    Returns:
        confidences: List of oracle confidence values
        correct: List of whether oracle answer matched ground truth
        responses: List of response dicts for inspection
    """
    confidences = []
    correct = []
    responses = []

    model.eval()

    for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
        batch = eval_data[i : i + batch_size]

        # Materialize steering vectors if needed
        batch = materialize_missing_steering_vectors(batch, tokenizer, model)

        for data_point in batch:
            # Build oracle prompt
            num_positions = len(data_point.positions)
            prefix = get_introspection_prefix(data_point.layer, num_positions)

            # The classification prompt is stored in target_output context
            # We need to extract the question - it's in the input_ids after prefix
            question = tokenizer.decode(data_point.input_ids, skip_special_tokens=True)
            # Clean up to get just the question part
            if "Answer with" in question:
                question = question[question.find("Answer with") :]

            messages = [
                {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
                {"role": "user", "content": prefix + question},
            ]

            input_dict = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = input_dict["input_ids"].to(device)

            # Find special token positions for activation injection
            input_ids_list = input_ids[0].tolist()
            special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
            inject_positions = [
                j for j, tok in enumerate(input_ids_list) if tok == special_token_id
            ][:num_positions]

            if len(inject_positions) != num_positions:
                inject_positions = list(range(num_positions))

            # Create steering hook
            steering_vectors = data_point.steering_vectors
            if steering_vectors is None:
                # Should have been materialized above
                continue

            hook_fn = get_hf_activation_steering_hook(
                vectors=[steering_vectors],
                positions=[inject_positions],
                steering_coefficient=1.0,
                device=device,
                dtype=dtype,
            )

            # Generate with activation injection
            with torch.no_grad(), add_hook(submodule, hook_fn):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
            )
            parsed = parse_oracle_output(response)

            # Check correctness
            target = data_point.target_output.strip().lower()
            answer = parsed.answer.strip().lower().rstrip(".!?,;:")

            is_correct = answer == target

            confidences.append(parsed.confidence_normalized)  # 0-1 for calibration
            correct.append(is_correct)
            responses.append({
                "question": question[:100],
                "target": target,
                "response": response,
                "answer": answer,
                "confidence": parsed.confidence,  # 0-100 raw for readability
                "correct": is_correct,
            })

    return confidences, correct, responses


def eval_all_datasets(
    model,
    tokenizer,
    submodule,
    device: torch.device,
    dtype: torch.dtype,
    model_name: str,
    layer_percents: list[int],
    datasets: dict[str, dict],
    batch_size: int = 8,
) -> dict[str, dict]:
    """Evaluate on all classification datasets, returning per-dataset metrics."""
    all_results = {}
    all_confidences = []
    all_correct = []

    for ds_name, ds_config in datasets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {ds_name}")
        print(f"{'='*60}")

        try:
            eval_data = load_classification_dataset(
                dataset_name=ds_name,
                model_name=model_name,
                layer_percents=layer_percents,
                num_test=ds_config["num_test"],
            )
        except Exception as e:
            print(f"Failed to load {ds_name}: {e}")
            continue

        if not eval_data:
            print(f"No data for {ds_name}, skipping")
            continue

        print(f"Loaded {len(eval_data)} samples")

        confidences, correct, responses = evaluate_on_dataset(
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=device,
            dtype=dtype,
            eval_data=eval_data,
            batch_size=batch_size,
        )

        if not confidences:
            continue

        metrics = compute_calibration_metrics(confidences, correct)
        print_calibration_report(metrics, ds_name)

        all_results[ds_name] = {
            "accuracy": metrics.accuracy,
            "mean_confidence": metrics.mean_confidence,
            "ece": metrics.ece,
            "brier": metrics.brier,
            "n_samples": metrics.n_samples,
            "responses": responses[:10],  # Save first 10 for inspection
        }

        # Aggregate
        all_confidences.extend(confidences)
        all_correct.extend(correct)

    # Overall metrics
    if all_confidences:
        overall_metrics = compute_calibration_metrics(all_confidences, all_correct)
        print_calibration_report(overall_metrics, "OVERALL")
        all_results["_overall"] = {
            "accuracy": overall_metrics.accuracy,
            "mean_confidence": overall_metrics.mean_confidence,
            "ece": overall_metrics.ece,
            "brier": overall_metrics.brier,
            "n_samples": overall_metrics.n_samples,
        }

        # Plot overall reliability diagram
        try:
            plot_reliability_diagram(overall_metrics, "reliability_diagram.png")
        except Exception as e:
            print(f"Could not plot reliability diagram: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained ReST checkpoint")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--base_lora", type=str,
                        default="adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--layer_percents", type=int, nargs="+", default=[25, 50, 75])
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to evaluate on (default: all)")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"Loading model: {args.model}")
    tokenizer = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="eager",
    )

    print(f"Loading base LoRA: {args.base_lora}")
    model = PeftModel.from_pretrained(model, args.base_lora)

    if args.checkpoint:
        print(f"Loading ReST checkpoint: {args.checkpoint}")
        model.load_adapter(args.checkpoint, adapter_name="rest")
        model.set_adapter("rest")

    model.eval()

    # Get hook submodule (layer 1 for activation injection)
    base_model = model.base_model.model
    submodule = base_model.model.layers[1]

    # Select datasets
    if args.datasets:
        datasets = {k: v for k, v in CLASSIFICATION_DATASETS.items() if k in args.datasets}
    else:
        datasets = CLASSIFICATION_DATASETS

    print(f"\nEvaluating on datasets: {list(datasets.keys())}")

    results = eval_all_datasets(
        model=model,
        tokenizer=tokenizer,
        submodule=submodule,
        device=device,
        dtype=dtype,
        model_name=args.model,
        layer_percents=args.layer_percents,
        datasets=datasets,
        batch_size=args.batch_size,
    )

    # Add metadata
    results["_meta"] = {
        "model": args.model,
        "base_lora": args.base_lora,
        "checkpoint": args.checkpoint,
        "layer_percents": args.layer_percents,
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
