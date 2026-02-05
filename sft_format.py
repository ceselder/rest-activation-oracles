#!/usr/bin/env python3
"""SFT to teach the oracle the epistemic status format.

The pre-trained LoRA doesn't know about [epistemic status: X] format.
We just need to teach the format - capabilities come from RL later.

Strategy:
1. Generate responses using the pre-trained oracle (no format)
2. Wrap with random confidence levels (varied distribution)
3. Fine-tune for ~200 steps
"""

import argparse
import os
import random
from pathlib import Path

# Load .env
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from grpo_ao.config import GRPOConfig
from grpo_ao.epistemic_status import ORACLE_SYSTEM_PROMPT, format_epistemic_output

import sys
for path in [
    str(Path(__file__).parent.parent / "activation_oracles"),
    "/root/activation_oracles",
]:
    if Path(path).exists():
        sys.path.insert(0, path)
        break

from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.common import load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import get_introspection_prefix, SPECIAL_TOKEN


def format_messages_for_model(system_prompt: str, user_content: str, assistant_content: str | None = None) -> list[dict]:
    """Format messages, merging system into user for Gemma 2 compatibility."""
    merged_user = f"{system_prompt}\n\n{user_content}"
    if assistant_content is not None:
        return [
            {"role": "user", "content": merged_user},
            {"role": "assistant", "content": assistant_content},
        ]
    else:
        return [{"role": "user", "content": merged_user}]


def random_confidence() -> int:
    """Generate uniform confidence levels across the full 0-10 range."""
    return random.randint(0, 10)


def generate_sft_data(cfg: GRPOConfig, num_examples: int = 500) -> list[dict]:
    """Generate SFT examples with certainty format."""
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    print(f"Loading model: {cfg.model_name}")
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )

    # Load pretrained LoRA
    if cfg.oracle_lora_path:
        print(f"Loading LoRA: {cfg.oracle_lora_path}")
        model = PeftModel.from_pretrained(
            model, cfg.oracle_lora_path, is_trainable=True
        )

    # Find layers
    base_model = model.base_model.model
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    elif hasattr(base_model, 'layers'):
        layers = base_model.layers
    else:
        raise RuntimeError(f"Could not find layers in {type(base_model)}")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("ceselder/wildchat-oracle-questions", split="train")

    # Generate examples
    sft_examples = []
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    model.eval()

    for i in tqdm(range(min(num_examples, len(indices))), desc="Generating SFT data"):
        idx = indices[i]
        example = dataset[idx]
        prompt = example["wildchat_question"]
        questions = example["oracle_questions"]

        # Pick random question and layer
        question = random.choice(questions)
        layer_percent = random.choice(cfg.layer_percents)

        # Extract activations
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        num_layers = len(layers)
        layer_idx = int(num_layers * layer_percent / 100)

        activations = []
        def capture_hook(module, inp, out):
            if isinstance(out, tuple):
                activations.append(out[0].detach())
            else:
                activations.append(out.detach())

        handle = layers[layer_idx].register_forward_hook(capture_hook)
        with model.disable_adapter(), torch.no_grad():
            model(input_ids)
        handle.remove()

        acts = activations[0][0]
        window = min(5, acts.shape[0])
        positions = list(range(acts.shape[0] - window, acts.shape[0]))
        steering_vectors = acts[positions]

        # Generate response (without format)
        num_positions = steering_vectors.shape[0]
        prefix = get_introspection_prefix(layer_idx, num_positions)

        # Don't use ORACLE_SYSTEM_PROMPT yet - generate raw answer
        messages = [
            {"role": "user", "content": prefix + question},
        ]

        try:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors=None,
                enable_thinking=False,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors=None
            )

        input_ids_list = encoded.input_ids if hasattr(encoded, 'input_ids') else list(encoded)
        gen_input_ids = torch.tensor([input_ids_list], device=device)

        special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
        hook_positions = [i for i, t in enumerate(input_ids_list) if t == special_token_id][:num_positions]
        if len(hook_positions) != num_positions:
            hook_positions = list(range(num_positions))

        hook_fn = get_hf_activation_steering_hook(
            vectors=[steering_vectors],
            positions=[hook_positions],
            steering_coefficient=1.0,
            device=device,
            dtype=dtype,
        )

        # Inject at layer 1 (oracle injection layer), NOT extraction layer
        with torch.no_grad(), add_hook(layers[1], hook_fn):
            output_ids = model.generate(
                gen_input_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.oracle_temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        raw_answer = tokenizer.decode(output_ids[0][gen_input_ids.shape[1]:], skip_special_tokens=True).strip()

        if not raw_answer:
            continue

        # Random confidence - GRPO will learn calibration later
        confidence = random_confidence()

        # Format with epistemic status
        formatted_answer = format_epistemic_output(confidence, raw_answer)

        sft_examples.append({
            "prompt": prompt,
            "question": question,
            "layer_percent": layer_percent,
            "prefix": prefix,
            "answer": formatted_answer,
            "raw_answer": raw_answer,
            "confidence": confidence,
        })

        if i % 100 == 0 and sft_examples:
            print(f"\nExample {i}: conf={confidence}")
            print(f"  Q: {question[:60]}...")
            print(f"  A: {formatted_answer[:80]}...")

    return sft_examples


def train_sft(cfg: GRPOConfig, sft_examples: list[dict], num_steps: int = 200):
    """Fine-tune on SFT examples."""
    import wandb
    from torch.optim import AdamW

    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    print(f"\nLoading model for SFT...")
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )

    if cfg.oracle_lora_path:
        model = PeftModel.from_pretrained(
            model, cfg.oracle_lora_path, is_trainable=True
        )

    model.enable_input_require_grads()
    model.train()

    # Find layers
    base_model = model.base_model.model
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    elif hasattr(base_model, 'layers'):
        layers = base_model.layers
    else:
        raise RuntimeError(f"Could not find layers")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    wandb.init(
        project=cfg.wandb_project,
        name=f"sft_format_{cfg.model_name.split('/')[-1]}",
        config={"sft_steps": num_steps, "num_examples": len(sft_examples)},
    )

    pbar = tqdm(range(num_steps), desc="SFT")

    for step in pbar:
        # Sample example
        ex = random.choice(sft_examples)

        # Build input with system prompt merged into user (Gemma 2 compatibility)
        messages = format_messages_for_model(ORACLE_SYSTEM_PROMPT, ex["prefix"] + ex["question"], ex["answer"])

        try:
            full_encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors=None,
                enable_thinking=False,
            )
        except TypeError:
            full_encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors=None
            )

        full_ids = full_encoded.input_ids if hasattr(full_encoded, 'input_ids') else list(full_encoded)
        input_ids = torch.tensor([full_ids], device=device)

        # Create labels (mask prompt)
        try:
            prompt_encoded = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors=None,
                enable_thinking=False,
            )
        except TypeError:
            prompt_encoded = tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True, return_tensors=None
            )
        prompt_ids = prompt_encoded.input_ids if hasattr(prompt_encoded, 'input_ids') else list(prompt_encoded)
        labels = input_ids.clone()
        labels[0, :len(prompt_ids)] = -100

        # Recompute steering vectors for this example
        prompt_input_ids = tokenizer.encode(ex["prompt"], return_tensors="pt").to(device)
        layer_idx = int(len(layers) * ex["layer_percent"] / 100)

        activations = []
        def capture_hook(module, inp, out):
            if isinstance(out, tuple):
                activations.append(out[0].detach())
            else:
                activations.append(out.detach())

        handle = layers[layer_idx].register_forward_hook(capture_hook)
        with model.disable_adapter(), torch.no_grad():
            model(prompt_input_ids)
        handle.remove()

        acts = activations[0][0]
        window = min(5, acts.shape[0])
        positions = list(range(acts.shape[0] - window, acts.shape[0]))
        steering_vectors = acts[positions]
        num_positions = steering_vectors.shape[0]

        # Find hook positions in full sequence
        special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
        hook_positions = [i for i, t in enumerate(full_ids) if t == special_token_id][:num_positions]
        if len(hook_positions) != num_positions:
            hook_positions = list(range(num_positions))

        hook_fn = get_hf_activation_steering_hook(
            vectors=[steering_vectors],
            positions=[hook_positions],
            steering_coefficient=1.0,
            device=device,
            dtype=dtype,
        )

        # Forward with steering - inject at layer 1 (oracle injection layer)
        with add_hook(layers[1], hook_fn):
            outputs = model(input_ids=input_ids, labels=labels)

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"sft/loss": loss.item(), "sft/step": step})
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % 50 == 0:
            torch.cuda.empty_cache()

    # Save checkpoint
    save_path = Path(cfg.save_dir) / "sft_format"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"\nSaved SFT checkpoint to {save_path}")

    wandb.finish()
    return save_path


def main():
    parser = argparse.ArgumentParser(description="SFT to teach epistemic status format")
    parser.add_argument("--num_examples", type=int, default=500,
                        help="Number of SFT examples to generate")
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Number of SFT training steps")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip data generation, load from cache")
    args = parser.parse_args()

    cfg = GRPOConfig()
    cache_path = Path(cfg.save_dir) / "sft_data.pt"

    if args.skip_generation and cache_path.exists():
        print(f"Loading cached SFT data from {cache_path}")
        sft_examples = torch.load(cache_path)
    else:
        sft_examples = generate_sft_data(cfg, num_examples=args.num_examples)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sft_examples, cache_path)
        print(f"Saved {len(sft_examples)} SFT examples to {cache_path}")

    print(f"\n{'='*60}")
    print(f"SFT Training: {len(sft_examples)} examples, {args.num_steps} steps")
    print(f"{'='*60}")

    sft_checkpoint = train_sft(cfg, sft_examples, num_steps=args.num_steps)

    print(f"\n{'='*60}")
    print(f"SFT complete! Checkpoint saved to: {sft_checkpoint}")
    print(f"Now run GRPO with: --oracle_lora_path {sft_checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
