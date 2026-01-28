#!/usr/bin/env python3
"""Quick test of the ReST pipeline components."""

import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add activation oracle repo to path
sys.path.insert(0, "/root/activation_oracles")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Test imports from original repo
from nl_probes.utils.steering_hooks import add_hook, get_hf_activation_steering_hook
from nl_probes.utils.activation_utils import get_hf_submodule, collect_activations_multiple_layers
from nl_probes.utils.common import layer_percent_to_layer
from nl_probes.utils.dataset_utils import get_introspection_prefix, SPECIAL_TOKEN

# Test imports from our module
sys.path.insert(0, "/root/rest-activation-oracles")
from grpo_ao.epistemic_status import parse_oracle_output, format_epistemic_output, ORACLE_SYSTEM_PROMPT
from grpo_ao.reward import compute_reward
from grpo_ao.config import QUESTION_TEMPLATES


def main():
    print("=" * 60)
    print("ReST Pipeline Test")
    print("=" * 60)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # 1. Load model
    print("\n1. Loading model...")
    model_name = "Qwen/Qwen3-1.7B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda",
    )
    print(f"   Model loaded, VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # 2. Add LoRA
    print("\n2. Adding LoRA adapter...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Test activation extraction
    print("\n3. Testing activation extraction...")
    test_text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(test_text, add_special_tokens=False)

    layer = layer_percent_to_layer(model_name, 50)
    print(f"   Using layer {layer} (50% of model)")

    inputs = {
        "input_ids": torch.tensor([input_ids], device=device),
        "attention_mask": torch.ones(1, len(input_ids), device=device),
    }

    submodules = {layer: get_hf_submodule(model, layer, use_lora=True)}

    with model.disable_adapter():
        acts = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs,
            min_offset=None,
            max_offset=None,
        )

    acts_tensor = acts[layer][0]  # [L, D]
    print(f"   Activations shape: {acts_tensor.shape}")
    print(f"   Activation norm: {acts_tensor.norm(dim=-1).mean():.2f}")

    # 4. Test activation injection
    print("\n4. Testing activation injection...")
    num_positions = 3
    prefix = get_introspection_prefix(layer, num_positions)
    print(f"   Prefix: {repr(prefix[:50])}...")

    question = "What is this text about?"
    messages = [
        {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
        {"role": "user", "content": prefix + question},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    # Find positions of special tokens
    special_token_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)[0]
    positions = [i for i, tok in enumerate(prompt_ids[0].tolist()) if tok == special_token_id]
    positions = positions[:num_positions]
    print(f"   Injection positions: {positions}")

    # Use first few activations
    steering_vecs = acts_tensor[:num_positions]

    hook_fn = get_hf_activation_steering_hook(
        vectors=[steering_vecs],
        positions=[positions],
        steering_coefficient=1.0,
        device=device,
        dtype=dtype,
    )

    submodule = get_hf_submodule(model, 1)  # Hook at layer 1

    with add_hook(submodule, hook_fn):
        outputs = model.generate(
            prompt_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][prompt_ids.shape[1]:], skip_special_tokens=True)
    print(f"   Generated response: {response[:100]}...")

    # 5. Test epistemic status parsing
    print("\n5. Testing epistemic status parsing...")

    test_outputs = [
        "[epistemic status: 0.85] This is about a fox.",
        "[epistemic status: 0.3] Something about animals maybe.",
        "No epistemic status here.",
        "[epistemic status: 1.5] Invalid confidence",
    ]

    for out in test_outputs:
        parsed = parse_oracle_output(out)
        print(f"   '{out[:40]}...' -> conf={parsed.confidence:.2f}, success={parsed.parse_success}")

    # 6. Test reward computation
    print("\n6. Testing reward computation...")

    test_cases = [
        ("Detailed + right + calibrated", 0.9, 0.9),
        ("Detailed + wrong + overconfident", 0.2, 0.9),
        ("Vague + 'correct' + overconfident", 0.2, 1.0),
        ("Vague + 'correct' + calibrated", 0.2, 0.2),
    ]

    for desc, info, conf in test_cases:
        parsed = parse_oracle_output(f"[epistemic status: {conf}] test answer")
        result = compute_reward(parsed, info, calibration_lambda=0.5)
        print(f"   {desc}: reward={result.reward:.2f} (info={info}, conf={conf}, brier={result.brier_score:.2f})")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
