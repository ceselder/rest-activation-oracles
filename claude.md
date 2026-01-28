# ReST Activation Oracle - Progress Log

## Project Overview

Training a **calibrated Activation Oracle** using **Reinforced Self-Training (ReST)**.

The oracle learns to output honest confidence estimates alongside informative answers:
```
[epistemic status: 0.85] The text discusses a long-distance relationship...
```

## Key Insight

The asymmetry that makes this work:
- **Oracle sees:** activations only
- **Judge sees:** full prompt + question + oracle's answer

We have infinite cheap ground truth for any question we can generate.

## Reward Function

```
reward = informativeness - λ × (confidence - informativeness)²
```

This prevents:
1. Vague but "correct" answers (low informativeness)
2. Overconfident wrong answers (high Brier penalty)

---

## Progress Log

### 2026-01-28: Project Setup + GPU Testing

**Completed:**
- [x] Cloned activation-oracle repo for reference implementation
- [x] Read paper for technical details (norm-matched additive steering at layer 1)
- [x] Created project structure:
- [x] SSH to GPU machine (RTX 5090 32GB) on vast.ai
- [x] Verified model loads (3.44GB VRAM for Qwen3-1.7B)
- [x] Verified activation extraction works
- [x] Verified steering injection works
- [x] Verified reward computation works
- [x] Ran mini training test successfully
- [x] Pushed to GitHub: https://github.com/ceselder/rest-activation-oracles

**Created project structure:
  - `config.py` - RESTConfig with all hyperparameters + 10 question templates
  - `epistemic_status.py` - Parse/format `[epistemic status: X.XX] answer`
  - `reward.py` - Compute reward = informativeness - λ × Brier
  - `judge.py` - InformativenessJudge (OpenRouter) + LocalJudge
  - `question_generation.py` - Generate diverse questions with templates
  - `data_pipeline.py` - Load prompts from diverse sources
  - `rest_trainer.py` - Main ReST loop (GROW → SCORE → IMPROVE)
  - `train.py` - Entry point

**Key Implementation Details:**
- Norm matching from original code: `steered = normalize(v) * ||orig|| + orig`
- Hook at layer 1 for activation injection
- Extract activations from layers at 25%, 50%, 75% depth
- Using Qwen3-1.7B for fast iteration (user's instruction)

**Next Steps:**
1. ~~Test SSH connection to GPU machine~~ ✓
2. ~~Install dependencies~~ ✓
3. ~~Run initial training with small dataset to verify pipeline~~ ✓
4. Run full ReST training

**Mini ReST Test Results (2026-01-28):**
- Model: Qwen3-8B + pretrained AO checkpoint
- VRAM: 17GB base → 19GB during training (32GB available)
- Response 1: `[epistemic status: 0.75]` - correctly formatted!
- Response 3: `[epistemic status: 0.85]` - also formatted!
- Reward correctly penalizes overconfident wrong answers
- Training loss: 1.85

---

## Architecture Notes

### Activation Injection (from paper)

For each placeholder token at position i with injected vector v_i:

```
h'_i = h_i + ||h_i|| × v_i/||v_i||
```

This is **additive** steering with norm matching at layer 1.

### Training Data Format

Oracle prompt structure:
```
Layer: {number}
? ? ? ... (placeholder tokens for activation injection)
{question}
```

### ReST Loop

1. **GROW**: Generate K samples per (activation, question) pair
2. **SCORE**: Judge scores informativeness, compute reward
3. **IMPROVE**: Weighted SFT on top (1 - filter_bottom_percent) samples
4. Repeat for N rounds

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen3-1.7B | Fast iteration |
| ReST rounds | 5 | |
| Samples per question | 5 | |
| Filter bottom | 20% | |
| λ (calibration) | 0.5 | Tune if needed |
| Question temp | 1.1 | High for diversity |
| Oracle temp | 0.7 | |
| LoRA rank | 64 | |
| LR | 1e-5 | |

---

## Failure Mode Alerts

- If confidence std < 0.1 → **confidence collapse**, reduce λ
- If answer length dropping → **vague answer exploit**, check judge
- If Brier not decreasing → **reward signal broken**

---

## Resources

- Paper: https://arxiv.org/abs/2512.15674
- Original repo: https://github.com/adamkarvonen/activation_oracles
- Pretrained checkpoints: https://huggingface.co/adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B
