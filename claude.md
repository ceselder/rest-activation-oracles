# GRPO Activation Oracle - Calibration Training

## Quick Resume Context

**Dataset:** `ceselder/wildchat-oracle-questions` (1000 prompts, 8k questions)
**Model:** Qwen3-8B + pretrained AO checkpoint (`adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B`)
**Method:** GRPO via TRL's GRPOTrainer
**Module:** `grpo_ao/` (renamed from rest_ao)

---

## Project Overview

Training a **calibrated Activation Oracle** using **Group Relative Policy Optimization (GRPO)**.

The oracle outputs honest confidence estimates (0-100) alongside informative answers:
```
[epistemic status: 85] The text discusses a long-distance relationship...
[epistemic status: 30] This appears to be about machine learning, but the activations are ambiguous.
[epistemic status: 10] I cannot determine the topic from these activations.
```

**Why GRPO over ReST:**
- Same core loop (sample N completions, score, learn from good ones)
- Learns from relative ranking within group, not absolute filtering
- Naturally prevents collapse: if all outputs identical, advantage = 0, no gradient
- Already implemented in TRL's GRPOTrainer
- No filter threshold to tune

## Key Insight: Infinite Ground Truth

The asymmetry that makes this work:
- **Oracle sees:** activations only
- **Judge sees:** full prompt + question + oracle's answer

The judge can trivially verify any question. We have infinite cheap ground truth.

---

## Reward Function: Two Axes

**Critical insight:** Pure Brier score is exploitable with vague answers.

**The fix:** Judge scores INFORMATIVENESS on a 0-1 scale, not binary correctness.

**Reward formula:**
```
reward = informativeness - λ × (confidence/100 - informativeness)²
```
(confidence is 0-100, informativeness is 0-1, so we normalize)

| Answer Type | Info | Conf | Brier | Reward |
|-------------|------|------|-------|--------|
| Detailed + right + calibrated | 0.9 | 90 | 0.00 | 0.90 ✓ |
| Detailed + wrong + overconfident | 0.2 | 90 | 0.49 | -0.05 ✗ |
| Vague + "correct" + overconfident | 0.2 | 100 | 0.64 | -0.12 ✗ |

Model must be informative AND calibrated.

---

## Data Pipeline

### Dataset
**WildChat Oracle Questions** (`ceselder/wildchat-oracle-questions`)
- 1000 user prompts from WildChat
- 7997 oracle probe questions
- 52.7% binary (yes/no), 47.3% open-ended
- ~50% negative questions (probing wrong topics/emotions)

Format:
```json
{
  "wildchat_question": "Hey there! I'm struggling with this Python code...",
  "language": "english",
  "oracle_questions": [
    "Is this user asking for help with programming?",
    "Is the user frustrated?",
    "Is this discussing cooking?",
    "What is the main topic of the message?"
  ]
}
```

### Activation Extraction
Extract activations from Qwen3-8B at layers 25%, 50%, 75% depth.

**Critical:** Follow the norm-matched additive steering from sft.py:
```
h'_i = h_i + ||h_i|| × v_i/||v_i||
```
Hook at layer 1 for injection.

---

## GRPO Training Loop

Use TRL's `GRPOTrainer`. Key mechanics:

### Group Sampling
For each (activation, question) pair, sample G completions (e.g., G=8).

### Advantage Computation
```
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

**Why collapse is impossible:** If all G outputs are identical with same reward, std = 0, advantage = 0, no learning. Model MUST differentiate.

### Custom Reward Function
For each response:
1. Parse epistemic status (confidence 0-100, normalize to 0-1)
2. Have judge score informativeness (0-1)
3. Compute: `reward = informativeness - λ * (confidence/100 - informativeness)²`

### Integration with Activation Oracle
The tricky part: GRPOTrainer expects standard LLM. Need to wrap AO so:
1. Input includes activation (injected via hook from sft.py)
2. Generation produces `[epistemic status: XX] answer` (XX = 0-100)
3. Reward function can access original prompt for judging

May need to adapt GRPOTrainer or write custom loop borrowing GRPO's advantage computation.

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen3-8B | Fast iteration |
| AO Checkpoint | adamkarvonen/checkpoints_all_single_and_multi_pretrain_cls_latentqa_posttrain_Qwen3-8B | |
| Group size | 8 | Completions per (activation, question) |
| KL penalty (β) | 0.05 | Tune this |
| λ (calibration) | 0.5 | Start here |
| Learning rate | 1e-6 | Lower than SFT for RL stability |
| LoRA rank | 64 | Match sft.py |
| Oracle temp | 0.9 | For diverse samples |
| Layer percents | [25, 50, 75] | Activation extraction |

---

## Metrics to Track (wandb)

**Per training step:**
- Mean reward
- Mean Brier score (should decrease)
- Mean informativeness (should stay stable or increase)
- Mean confidence (watch for collapse to 50)
- Confidence std dev (should NOT collapse)
- Advantage std dev (if this ≈ 0, no learning)
- KL divergence from reference

**Calibration diagnostics:**
- ECE (Expected Calibration Error)
- Reliability diagram: accuracy vs confidence per bin

**Failure mode alerts:**
- confidence_std < 0.1 → confidence collapse
- advantage_std ≈ 0 → all outputs identical
- mean_answer_length dropping → vague answer exploit
- KL exploding → reduce LR or increase β

---

## Evaluation

**Classification benchmarks** (from evaluate.py):
- geometry_of_truth, relations, sst2, md_gender, snli, ag_news, ner, tense, singular_plural
- 250 samples per dataset

**Calibration metrics:**
- ECE, MCE, Brier, reliability diagrams
- Run at start of training (`--no_initial_eval` to skip)

**Our eval vs reference (neural_chameleons):**
| Aspect | Reference | Our GRPO |
|--------|-----------|----------|
| Primary Metric | TPR (binary detection) | Accuracy + Calibration |
| Confidence | Binary (>0.5 threshold) | Continuous (0-100) |
| Calibration | Not measured | Core focus (ECE, Brier) |
| Test Size | ~20 samples/concept | 250 samples/task |

Our eval focuses on calibration quality, not concept detection. Different goals.

---

## Usage

```bash
# Default: Qwen3-8B with initial eval
python train.py

# Skip initial eval for faster start
python train.py --no_initial_eval

# Use Gemma 3 27B (slower but stronger)
python train.py --model google/gemma-3-27b-it \
    --oracle_lora_path adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it

# Adjust GRPO params
python train.py --num_generations 16 --kl_penalty 0.1 --lr 5e-7
```

---

## Implementation Plan

### Phase 1: Data Loading
- [x] WildChat dataset script (`fetch_wildchat_questions.py`)
- [x] Dataset on HuggingFace (`ceselder/wildchat-oracle-questions`)
- [ ] Data loader that creates (prompt, question, activation) tuples

### Phase 2: GRPO Integration
- [ ] Study TRL's GRPOTrainer source
- [ ] Create wrapper for activation-injected generation
- [ ] Custom reward function with judge integration
- [ ] Handle activation injection within GRPO's sampling loop

### Phase 3: Training
- [x] Initial eval toggle (`--no_initial_eval`)
- [ ] Set up wandb logging
- [ ] Run initial training (100 prompts)
- [ ] Scale to full dataset (1000 prompts)
- [ ] Monitor for failure modes

### Phase 4: Evaluation
- [x] Classification benchmarks
- [x] Calibration metrics (ECE, Brier)
- [ ] Compare to baseline

---

## Architecture Notes

### Activation Injection (from paper)
Additive steering with norm matching at layer 1:
```python
h'_i = h_i + ||h_i|| × v_i/||v_i||
```

### Oracle Prompt Format
```
Layer: {number}
? ? ? ... (placeholder tokens for activation injection)
{question}
```

### Epistemic Status Parsing
```python
def parse_oracle_output(response: str) -> tuple[int, str]:
    # Match [epistemic status: XX] where XX is 0-100
    # Default to 50 if unparseable (naturally penalizes bad formatting)
```

---

## Resources

- **Activation Oracle paper:** https://arxiv.org/abs/2512.15674
- **GRPO paper:** https://arxiv.org/abs/2402.03300 (DeepSeekMath, Section 4)
- **Original repo:** https://github.com/adamkarvonen/activation_oracles
- **TRL GRPOTrainer:** https://github.com/huggingface/trl

---

## Progress Log

### 2026-01-28: Initial ReST Implementation
- Created project structure with ReST trainer
- Implemented reward function, judge, question generation
- Batched operations for GPU efficiency
- Generated WildChat oracle questions dataset (1000 prompts, $0.056)

### 2026-01-28: Switching to GRPO
- Renamed module: `rest_ao` → `grpo_ao`
- Renamed classes: `RESTConfig` → `GRPOConfig`, `RESTTrainer` → `GRPOTrainer`
- Changed epistemic status format: 0.0-1.0 → 0-100 integers
- Switched default model to Qwen3-8B for fast iteration
- Added initial eval toggle (`run_initial_eval`, `--no_initial_eval`)
- Compared eval to reference: ours focuses on calibration, theirs on TPR
