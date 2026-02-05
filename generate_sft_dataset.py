#!/usr/bin/env python3
"""Generate SFT dataset to teach the oracle the epistemic status format.

The oracle already knows how to answer questions about activations.
We just need to teach it to output [epistemic status: X] before the answer.

This generates LOGICALLY CONSISTENT Q&A pairs with varied confidence levels.
"""

import json
import random
from pathlib import Path

# Consistent Q&A templates - each has a question and valid answers
QA_PAIRS = [
    # Yes/No questions
    {"q": "Is the user asking about programming?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about cooking?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about travel?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about relationships?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about health?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about sports?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about music?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about movies?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about politics?", "answers": ["Yes", "No"]},
    {"q": "Is the user asking about science?", "answers": ["Yes", "No"]},
    {"q": "Is the model being asked to write code?", "answers": ["Yes", "No"]},
    {"q": "Is the model being asked to give advice?", "answers": ["Yes", "No"]},
    {"q": "Is the model being asked to explain something?", "answers": ["Yes", "No"]},
    {"q": "Is the model being asked to translate?", "answers": ["Yes", "No"]},
    {"q": "Is the model being asked to summarize?", "answers": ["Yes", "No"]},
    {"q": "Is the model being asked to create a story?", "answers": ["Yes", "No"]},
    {"q": "Is this a technical question?", "answers": ["Yes", "No"]},
    {"q": "Is this a personal question?", "answers": ["Yes", "No"]},
    {"q": "Is this a factual question?", "answers": ["Yes", "No"]},
    {"q": "Is this a creative request?", "answers": ["Yes", "No"]},
    {"q": "Does the user seem frustrated?", "answers": ["Yes", "No"]},
    {"q": "Does the user seem happy?", "answers": ["Yes", "No"]},
    {"q": "Does the user seem confused?", "answers": ["Yes", "No"]},
    {"q": "Is the sentiment positive?", "answers": ["Yes", "No"]},
    {"q": "Is the sentiment negative?", "answers": ["Yes", "No"]},
    {"q": "Is the model confident in its response?", "answers": ["Yes", "No"]},
    {"q": "Is there a clear question being asked?", "answers": ["Yes", "No"]},
    {"q": "Does this require specialized knowledge?", "answers": ["Yes", "No"]},
    {"q": "Is this a multi-part request?", "answers": ["Yes", "No"]},
    {"q": "Does the user want a detailed explanation?", "answers": ["Yes", "No"]},

    # What questions with appropriate answers
    {"q": "What language is being discussed?", "answers": ["English", "Python", "JavaScript", "French", "Spanish", "Chinese", "German", "Japanese", "Multiple languages", "No specific language"]},
    {"q": "What is the main topic?", "answers": ["Programming", "Cooking", "Travel", "Health", "Science", "Technology", "Personal advice", "Creative writing", "Education", "Entertainment"]},
    {"q": "What is the user's emotional state?", "answers": ["Curious", "Frustrated", "Happy", "Confused", "Neutral", "Excited", "Anxious", "Calm", "Uncertain", "Determined"]},
    {"q": "What type of response would be appropriate?", "answers": ["Factual explanation", "Step-by-step guide", "Code example", "Personal advice", "Creative response", "Brief answer", "Detailed analysis", "Comparison", "Definition", "Example"]},
    {"q": "What is the sentiment of the text?", "answers": ["Positive", "Negative", "Neutral", "Mixed", "Uncertain"]},
    {"q": "What domain does this question belong to?", "answers": ["Technology", "Science", "Arts", "Personal", "Professional", "Academic", "Entertainment", "Health", "Finance", "General"]},
]


# The oracle uses a special token for steering injection points
SPECIAL_TOKEN = " ?"  # From nl_probes.utils.dataset_utils

# Qwen3-8B has 36 layers - use 50% = layer 18
NUM_LAYERS = 36  # Qwen3-8B
LAYER_IDX = 18   # 50% of 36 layers


def get_introspection_prefix(layer_idx: int, num_positions: int) -> str:
    """Generate the introspection prefix that the oracle expects.

    Format: "Layer: {layer}\n{SPECIAL_TOKEN * num_positions} \n"
    """
    return f"Layer: {layer_idx}\n{SPECIAL_TOKEN * num_positions} \n"


def random_confidence() -> int:
    """Generate random confidence 0-10 with varied distribution."""
    r = random.random()
    if r < 0.15:
        return random.randint(0, 2)  # Low
    elif r < 0.35:
        return random.randint(3, 4)  # Low-mid
    elif r < 0.65:
        return random.randint(5, 6)  # Mid
    elif r < 0.85:
        return random.randint(7, 8)  # High-mid
    else:
        return random.randint(9, 10)  # High


def generate_example() -> dict:
    """Generate one SFT example with logically consistent Q&A.

    NOTE: The oracle was trained WITHOUT a system prompt, using just:
    - user: "Layer: X\n ? \n{question}"
    - assistant: "{answer}"

    We add the epistemic format to teach it calibrated confidence.
    """
    qa = random.choice(QA_PAIRS)
    base_question = qa["q"]
    answer = random.choice(qa["answers"])
    confidence = random_confidence()

    # Add introspection prefix like the oracle expects
    # Use single layer (50% = layer 18) for simplicity
    # num_positions: typically 1 in inference, using 1-5 for variety
    num_positions = random.randint(1, 5)
    prefix = get_introspection_prefix(LAYER_IDX, num_positions)

    question = prefix + base_question

    formatted_response = f"[epistemic status: {confidence}] {answer}"

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "formatted_response": formatted_response,
        # No system prompt - oracle wasn't trained with one
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT dataset for epistemic status format")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="datasets/sft_format_dataset.jsonl", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Generate examples
    examples = [generate_example() for _ in range(args.num_examples)]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(examples)} examples to {output_path}")

    # Show distribution
    confidences = [ex["confidence"] for ex in examples]
    print("\nConfidence distribution:")
    for i in range(0, 11, 2):
        count = sum(1 for c in confidences if i <= c < i + 2)
        bar = "█" * (count // 20)
        print(f"  {i}-{i+1}: {bar} ({count})")

    # Show some examples
    print("\nSample examples:")
    for ex in random.sample(examples, 5):
        print(f"  Q: {ex['question']}")
        print(f"  A: {ex['formatted_response']}")
        print()


if __name__ == "__main__":
    main()
