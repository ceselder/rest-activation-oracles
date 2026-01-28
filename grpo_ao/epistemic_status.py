"""Handle epistemic status format for oracle outputs.

Format: [epistemic status: XX] <answer>

Where XX is an integer from 0 to 100.

Examples:
    [epistemic status: 85] The text discusses a long-distance relationship...
    [epistemic status: 30] This appears to be about machine learning...
    [epistemic status: 10] I cannot determine the topic from these activations.
"""

import re
from dataclasses import dataclass


@dataclass
class OracleOutput:
    """Parsed oracle output with confidence and answer."""

    confidence: int  # 0-100
    answer: str
    raw: str
    parse_success: bool

    @property
    def confidence_normalized(self) -> float:
        """Return confidence as 0-1 float for reward computation."""
        return self.confidence / 100.0


# Regex to parse epistemic status (now 0-100 integer)
EPISTEMIC_PATTERN = re.compile(
    r"^\s*\[epistemic status:\s*(\d+)\s*\]\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)


def parse_oracle_output(raw_output: str, default_confidence: int = 50) -> OracleOutput:
    """Parse oracle output into confidence and answer.

    If parsing fails, returns default_confidence (50) which penalizes
    bad formatting naturally through the Brier score.

    Args:
        raw_output: Raw string from oracle generation
        default_confidence: Confidence to use if parsing fails (0-100)

    Returns:
        OracleOutput with parsed or default values
    """
    raw_output = raw_output.strip()

    match = EPISTEMIC_PATTERN.match(raw_output)
    if match:
        try:
            confidence = int(match.group(1))
            # Clamp to [0, 100]
            confidence = max(0, min(100, confidence))
            answer = match.group(2).strip()
            return OracleOutput(
                confidence=confidence,
                answer=answer,
                raw=raw_output,
                parse_success=True
            )
        except ValueError:
            pass

    # Parsing failed - return default
    return OracleOutput(
        confidence=default_confidence,
        answer=raw_output,
        raw=raw_output,
        parse_success=False
    )


def format_epistemic_output(confidence: int, answer: str) -> str:
    """Format answer with epistemic status prefix.

    Args:
        confidence: Confidence level (0-100)
        answer: The answer text

    Returns:
        Formatted string with epistemic status prefix
    """
    return f"[epistemic status: {confidence}] {answer}"


# System prompt to teach the oracle the format
ORACLE_SYSTEM_PROMPT = """You are an Activation Oracle. You receive hidden activations from a language model and answer questions about what the model was processing.

You MUST always respond in this exact format:
[epistemic status: XX] Your answer here

Where XX is an integer from 0 to 100 representing your confidence percentage.

Guidelines for epistemic status:
- 90-100: Very confident, clear signal in activations
- 70-89: Confident, strong evidence
- 50-69: Moderate confidence, some uncertainty
- 30-49: Low confidence, weak or ambiguous signal
- 10-29: Very uncertain, mostly guessing
- 0-9: Cannot determine from activations

Be CALIBRATED: your confidence should match your actual accuracy. If you're often wrong when confident, lower your confidence. If you're often right when uncertain, raise it.

Examples:
[epistemic status: 85] The text discusses a long-distance relationship between two people navigating career uncertainty.
[epistemic status: 30] This appears to be about machine learning, but the activations are ambiguous about the specific subfield.
[epistemic status: 10] I cannot determine the topic from these activations."""
