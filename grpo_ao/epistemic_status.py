"""Handle epistemic status format for oracle outputs.

Format: [epistemic status: X] <answer>

Where X is an integer from 0 to 10.

Examples:
    [epistemic status: 9] The text discusses a long-distance relationship...
    [epistemic status: 3] This appears to be about machine learning...
    [epistemic status: 1] I cannot determine the topic from these activations.
"""

import re
from dataclasses import dataclass


@dataclass
class OracleOutput:
    """Parsed oracle output with confidence and answer."""

    confidence: int  # 0-10
    answer: str
    raw: str
    parse_success: bool

    @property
    def confidence_normalized(self) -> float:
        """Return confidence as 0-1 float for reward computation."""
        return self.confidence / 10.0


# Regex to parse epistemic status (0-10 integer)
EPISTEMIC_PATTERN = re.compile(
    r"^\s*\[epistemic status:\s*(\d+)\s*\]\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)


def parse_oracle_output(raw_output: str, default_confidence: int = -1) -> OracleOutput:
    """Parse oracle output into confidence and answer.

    If parsing fails, returns default_confidence (-1) which signals
    malformed output for a penalty in reward computation.

    Args:
        raw_output: Raw string from oracle generation
        default_confidence: Confidence to use if parsing fails (0-10)

    Returns:
        OracleOutput with parsed or default values
    """
    raw_output = raw_output.strip()

    match = EPISTEMIC_PATTERN.match(raw_output)
    if match:
        try:
            confidence = int(match.group(1))
            # Clamp to [0, 10]
            confidence = max(0, min(10, confidence))
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
        confidence: Confidence level (0-10)
        answer: The answer text

    Returns:
        Formatted string with epistemic status prefix
    """
    return f"[epistemic status: {confidence}] {answer}"


# System prompt to teach the oracle the format
ORACLE_SYSTEM_PROMPT = """Respond with: [epistemic status: X] Answer

X is your confidence from 0-10. Use the FULL range:
- 10: absolutely certain
- 8: very confident
- 6: fairly confident
- 4: somewhat uncertain
- 2: quite uncertain
- 0: no idea

If you're certain you CAN'T answer, that's still high confidence:
[epistemic status: 9] I can't answer this question given these activations.

Examples:
[epistemic status: 7] Yes
[epistemic status: 6] No
[epistemic status: 9] I can't answer this question given these activations.
[epistemic status: 8] The user is asking about Python debugging.
[epistemic status: 2] The question is about cooking"""
