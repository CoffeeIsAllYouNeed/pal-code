"""
Permanent behavior-constraint document, not a text-processing function.

The insight: trimming output AFTER the model generates it only saves
what gets displayed/logged -- the model already paid the generation
cost (and, for API backends, the billed cost) for every verbose token.
Instructing the model up front to be terse means fewer output tokens
get generated in the first place. That's a strictly better lever than
post-hoc compression, so this is wired in as a permanent addition to
every system prompt.
"""
from pathlib import Path

_CONSTRAINTS_FILE = Path(__file__).parent / "AGENT_OUTPUT.md"


def load_agent_constraints() -> str:
    """Returns the permanent output-behavior constraints as text, ready
    to append to any system prompt."""
    if not _CONSTRAINTS_FILE.exists():
        return ""
    return _CONSTRAINTS_FILE.read_text(encoding="utf-8").strip()
