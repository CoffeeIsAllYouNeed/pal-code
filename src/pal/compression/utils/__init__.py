"""Optional NLP-backed setup helpers -- see components/__init__.py for
why each import here is individually guarded rather than allowed to
raise straight out of this package's __init__."""

try:
    from pal.compression.utils.negations import Negations
except Exception:  # missing OR broken (e.g. a torch DLL error)
    Negations = None

try:
    from pal.compression.utils.nltk_setup import NltkSetup
except Exception:  # missing OR broken (e.g. a torch DLL error)
    NltkSetup = None

try:
    from pal.compression.utils.proselint_setup import ProselintSetup
except Exception:  # missing OR broken (e.g. a torch DLL error)
    ProselintSetup = None

try:
    from pal.compression.utils.pronouns import Pronouns
except Exception:  # missing OR broken (e.g. a torch DLL error)
    Pronouns = None

try:
    from pal.compression.utils.spacy_setup import SpacySetup
except Exception:  # missing OR broken (e.g. a torch DLL error)
    SpacySetup = None

__all__ = ["Negations", "NltkSetup", "ProselintSetup", "Pronouns", "SpacySetup"]
