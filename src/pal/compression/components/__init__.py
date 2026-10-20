"""
Optional NLP-backed components (nltk/spacy/proselint). Each import is
independently guarded: if a dependency isn't installed yet (before
install.sh's model-download step has run, or in a minimal/CI
environment), the corresponding name is simply left unavailable here
rather than crashing the entire pal import chain -- Python
imports a package's __init__.py as a side effect of importing ANY of
its submodules, so this file being fragile would break unrelated
components too (llm_wrapper, orchestrator, ...) that have nothing to do
with text compression.

pal.compression.pipeline.CompressionPipeline already degrades
gracefully (falls back to whitespace/dedupe-only compression) when
TextMinifier itself can't be built for the same reason.
"""

try:
    from pal.compression.components.adverbs import Adverbs
except Exception:  # missing OR broken (e.g. a torch DLL error)
    Adverbs = None

try:
    from pal.compression.components.hedges import Hedges
except Exception:  # missing OR broken (e.g. a torch DLL error)
    Hedges = None

try:
    from pal.compression.components.interjections import Interjections
except Exception:  # missing OR broken (e.g. a torch DLL error)
    Interjections = None

try:
    from pal.compression.components.pleonasms import Pleonasms
except Exception:  # missing OR broken (e.g. a torch DLL error)
    Pleonasms = None

try:
    from pal.compression.components.punctuations import Punctuations
except Exception:  # missing OR broken (e.g. a torch DLL error)
    Punctuations = None

try:
    from pal.compression.components.stopwords import Stopwords
except Exception:  # missing OR broken (e.g. a torch DLL error)
    Stopwords = None

try:
    from pal.compression.components.minifier import TextMinifier
except Exception:  # missing OR broken (e.g. a torch DLL error)
    TextMinifier = None

# No optional deps -- always available.
from pal.compression.components.output_compressor import OutputCompressor

__all__ = [
    "Adverbs",
    "Hedges",
    "Interjections",
    "TextMinifier",
    "OutputCompressor",
    "Pleonasms",
    "Punctuations",
    "Stopwords",
]
