from __future__ import annotations

from pal.compression.components.adverbs import Adverbs
from pal.compression.components.hedges import Hedges
from pal.compression.components.interjections import Interjections
from pal.compression.components.pleonasms import Pleonasms
from pal.compression.components.punctuations import Punctuations
from pal.compression.components.stopwords import Stopwords


class TextMinifier:
    """Strips hedges, pleonasms, interjections, filler adverbs, and
    stopwords, then normalizes punctuation. Real input-side token
    reduction, applied before a message is ever sent to an LLM core."""

    def __init__(
        self,
        hedges: Hedges | None = None,
        pleonasms: Pleonasms | None = None,
        interjections: Interjections | None = None,
        adverbs: Adverbs | None = None,
        stopwords: Stopwords | None = None,
        punctuations: Punctuations | None = None,
    ) -> None:
        self.hedges = hedges or Hedges()
        self.pleonasms = pleonasms or Pleonasms()
        self.interjections = interjections or Interjections()
        self.adverbs = adverbs or Adverbs()
        self.stopwords = stopwords or Stopwords()
        self.punctuations = punctuations or Punctuations()

    def _remove_hedges(self, text: str) -> str:
        return self.hedges.remove(text)

    def _remove_pleonasms(self, text: str) -> str:
        return self.pleonasms.remove(text)

    def _remove_interjections(self, text: str) -> str:
        return self.interjections.remove(text)

    def _remove_adverbs(self, text: str) -> str:
        return self.adverbs.remove(text)

    def _remove_stopwords(self, text: str) -> str:
        return self.stopwords.remove(text)

    def _normalize_punctuations(self, text: str) -> str:
        return self.punctuations.remove(text)

    def minify(self, text: str) -> str:
        without_hedges = self._remove_hedges(text)
        without_pleonasms = self._remove_pleonasms(without_hedges)
        without_interjections = self._remove_interjections(without_pleonasms)
        without_adverbs = self._remove_adverbs(without_interjections)
        without_stopwords = self._remove_stopwords(without_adverbs)
        return self._normalize_punctuations(without_stopwords)
