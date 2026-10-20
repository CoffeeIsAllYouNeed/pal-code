from __future__ import annotations

from pal.compression.utils.spacy_setup import SpacySetup


class Adverbs:
    def __init__(self, spacy_setup: SpacySetup | None = None) -> None:
        self.spacy_setup = spacy_setup or SpacySetup()
        self.nlp = self.spacy_setup.load_model()

    def _tokenize(self, text: str):
        return self.nlp(text)

    def _is_adverb(self, token) -> bool:
        return token.pos_ == "ADV"

    def remove(self, text: str) -> str:
        doc = self._tokenize(text)
        kept_tokens = [token.text for token in doc if not self._is_adverb(token)]
        return " ".join(kept_tokens)
