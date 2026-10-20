from __future__ import annotations

from pal.compression.utils.spacy_setup import SpacySetup


class Interjections:
    def __init__(self, spacy_setup: SpacySetup | None = None) -> None:
        self.spacy_setup = spacy_setup or SpacySetup()
        self.nlp = self.spacy_setup.load_model()

    def _tokenize(self, text: str):
        return self.nlp(text)

    def _is_interjection(self, token) -> bool:
        return token.pos_ == "INTJ"

    def remove(self, text: str) -> str:
        doc = self._tokenize(text)
        kept_tokens = [token.text for token in doc if not self._is_interjection(token)]
        return " ".join(kept_tokens)
