from __future__ import annotations

import nltk
from nltk.corpus import stopwords as nltk_stopwords

from pal.compression.utils.negations import Negations
from pal.compression.utils.pronouns import Pronouns


class Stopwords:
    def __init__(
        self,
        pronouns: Pronouns | None = None,
        negations: Negations | None = None,
        exclude_contractions: bool = True,
    ) -> None:
        self.pronouns = pronouns or Pronouns()
        self.negations = negations or Negations()
        self.words = self._build_stop_words(exclude_contractions)

    def _build_stop_words(self, exclude_contractions: bool) -> set[str]:
        raw_stop_words = set(nltk_stopwords.words("english"))
        if exclude_contractions:
            return {word for word in raw_stop_words if "'" not in word}
        return raw_stop_words

    def _tokenize(self, text: str) -> list[str]:
        return text.split()

    def _tag_tokens(self, tokens: list[str]) -> list[tuple[str, str]]:
        return nltk.pos_tag(tokens)

    def _is_stop_word(self, word: str) -> bool:
        return word.lower() in self.words

    def _is_protected(self, word: str, tag: str) -> bool:
        if self.pronouns.is_pronoun(word, tag):
            return True
        return self.negations.is_negation(word)

    def _is_removable(self, word: str, tag: str) -> bool:
        if self._is_protected(word, tag):
            return False
        return self._is_stop_word(word)

    def _filter_tokens(
        self, tagged_tokens: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        return [
            (word, tag)
            for word, tag in tagged_tokens
            if not self._is_removable(word, tag)
        ]

    def _join_tokens(self, tagged_tokens: list[tuple[str, str]]) -> str:
        return " ".join(word for word, _ in tagged_tokens)

    def remove(self, text: str) -> str:
        tokens = self._tokenize(text)
        tagged_tokens = self._tag_tokens(tokens)
        filtered_tokens = self._filter_tokens(tagged_tokens)
        return self._join_tokens(filtered_tokens)
