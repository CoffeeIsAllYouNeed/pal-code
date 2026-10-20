from __future__ import annotations

import re


class Punctuations:
    def _space_around_marks(self, text: str) -> str:
        return re.sub(r"\s*([,.!?;:])\s*", r"\1 ", text)

    def _collapse_repeated_marks(self, text: str) -> str:
        return re.sub(r"([,.!?;:]\s*){2,}", r"\1", text)

    def _remove_comma_before_terminal_mark(self, text: str) -> str:
        return re.sub(r",(\s*[.!?])", r"\1", text)

    def _trim_leading_marks(self, text: str) -> str:
        return re.sub(r"^[\s,.!?;:]+", "", text)

    def _trim_trailing_marks(self, text: str) -> str:
        return re.sub(r"[\s,.!?;:]+$", "", text)

    def _normalize_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _remove_trailing_period(self, text: str) -> str:
        return re.sub(r"\.$", "", text)

    def remove(self, text: str) -> str:
        text = self._space_around_marks(text)
        text = self._collapse_repeated_marks(text)
        text = self._remove_comma_before_terminal_mark(text)
        text = self._trim_leading_marks(text)
        text = self._trim_trailing_marks(text)
        text = self._normalize_whitespace(text)
        text = self._remove_trailing_period(text)
        return text
