from __future__ import annotations


class Pronouns:
    TAGS: set[str] = {"PRP", "PRP$", "WP", "WP$"}

    WORDS: set[str] = {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
        "who", "whom", "whose", "which", "that",
    }

    def _matches_tag(self, tag: str | None) -> bool:
        return tag is not None and tag in self.TAGS

    def _matches_word(self, word: str) -> bool:
        return word.lower() in self.WORDS

    def is_pronoun(self, word: str, tag: str | None = None) -> bool:
        if self._matches_tag(tag):
            return True
        return self._matches_word(word)
