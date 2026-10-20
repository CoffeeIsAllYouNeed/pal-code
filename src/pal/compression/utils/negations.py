from __future__ import annotations


class Negations:
    WORDS: set[str] = {
        "no", "not", "nor", "never", "none", "nobody", "nothing",
        "nowhere", "neither", "without", "cannot", "cant", "wont",
        "dont", "isnt", "arent", "wasnt", "werent", "hasnt",
        "havent", "hadnt", "doesnt", "didnt", "wouldnt", "shouldnt",
        "couldnt",
    }

    def _strip_apostrophes(self, word: str) -> str:
        return word.replace("'", "")

    def is_negation(self, word: str) -> bool:
        normalized = self._strip_apostrophes(word.lower())
        return normalized in self.WORDS
