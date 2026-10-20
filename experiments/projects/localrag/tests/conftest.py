import pytest

from localrag.engine import LocalRAG

CORPUS = {
    "cats.md": "# Cats\n\nCats are small carnivorous mammals. They purr, groom themselves and hunt mice at night.\n\n## Care\n\nFeed adult cats twice a day and keep fresh water out.\n",
    "dogs.md": "# Dogs\n\nDogs are loyal animals. They bark, fetch sticks and need a walk every day.\n\n## Training\n\nReward good behaviour with treats and praise.\n",
    "cooking.md": "# Cooking pasta\n\nBoil the pasta in salted water for ten minutes, then drain it and add the sauce.\n\n## Sauce\n\nSimmer tomatoes with garlic and basil for twenty minutes.\n",
    "util.py": "import math\n\nLIMIT = 10\n\n\ndef add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n\n\ndef hypotenuse(a, b):\n    \"\"\"Length of the longest side of a right triangle.\"\"\"\n    return math.sqrt(a * a + b * b)\n\n\nclass Counter:\n    def __init__(self):\n        self.n = 0\n\n    def inc(self):\n        self.n += 1\n        return self.n\n",
}


@pytest.fixture
def engine():
    e = LocalRAG()
    for name, text in CORPUS.items():
        e.ingest_text(text, name)
    return e
