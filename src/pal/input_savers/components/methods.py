"""The input-token-saving methods. Each is a single rule handed to a
local LLM; nothing here edits text itself. Order is the order applied."""
from pal.input_savers.components.base import Method

_KEEP = (
    " Keep everything else exactly as written: same words, order, numbers, "
    "names, line breaks. Tokens like \u27e6P0\u27e7 are protected: copy them unchanged."
)

METHODS = {
    "remove_articles": Method(
        "remove_articles", "Remove articles",
        "Delete the articles 'a', 'an' and 'the' (any capitalisation). Delete nothing else." + _KEEP,
    ),
    "remove_punctuation": Method(
        "remove_punctuation", "Remove punctuation",
        "Delete punctuation marks (. , ; : ! ? quotes, brackets, dashes, hyphens, apostrophes, slashes). "
        "Keep punctuation that is part of a number (3.5, 1,000). Do not add, delete or change any letter or digit." + _KEEP,
    ),
    "remove_fillers": Method(
        "remove_fillers", "Remove filler words",
        "Delete filler words and phrases that carry no meaning, such as: please, just, really, very, "
        "basically, actually, simply, quite, kind of, sort of, you know, I think, um, uh. "
        "Never delete negations (not, no, never, n't), numbers, names, requirements or the request itself. "
        "Only delete; do not reword or add words." + _KEEP,
    ),
    "abbreviate": Method(
        "abbreviate", "Abbreviate long words",
        "Replace long words and phrases with shorter, widely understood abbreviations or short synonyms "
        "(information->info, for example->e.g., application->app, approximately->approx). "
        "Do not abbreviate names, identifiers, technical terms, or anything whose meaning could become ambiguous. "
        "Do not delete anything else." + _KEEP,
    ),
    "collapse_whitespace": Method(
        "collapse_whitespace", "Remove extra whitespace",
        "Collapse runs of spaces and tabs into one space, remove trailing spaces, trim the start and end, "
        "and collapse runs of blank lines into one blank line. Change no characters other than whitespace." + _KEEP,
    ),
}

METHOD_ORDER = list(METHODS)
