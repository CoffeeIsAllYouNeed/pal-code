import re

from pal.compression.components.output_compressor import OutputCompressor

try:
    from pal.compression.components.minifier import TextMinifier
    _MINIFIER_AVAILABLE = True
except Exception:
    # NLTK/spaCy data not downloaded yet, or a dependency failed to
    # import for some other reason. Fall back to the old whitespace/
    # dedupe-only behavior rather than crashing the whole pipeline --
    # real compression kicks in once install.sh's NLTK/spaCy download
    # steps have run.
    _MINIFIER_AVAILABLE = False


class CompressionPipeline:
    def __init__(
        self,
        strip_whitespace: bool = True,
        dedupe_lines: bool = True,
        use_minifier: bool = True,
        max_blank_lines: int = 1,
        trim_trailing: bool = True,
    ):
        self.strip_whitespace = strip_whitespace
        self.dedupe_lines = dedupe_lines
        self.use_minifier = use_minifier and _MINIFIER_AVAILABLE
        self._minifier = TextMinifier() if self.use_minifier else None
        self._output_compressor = OutputCompressor(max_blank_lines=max_blank_lines, trim_trailing=trim_trailing)

    def compress_input(self, text: str) -> str:
        out = text

        # Real token-reducing step: strips hedges, pleonasms,
        # interjections, filler adverbs, stopwords, and normalizes
        # punctuation.
        if self.use_minifier:
            out = self._minifier.minify(out)

        if self.strip_whitespace:
            out = re.sub(r"[ \t]+", " ", out)
            out = re.sub(r"\n{3,}", "\n\n", out)
            out = out.strip()
        if self.dedupe_lines:
            seen = set()
            lines = []
            for line in out.split("\n"):
                key = line.strip()
                if key and key in seen:
                    continue
                seen.add(key)
                lines.append(line)
            out = "\n".join(lines)
        return out

    def compress_output(self, text: str) -> str:
        return self._output_compressor.compress(text)
