import re


class OutputCompressor:
    def __init__(self, max_blank_lines: int = 1, trim_trailing: bool = True):
        self.max_blank_lines = max_blank_lines
        self.trim_trailing = trim_trailing

    def compress(self, text: str) -> str:
        out = re.sub(r"\n{" + str(self.max_blank_lines + 2) + ",}", "\n" * (self.max_blank_lines + 1), text)
        if self.trim_trailing:
            out = "\n".join(line.rstrip() for line in out.split("\n"))
        return out.strip()
