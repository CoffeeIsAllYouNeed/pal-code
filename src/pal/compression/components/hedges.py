from __future__ import annotations

from proselint.tools import DEFAULT, LintFile


class Hedges:
    RELEVANT_PREFIX: str = "hedging"

    def _lint(self, text: str) -> list:
        lint_file = LintFile("<memory>", text)
        return [
            result
            for result in lint_file.lint(DEFAULT)
            if result.check_result.check_path.startswith(self.RELEVANT_PREFIX)
        ]

    def _extract_edits(self, results: list) -> list[tuple[int, int, str | None]]:
        return sorted(
            (
                (
                    result.check_result.span[0] - 1,
                    result.check_result.span[1] - 1,
                    result.check_result.replacements,
                )
                for result in results
            ),
            key=lambda edit: (edit[0], -(edit[1] - edit[0])),
        )

    def _resolve_overlaps(
        self, edits: list[tuple[int, int, str | None]]
    ) -> list[tuple[int, int, str | None]]:
        resolved: list[tuple[int, int, str | None]] = []
        last_end = -1
        for start, end, replacement in edits:
            if start < last_end:
                continue
            resolved.append((start, end, replacement))
            last_end = end
        return resolved

    def _apply_edits(
        self, text: str, edits: list[tuple[int, int, str | None]]
    ) -> str:
        cleaned = text
        for start, end, replacement in reversed(edits):
            cleaned = cleaned[:start] + (replacement or "") + cleaned[end:]
        return cleaned

    def remove(self, text: str) -> str:
        results = self._lint(text)
        edits = self._extract_edits(results)
        resolved_edits = self._resolve_overlaps(edits)
        return self._apply_edits(text, resolved_edits)
