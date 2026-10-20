"""CSV files: every row becomes one line of "column: value" pairs, so a row can be retrieved as a unit."""
from __future__ import annotations

import csv
import io

from localrag.core.errors import IngestError
from localrag.ingest.loaders.base import LoadedText, Loader, decode_bytes


class CsvLoader(Loader):
    extensions = (".csv", ".tsv")
    max_rows = 100_000

    def load(self, raw: bytes, name: str) -> LoadedText:
        text = decode_bytes(raw, name)
        delimiter = "\t" if name.lower().endswith(".tsv") else ","
        try:
            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            columns = [c.strip() for c in (reader.fieldnames or []) if c is not None]
            if not columns:
                raise IngestError(f"{name} has no header row")
            lines = []
            for i, row in enumerate(reader):
                if i >= self.max_rows:
                    raise IngestError(f"{name} has more than {self.max_rows} rows")
                pairs = [f"{k.strip()}: {str(v).strip()}" for k, v in row.items() if k is not None and v not in (None, "")]
                if pairs:
                    lines.append("; ".join(pairs))
        except csv.Error as e:
            raise IngestError(f"{name} is not valid CSV: {e}") from e
        return LoadedText("\n".join(lines), {"format": "csv", "title": name, "columns": columns, "rows": len(lines)})
