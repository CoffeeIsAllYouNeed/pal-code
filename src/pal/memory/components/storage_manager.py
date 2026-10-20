"""
Minimal disk-backed persistence for MemoryNotes: one JSON file per note
under `root`, keyed by note id.

Stands in for the full Storage Manager AIOS section 3.6 describes (a
general read/write file-access layer for agents) -- this project
doesn't implement that broader module yet, only the narrow slice the
memory manager's RAM<->disk swapping needs: durably persist a note,
load it back by id, delete it, and enumerate every persisted id.
"""
import json
from pathlib import Path
from typing import Optional

from pal.memory.components.note import MemoryNote


class StorageManager:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, note_id: str) -> Path:
        return self.root / f"{note_id}.json"

    def write(self, note: MemoryNote) -> None:
        with open(self._path(note.id), "w", encoding="utf-8") as f:
            json.dump(note.to_dict(), f)

    def read(self, note_id: str) -> Optional[MemoryNote]:
        path = self._path(note_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return MemoryNote.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            # Corrupt/partial file (e.g. killed mid-write) -- treat as
            # missing rather than raising, same tolerance the old
            # single-file store had for a corrupt memory.json.
            return None

    def delete(self, note_id: str) -> None:
        path = self._path(note_id)
        if path.exists():
            path.unlink()

    def exists(self, note_id: str) -> bool:
        return self._path(note_id).exists()

    def list_ids(self) -> list[str]:
        return [p.stem for p in self.root.glob("*.json")]

    def read_all(self) -> list[MemoryNote]:
        notes = []
        for note_id in self.list_ids():
            note = self.read(note_id)
            if note is not None:
                notes.append(note)
        return notes
