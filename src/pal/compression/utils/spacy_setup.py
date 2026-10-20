from __future__ import annotations

import spacy
from spacy.cli import download


class SpacySetup:
    MODEL: str = "en_core_web_sm"

    def _is_model_available(self) -> bool:
        return spacy.util.is_package(self.MODEL)

    def _download_model(self) -> None:
        download(self.MODEL)

    def ensure_model(self) -> None:
        if not self._is_model_available():
            self._download_model()

    def load_model(self):
        self.ensure_model()
        return spacy.load(self.MODEL)
