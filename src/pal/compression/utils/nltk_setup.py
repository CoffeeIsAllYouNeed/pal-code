from __future__ import annotations

import nltk


class NltkSetup:
    REQUIRED_PACKAGES: tuple[str, ...] = (
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "stopwords",
    )

    def _download_package(self, package: str) -> None:
        nltk.download(package, quiet=True)

    def download_all(self) -> None:
        for package in self.REQUIRED_PACKAGES:
            self._download_package(package)
