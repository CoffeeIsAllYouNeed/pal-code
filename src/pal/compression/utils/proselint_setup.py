from __future__ import annotations

from proselint.checks import __register__
from proselint.registry import CheckRegistry


class ProselintSetup:
    def register_checks(self) -> None:
        CheckRegistry().register_many(__register__)
