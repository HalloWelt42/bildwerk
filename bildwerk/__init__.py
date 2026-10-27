"""Bildwerk: Bilddienste je Modell. Jeder Dienst ist ein eigener Prozess mit genau einem Bildmodell.

Die Fassung steht in `version.json` im Wurzelordner des Repos (einzige Quelle; installiert wird nur editierbar).
"""

from __future__ import annotations

import json
from pathlib import Path


def _fassung() -> str:
    quelle = Path(__file__).resolve().parent.parent / "version.json"
    if quelle.is_file():
        return json.loads(quelle.read_text(encoding="utf-8"))["version"]
    return "0.0.0"


__version__ = _fassung()
