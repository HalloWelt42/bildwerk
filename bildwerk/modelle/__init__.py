"""Verzeichnis der Bildmodelle: je Modell ein Modul mit PORT, ANGABEN, PRUEFLISTE, Motor und mflux_herkunft().

Ein neues Modell (etwa FLUX.2 klein, Qwen-Image, Z-Image) ist ein weiteres Modul nach dem Muster von `moss_image`
und eine Zeile hier; Schnittstelle, Unterbau und Dienstdatei bleiben gleich.
"""

from __future__ import annotations

import importlib
from types import ModuleType

MODELLE: dict[str, str] = {
    "moss-image": "bildwerk.modelle.moss_image",
}


def modul(kennung: str) -> ModuleType:
    return importlib.import_module(MODELLE[kennung])
