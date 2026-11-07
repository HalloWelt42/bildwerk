"""Verzeichnis der Bildmodelle: je Modell ein Modul mit PORT, ANGABEN, PRUEFLISTE, Motor und herkunft() (Repo, Revision,
Dateimuster der Gewichte); wahlweise DIENST (Name nach Fähigkeit) und RECHENWERK (ohne Angabe mflux).

Ein neues Modell (etwa Qwen-Image, Z-Image) ist ein weiteres Modul nach dem Muster von `moss_image`
und eine Zeile hier; Schnittstelle, Unterbau und Dienstdatei bleiben gleich.
"""

from __future__ import annotations

import importlib
from types import ModuleType

# Kennung eines Dienstes im Maschinenraum (Name seiner Dienstdatei): bildwerk-<modell>, oder der DIENST des Moduls
DIENST_PRAEFIX = "bildwerk-"

MODELLE: dict[str, str] = {
    "moss-image": "bildwerk.modelle.moss_image",
    "flux2-klein": "bildwerk.modelle.flux2_klein",
    "qwen-image-edit": "bildwerk.modelle.qwen_image_edit",
    "qwen-image-edit-schnell": "bildwerk.modelle.qwen_image_edit_schnell",
    "lama": "bildwerk.modelle.lama",
}


def modul(kennung: str) -> ModuleType:
    return importlib.import_module(MODELLE[kennung])


def dienstkennung(kennung: str) -> str:
    """Name des Dienstes im Maschinenraum: was das Modul als DIENST nennt (nach Fähigkeit und Eigenschaft, HELF-189),
    sonst bildwerk-<modell>."""
    return getattr(modul(kennung), "DIENST", DIENST_PRAEFIX + kennung)
