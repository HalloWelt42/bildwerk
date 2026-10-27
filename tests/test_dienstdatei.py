"""Die Dienstdateien unter deploy/ passen zu den Modellen: je Modell genau eine, Port, Befehl, Fähigkeiten und
Selbstauskunft (Eignung, Sprachen) wie im Modul des Modells, das Netz gesperrt."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from bildwerk.modelle import MODELLE, dienstkennung, modul
from bildwerk.unterbau import Quantisierung

DEPLOY = Path(__file__).resolve().parent.parent / "deploy"


def test_je_modell_eine_dienstdatei():
    assert sorted(p.stem for p in DEPLOY.glob("bildwerk-*.toml")) == sorted(dienstkennung(k) for k in MODELLE)


@pytest.mark.parametrize("kennung", sorted(MODELLE))
def test_dienstdatei_passt_zum_modell(kennung: str):
    m = modul(kennung)
    d = tomllib.loads((DEPLOY / f"{dienstkennung(kennung)}.toml").read_text(encoding="utf-8"))
    assert d["befehl"] == [".venv/bin/python", "-m", "bildwerk", kennung, "--port", "$MASCHINENRAUM_PORT"]
    assert d["port"] == m.PORT and d["pruefadresse"] == f"http://127.0.0.1:{m.PORT}/gesundheit"
    assert d["faehigkeiten"] == m.ANGABEN.faehigkeiten
    assert d["eignung"] == m.ANGABEN.eignung and d["sprachen"] == m.ANGABEN.sprachen
    assert d["umgebung"]["HF_HUB_OFFLINE"] == "1"
    Quantisierung(d["umgebung"]["BILDWERK_QUANTISIERUNG"])
    assert d["speicher_gb"] > 0 and d["startzeit_s"] > 0
