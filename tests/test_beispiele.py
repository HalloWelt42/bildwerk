"""Beispiele im Steckbrief (ATL-45): aus dem Portfolio klein als data-URL, je Aufgabe Vorher und Nachher mit Dauer;
ohne Bild im Portfolio kein Beispiel."""

from __future__ import annotations

import base64
import io
import json

from PIL import Image

from bildwerk.beispiele import BREITE, HOEHE, mit_beispielen
from bildwerk.bildformat import Aufgabe, Aufgabenvorlage


def test_beispiel_aus_dem_portfolio(tmp_path):
    (tmp_path / "vorlagen").mkdir()
    (tmp_path / "entfernen").mkdir()
    Image.new("RGB", (960, 640), (255, 0, 255)).save(tmp_path / "vorlagen" / "entfernen.webp")
    Image.new("RGB", (960, 640), (20, 120, 40)).save(tmp_path / "entfernen" / "bild-entfernen.webp")
    (tmp_path / "entfernen" / "bild-entfernen.json").write_text(json.dumps({"dauer_s": 12.5}))
    (tmp_path / "beispiele.toml").write_text('[aufgabe.entfernen]\ntitel = "Bank am See"\n')
    vorlagen = {Aufgabe.ENTFERNEN: Aufgabenvorlage(anweisung="Remove.", sprache="en"),
                Aufgabe.FUELLEN: Aufgabenvorlage(anweisung="Fill {wunsch}.", sprache="en")}
    mit = mit_beispielen(vorlagen, "bild-entfernen", tmp_path)
    b = mit[Aufgabe.ENTFERNEN].beispiel
    assert b is not None and b.motiv == "Bank am See" and b.dauer_s == 12.5
    klein = Image.open(io.BytesIO(base64.b64decode(b.nachher.split(",", 1)[1])))
    assert klein.format == "WEBP" and klein.width <= BREITE and klein.height <= HOEHE
    assert mit[Aufgabe.FUELLEN].beispiel is None                       # kein Bild im Portfolio: kein Beispiel
    assert mit_beispielen(vorlagen, "anderer-dienst", tmp_path)[Aufgabe.ENTFERNEN].beispiel is None


def test_portfolio_im_repo_passt_zu_den_diensten():
    """Das mitgelieferte Portfolio: Bank am See für bild-entfernen-schnell, klein genug für den Steckbrief."""
    v = mit_beispielen({Aufgabe.ENTFERNEN: Aufgabenvorlage(anweisung="Remove.", sprache="en")}, "bild-entfernen-schnell")
    b = v[Aufgabe.ENTFERNEN].beispiel
    assert b is not None and b.motiv and len(b.vorher) + len(b.nachher) < 120_000
