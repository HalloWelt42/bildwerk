"""Der Dienst bild-erzeugen (FLUX 3 Image Klein) ohne Modell: erzeugt aus Text, lehnt Vorlagen ab (Bearbeiten ist
zurückgestellt), Steckbrief und Dienstname nach Fähigkeit."""

from __future__ import annotations

import threading

import pytest
from PIL import Image

from bildwerk.bildformat import Bildauftrag
from bildwerk.modelle import dienstkennung
from bildwerk.modelle import flux3_klein as f3
from bildwerk.unterbau import Einstellungen


class Ergebnis:
    def __init__(self, image: Image.Image) -> None:
        self.image = image


class ScheinModell:
    def __init__(self) -> None:
        self.aufrufe: list[dict] = []

    def generate_image(self, **art) -> Ergebnis:
        self.aufrufe.append(art)
        return Ergebnis(Image.new("RGB", (art["width"], art["height"]), (10, 20, 30)))


def test_erzeugt_aus_text_und_lehnt_vorlagen_ab():
    m = f3.Motor(Einstellungen())
    m.m = ScheinModell()
    bilder = m.erzeugen(Bildauftrag(beschreibung="a lake", breite=512, hoehe=384, seeds=[1, 2], schritte=16, vorlagen=[]), threading.Event())
    assert [b.size for b in bilder] == [(512, 384)] * 2 and m.m.aufrufe[0]["num_inference_steps"] == 16
    with pytest.raises(ValueError):
        m.erzeugen(Bildauftrag(beschreibung="x", breite=512, hoehe=384, seeds=[1], schritte=16, vorlagen=[b"png"]), threading.Event())


def test_angaben():
    assert dienstkennung("flux3-klein") == "bild-erzeugen"
    assert not f3.ANGABEN.kann_bearbeiten and f3.ANGABEN.schritte.vorgabe == 16 and f3.ANGABEN.lizenz_hinweis
