"""Der Dienst bild-fuellen (Qwen-Image 2.1 Turbo mit Maske) ohne Modell: Quelle und Maske gehen als Bilder an mflux,
die Signalfarbe nie; die Anweisung aus der Vorlage; eingesetzt wird nur in der Markierung; Steckbrief und Dienstname."""

from __future__ import annotations

import io
import threading

from PIL import Image

from bildwerk.bildformat import Aufgabe, Bildauftrag
from bildwerk.modelle import dienstkennung
from bildwerk.modelle import qwen21_fuellen as q
from bildwerk.unterbau import Einstellungen

from test_lama import MAGENTA, foto


class Ergebnis:
    def __init__(self, image: Image.Image) -> None:
        self.image = image


class ScheinModell:
    """Wie QwenImage21Controlnet: liefert ein grünes RGBA-Bild und merkt sich den Aufruf."""

    def __init__(self) -> None:
        self.aufrufe: list[dict] = []

    def generate_image(self, **art) -> Ergebnis:
        self.aufrufe.append(art)
        return Ergebnis(Image.new("RGBA", (art["width"], art["height"]), (0, 200, 0, 255)))


def test_fuellt_nur_die_markierte_flaeche_mit_maske():
    m = q.Motor(Einstellungen())
    m.m = ScheinModell()
    b = foto()
    roh = io.BytesIO()
    b.save(roh, format="PNG")
    [ergebnis] = m.erzeugen(Bildauftrag(beschreibung="a vase", breite=320, hoehe=200, seeds=[3], schritte=8, vorlagen=[roh.getvalue()]),
                            threading.Event())
    assert ergebnis.mode == "RGB" and ergebnis.getpixel((170, 90)) == (0, 200, 0)
    assert ergebnis.getpixel((5, 5)) == b.getpixel((5, 5))
    a = m.m.aufrufe[0]
    assert a["prompt"] == "a vase" and a["num_inference_steps"] == 8 and a["seed"] == 3
    assert a["image_path"].size == a["mask_image"].size == (q.ARBEIT, q.ARBEIT)
    assert MAGENTA not in set(a["image_path"].get_flattened_data())
    assert set(a["mask_image"].get_flattened_data()) <= {(0, 0, 0), (255, 255, 255)}


def test_angaben():
    assert dienstkennung("qwen21-fuellen") == "bild-fuellen"
    a = q.ANGABEN
    assert set(a.aufgaben) == {Aufgabe.ENTFERNEN, Aufgabe.FUELLEN} and "{wunsch}" in a.aufgaben[Aufgabe.FUELLEN].anweisung
    assert a.aufgaben[Aufgabe.ENTFERNEN].signalfarbe == MAGENTA and a.lizenz_hinweis.startswith("Nur nicht")
