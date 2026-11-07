"""Der Dienst bild-entfernen-osor (OSOR auf SDXL-Inpainting) ohne Modell: eingesetzt wird, wo OSOR entfernen will (auch
über die Maske hinaus, etwa ein Schatten), mindestens aber die markierte Fläche; Steckbrief und Dienstname."""

from __future__ import annotations

import io
import threading

from PIL import Image, ImageDraw

from bildwerk.bildformat import Aufgabe, Bildauftrag
from bildwerk.modelle import dienstkennung
from bildwerk.modelle import osor_sdxl_entfernen as osor
from bildwerk.unterbau import Einstellungen

from test_lama import MAGENTA, foto


class ScheinMotor(osor.Motor):
    """Rechnet nicht: füllt grün und meldet als Alpha einen Streifen rechts neben der Maske (wie ein Schatten)."""

    def __init__(self) -> None:
        super().__init__(Einstellungen())
        self.eingaben: list[tuple[Image.Image, Image.Image, int]] = []

    def entfernen(self, teil: Image.Image, teilmaske: Image.Image, seed: int) -> tuple[Image.Image, Image.Image]:
        self.eingaben.append((teil, teilmaske, seed))
        alpha = Image.new("L", teil.size)
        x0, y0, x1, y1 = teilmaske.getbbox()
        ImageDraw.Draw(alpha).rectangle((x1, y0, min(teil.width - 1, x1 + 120), y1), fill=255)
        return Image.new("RGB", teil.size, (0, 200, 0)), alpha


def test_eingesetzt_wird_alpha_und_maske():
    m = ScheinMotor()
    b = foto()
    roh = io.BytesIO()
    b.save(roh, format="PNG")
    [ergebnis] = m.erzeugen(Bildauftrag(beschreibung="", breite=320, hoehe=200, seeds=[7], schritte=1, vorlagen=[roh.getvalue()]), threading.Event())
    assert ergebnis.getpixel((170, 90)) == (0, 200, 0)                   # markiert: gefüllt
    assert ergebnis.getpixel((210, 90)) == (0, 200, 0)                   # rechts daneben, wo OSOR mehr entfernen will
    assert ergebnis.getpixel((5, 5)) == b.getpixel((5, 5))               # sonst unverändert
    teil, teilmaske, seed = m.eingaben[0]
    assert teil.size == teilmaske.size == (osor.ARBEIT, osor.ARBEIT) and seed == 7
    assert MAGENTA not in set(teil.get_flattened_data())


def test_angaben():
    assert dienstkennung("osor-sdxl-entfernen") == "bild-entfernen-osor"
    a = osor.ANGABEN
    assert list(a.aufgaben) == [Aufgabe.ENTFERNEN] and a.vorlagen_min == a.vorlagen_max == 1 and a.lizenz == "CreativeML Open RAIL++-M"
