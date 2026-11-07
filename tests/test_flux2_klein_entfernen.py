"""Der Dienst bild-entfernen (FLUX.2 klein 4B mit erase-v1) ohne Modell: Auswahl wächst nach der Modellkarte, Foto und
Maske gehen als Bezugsbilder in der verlangten Reihenfolge, nur die markierte Fläche wird eingesetzt, Steckbrief und
Dienstname nach Fähigkeit. Kein Modell wird geladen."""

from __future__ import annotations

import io
import threading
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from bildwerk.bildformat import Aufgabe, Bildauftrag
from bildwerk.bilddienst import erstellen
from bildwerk.gewichte import Gewichtsstand
from bildwerk.modelle import dienstkennung
from bildwerk.modelle import flux2_klein_entfernen as fke
from bildwerk.unterbau import Einstellungen

from test_dienst import bereit, bild
from test_lama import MAGENTA, als_b64, foto


class Ergebnis:
    def __init__(self, image: Image.Image) -> None:
        self.image = image


class ScheinModell:
    """Wie Flux2KleinEdit: liefert ein grünes Bild und merkt sich die Bezugsbilder (Inhalt, nicht nur Pfad)."""

    def __init__(self) -> None:
        self.aufrufe: list[dict] = []

    def generate_image(self, **art) -> Ergebnis:
        bilder = [Image.open(Path(p)).convert("RGB") for p in art["image_paths"]]
        self.aufrufe.append(art | {"bilder": bilder})
        return Ergebnis(Image.new("RGB", (art["width"], art["height"]), (0, 200, 0)))


def motor() -> fke.Motor:
    m = fke.Motor(Einstellungen())
    m.m = ScheinModell()
    return m


def test_auswahl_waechst_nach_der_modellkarte():
    b = foto()
    eng = fke.maske_aus_signalfarbe(b, MAGENTA)
    maske, kasten = fke.ausschnitt_fuer(eng)
    assert sum(1 for p in maske.get_flattened_data() if p) > sum(1 for p in eng.get_flattened_data() if p)
    x0, y0, x1, y1 = maske.getbbox()
    assert kasten[0] <= x0 and kasten[1] <= y0 and kasten[2] >= x1 and kasten[3] >= y1
    assert fke.ausschnitt_fuer(Image.new("L", (50, 50))) is None


def test_foto_und_maske_als_bezugsbilder_und_nur_die_flaeche_gefuellt():
    m = motor()
    b = foto()
    roh = io.BytesIO()
    b.save(roh, format="PNG")
    bilder = m.erzeugen(Bildauftrag(beschreibung="", breite=320, hoehe=200, seeds=[1, 2], schritte=4, vorlagen=[roh.getvalue()]),
                        threading.Event())
    assert len(bilder) == 2 and bilder[0].size == (320, 200)
    assert bilder[0].getpixel((170, 90)) == (0, 200, 0)                 # markiert: gefüllt
    assert bilder[0].getpixel((5, 5)) == b.getpixel((5, 5))              # außerhalb: unverändert
    aufruf = m.m.aufrufe[0]
    assert aufruf["num_inference_steps"] == 4 and aufruf["guidance"] == 1.0
    foto_, maske = aufruf["bilder"]
    assert foto_.size == maske.size == (fke.ARBEIT, fke.ARBEIT)
    assert MAGENTA not in set(foto_.get_flattened_data())                          # keine Signalfarbe im Modell
    farben = set(maske.get_flattened_data())
    assert farben <= {(0, 0, 0), (255, 255, 255)} and (255, 255, 255) in farben


class ScheinMotor(fke.Motor):
    def gewichte(self) -> Gewichtsstand:
        return Gewichtsstand(ordner="/schein", groesse_gb=9.0, fehlen=[])

    def laden(self) -> None:
        self.m = ScheinModell()


def test_dienst_steckbrief():
    assert dienstkennung("flux2-klein-entfernen") == "bild-entfernen"
    with TestClient(erstellen("flux2-klein-entfernen", ScheinMotor(Einstellungen()), Einstellungen())) as c:
        bereit(c)
        s = c.get("/v1/models").json()["data"][0]["steckbrief"]
        assert s["vorlagen_min"] == s["vorlagen_max"] == 1 and s["lizenz"] == "Apache-2.0"
        assert list(s["aufgaben"]) == [Aufgabe.ENTFERNEN.value] and s["aufgaben"]["entfernen"]["signalfarbe"] == list(MAGENTA)
        r = c.post("/v1/images/edits", json={"prompt": "Remove the marked area.", "images": [als_b64(foto())], "size": "320x200",
                                             "model": "bild-entfernen"})
        assert r.status_code == 200, r.text
        assert bild(r.json()["data"][0]).getpixel((170, 90)) == (0, 200, 0)
