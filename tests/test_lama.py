"""Der Dienst bild-entfernen-schnell (LaMa) ohne Modell: Maske aus der Signalfarbe (auch weich umgerechnet), Ausschnitt
im Bild, Rückmischen nur in der markierten Fläche, der echte Motor mit nachgestelltem Core-ML-Modell, Steckbrief und
Dienstname nach Fähigkeit. Kein Modell wird geladen."""

from __future__ import annotations

import base64
import io
import threading

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from bildwerk.bildformat import Aufgabe, Bildauftrag
from bildwerk.bilddienst import erstellen
from bildwerk.gewichte import Gewichtsstand
from bildwerk.modelle import dienstkennung, lama
from bildwerk.unterbau import Einstellungen

from test_dienst import bereit, bild

MAGENTA = (255, 0, 255)


def foto(breite: int = 320, hoehe: int = 200) -> Image.Image:
    """Ein Verlauf mit einem markierten Kreis in der Signalfarbe, weich umgerechnet wie vom Atelier."""
    b = Image.new("RGB", (breite, hoehe))
    for x in range(breite):
        ImageDraw.Draw(b).line([(x, 0), (x, hoehe)], fill=(x % 256, 120, 60))
    ImageDraw.Draw(b).ellipse((140, 60, 200, 120), fill=MAGENTA)
    return b.resize((breite * 2, hoehe * 2), Image.BILINEAR).resize((breite, hoehe), Image.BILINEAR)


def als_b64(b: Image.Image) -> str:
    puffer = io.BytesIO()
    b.save(puffer, format="PNG")
    return base64.b64encode(puffer.getvalue()).decode("ascii")


class ScheinModell:
    """Wie ein Core-ML-Modell: füllt den Ausschnitt einfarbig grün und merkt sich, was es bekam."""

    def __init__(self) -> None:
        self.eingaben: list[dict] = []

    def predict(self, eingabe: dict) -> dict:
        self.eingaben.append(eingabe)
        return {"output": Image.new("RGB", eingabe["image"].size, (0, 200, 0))}


def test_maske_aus_weicher_signalfarbe_und_ausschnitt():
    b = foto()
    m = lama.maske_aus_signalfarbe(b, MAGENTA)
    x0, y0, x1, y1 = m.getbbox()
    # der ganze Kreis samt weichem Rand ist markiert, nichts weit daneben
    assert x0 <= 140 and y0 <= 60 and x1 >= 200 and y1 >= 120 and x1 - x0 < 80
    assert m.getpixel((20, 20)) == 0 and m.getpixel((170, 90)) == 255
    k = lama.ausschnitt(m)
    assert k[0] >= 0 and k[1] >= 0 and k[2] <= b.width and k[3] <= b.height
    assert k[0] <= x0 and k[1] <= y0 and k[2] >= x1 and k[3] >= y1
    assert lama.ausschnitt(Image.new("L", (50, 50))) is None


def test_motor_fuellt_nur_die_markierte_flaeche():
    motor = lama.Motor(Einstellungen())
    motor.m = ScheinModell()
    b = foto()
    roh = io.BytesIO()
    b.save(roh, format="PNG")
    [ergebnis] = motor.erzeugen(Bildauftrag(beschreibung="", breite=320, hoehe=200, seeds=[1], schritte=1, vorlagen=[roh.getvalue()]),
                                threading.Event())
    assert ergebnis.size == (320, 200)
    assert ergebnis.getpixel((170, 90)) == (0, 200, 0)                  # markiert: gefüllt
    assert ergebnis.getpixel((20, 20)) == b.getpixel((20, 20))          # außerhalb: unverändert
    [eingabe] = motor.m.eingaben
    assert eingabe["image"].size == (lama.ARBEIT, lama.ARBEIT) and eingabe["mask"].mode == "L"
    # keine Signalfarbe im Modell: die markierte Fläche ist neutral
    px = eingabe["image"].convert("RGB").getdata()
    assert MAGENTA not in set(px)


def test_ohne_markierung_eine_klare_meldung():
    motor = lama.Motor(Einstellungen())
    motor.m = ScheinModell()
    roh = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(roh, format="PNG")
    try:
        motor.erzeugen(Bildauftrag(beschreibung="", breite=64, hoehe=64, seeds=[1], schritte=1, vorlagen=[roh.getvalue()]), threading.Event())
    except ValueError as f:
        assert "Signalfarbe" in str(f)
    else:
        raise AssertionError("ohne Markierung muss der Motor ablehnen")


class ScheinMotor(lama.Motor):
    def gewichte(self) -> Gewichtsstand:
        return Gewichtsstand(ordner="/schein", groesse_gb=0.2, fehlen=[])

    def laden(self) -> None:
        self.m = ScheinModell()


def test_dienst_steckbrief_und_eine_variante():
    assert dienstkennung("lama") == "bild-entfernen-schnell"
    with TestClient(erstellen("lama", ScheinMotor(Einstellungen()), Einstellungen())) as c:
        bereit(c)
        s = c.get("/v1/models").json()["data"][0]["steckbrief"]
        assert s["anzahl_max"] == 1 and s["quantisierung"] == "fp32" and s["rechenwerk"]["name"] == "coremltools"
        assert s["vorlagen_min"] == s["vorlagen_max"] == 1 and s["eignung"] == ["schnell"]
        assert list(s["aufgaben"]) == [Aufgabe.ENTFERNEN.value] and s["aufgaben"]["entfernen"]["signalfarbe"] == list(MAGENTA)
        assert c.post("/v1/images/edits", json={"prompt": "Remove the marked area.", "images": [als_b64(foto())], "size": "320x200", "n": 2}).status_code == 400
        r = c.post("/v1/images/edits", json={"prompt": "Remove the marked area.", "images": [als_b64(foto())], "size": "320x200",
                                             "model": "bild-entfernen-schnell"})
        assert r.status_code == 200, r.text
        assert bild(r.json()["data"][0]).getpixel((170, 90)) == (0, 200, 0)
