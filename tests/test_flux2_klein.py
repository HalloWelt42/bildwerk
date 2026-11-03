"""Der Dienst flux2-klein mit nachgestelltem Motor: Bearbeiten mit Bildvorlagen, Grenzen und Steckbrief; dazu der echte
Motor mit nachgestelltem mflux-Modell (Vorlagen als Dateien, danach aufgeräumt). Kein Modell wird geladen."""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bildwerk.bildformat import Bildauftrag
from bildwerk.bilddienst import erstellen
from bildwerk.gewichte import Gewichtsstand, Pruefliste
from bildwerk.modelle import flux2_klein
from bildwerk.unterbau import Einstellungen, GewichteFehlen

from test_dienst import bereit, bild


class ScheinMotor:
    angaben = flux2_klein.ANGABEN

    def __init__(self) -> None:
        self.pruefliste = Pruefliste.lesen(flux2_klein.PRUEFLISTE)
        self.auftraege: list[Bildauftrag] = []

    def gewichte(self) -> Gewichtsstand:
        return Gewichtsstand(ordner="/schein", groesse_gb=34.7, fehlen=[])

    def laden(self) -> None:
        pass

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        self.auftraege.append(auftrag)
        return [Image.new("RGB", (auftrag.breite, auftrag.hoehe)) for _ in auftrag.seeds]


def vorlage(farbe=(200, 80, 20), fmt="PNG") -> str:
    puffer = io.BytesIO()
    Image.new("RGB", (64, 48), farbe).save(puffer, format=fmt)
    return base64.b64encode(puffer.getvalue()).decode("ascii")


def test_bearbeiten_mit_vorlagen():
    m = ScheinMotor()
    with TestClient(erstellen("flux2-klein", m, Einstellungen())) as c:
        bereit(c)
        r = c.post("/v1/images/edits", json={"prompt": "Eine Katze im Stil der Vorlage", "images": [vorlage(), vorlage(fmt="JPEG")],
                                             "size": "768x512", "seed": 7})
        assert r.status_code == 200, r.text[:300]
        d = r.json()
    assert bild(d["data"][0]).size == (768, 512)
    assert d["x-bildwerk"] | {"dauer_ms": 0} == {"modell": "flux2-klein", "seed": 7, "seeds": [7], "schritte": 4, "groesse": "768x512",
                                                 "dauer_ms": 0}
    a = m.auftraege[0]
    assert len(a.vorlagen) == 2 and a.vorlagen[0][:8] == b"\x89PNG\r\n\x1a\n" and a.vorlagen[1][:2] == b"\xff\xd8"


def test_erzeugen_ohne_vorlage():
    m = ScheinMotor()
    with TestClient(erstellen("flux2-klein", m, Einstellungen())) as c:
        bereit(c)
        r = c.post("/v1/images/generations", json={"prompt": "Leuchtturm"})
    assert r.status_code == 200 and m.auftraege[0].vorlagen == [] and m.auftraege[0].schritte == 4


@pytest.mark.parametrize("koerper, code, text", [
    ({"images": [vorlage()] * 5}, 422, ""),                       # mehr als 4 Vorlagen: schon die Form passt nicht
    ({"images": []}, 422, ""),
    ({"images": ["kein base64!"]}, 400, "Vorlage 1"),
    ({"images": [base64.b64encode(b"keinbild").decode()]}, 400, "kein lesbares Bild"),
    ({"images": [vorlage()], "size": "4096x4096"}, 400, "Kante"),
    ({"images": [vorlage()], "steps": 51}, 400, "Schritte"),
    ({"images": [vorlage()], "model": "moss-image"}, 409, ""),
])
def test_bearbeiten_fehler(koerper: dict, code: int, text: str):
    m = ScheinMotor()
    with TestClient(erstellen("flux2-klein", m, Einstellungen())) as c:
        bereit(c)
        r = c.post("/v1/images/edits", json={"prompt": "Katze"} | koerper)
    assert r.status_code == code, r.text[:300]
    assert text in r.text and not m.auftraege


def test_bearbeiten_ohne_json():
    with TestClient(erstellen("flux2-klein", ScheinMotor(), Einstellungen())) as c:
        bereit(c)
        r = c.post("/v1/images/edits", content=b"kein json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and "JSON" in r.json()["detail"]


def test_steckbrief_mit_vorlagen_und_lizenzhinweis():
    with TestClient(erstellen("flux2-klein", ScheinMotor(), Einstellungen())) as c:
        bereit(c)
        s = c.get("/v1/models").json()["data"][0]["steckbrief"]
    assert s["kann_bearbeiten"] is True and s["vorlagen_max"] == 4 and s["eignung"] == ["schnell"]
    assert s["lizenz"] == "FLUX Non-Commercial License" and s["lizenz_hinweis"] == "Nur nicht-kommerzielle Nutzung der Bilder."
    assert s["schritte"] == {"vorgabe": 4, "kleinste": 1, "groesste": 50} and s["guidance"] == 1.0
    assert s["quelle"]["repo"] == "black-forest-labs/FLUX.2-klein-9B"


def test_motor_gibt_vorlagen_als_dateien_und_raeumt_auf():
    gesehen: list[list[Path] | None] = []

    class Ergebnis:
        def __init__(self, groesse):
            self.image = Image.new("RGB", groesse)

    class Modell:
        prompt_cache: dict = {}

        def generate_image(self, **kw):
            pfade = kw["image_paths"]
            gesehen.append(pfade)
            if pfade:
                assert [Image.open(p).size for p in pfade] == [(64, 48), (64, 48)]
            self.prompt_cache[kw["prompt"]] = "kodiert"
            return Ergebnis((kw["width"], kw["height"]))

    motor = flux2_klein.Motor(Einstellungen())
    motor.m = Modell()
    roh = [base64.b64decode(vorlage()), base64.b64decode(vorlage((0, 0, 0)))]
    bilder = motor.erzeugen(Bildauftrag(beschreibung="Katze", breite=512, hoehe=512, seeds=[1, 2], schritte=4, vorlagen=roh), threading.Event())
    assert [b.size for b in bilder] == [(512, 512)] * 2 and len(gesehen) == 2
    assert not any(p.exists() for p in gesehen[0]) and motor.m.prompt_cache == {}
    motor.erzeugen(Bildauftrag(beschreibung="Katze", breite=512, hoehe=512, seeds=[3], schritte=4), threading.Event())
    assert gesehen[-1] is None


def test_motor_laedt_nie_ohne_vollstaendige_gewichte(tmp_path, monkeypatch):
    monkeypatch.setattr(flux2_klein, "ablage", lambda: tmp_path)
    with pytest.raises(GewichteFehlen, match="bildwerk gewichte flux2-klein --laden"):
        flux2_klein.Motor(Einstellungen()).laden()


def test_angaben_wie_mflux():
    from mflux.models.common.config import ModelConfig

    c = ModelConfig.flux2_klein_9b()
    a = flux2_klein.ANGABEN
    assert c.max_sequence_length == a.beschreibung_max_token and c.supports_guidance
