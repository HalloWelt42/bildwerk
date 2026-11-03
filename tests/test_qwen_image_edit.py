"""Der Dienst qwen-image-edit mit nachgestelltem Motor: nur mit Vorlage (1 bis 3), Erzeugen ohne Vorlage abgelehnt,
Steckbrief; dazu die eigene, quantisierte Ablage (Prüfliste mit `lokal`) und der echte Motor mit nachgestelltem
mflux-Modell. Kein Modell wird geladen."""

from __future__ import annotations

import base64
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bildwerk.bildformat import Bildauftrag
from bildwerk.bilddienst import erstellen
from bildwerk.gewichte import Gewichtsstand, Pruefliste, aus_ordner, pruefen
from bildwerk.modelle import qwen_image_edit
from bildwerk.unterbau import Einstellungen, GewichteFehlen

from test_dienst import bereit, bild
from test_flux2_klein import vorlage


class ScheinMotor:
    angaben = qwen_image_edit.ANGABEN

    def __init__(self) -> None:
        self.pruefliste = Pruefliste(repo="Qwen/Qwen-Image-Edit-2511", revision="6f3ccc0", lizenz="Apache-2.0",
                                     lokal=qwen_image_edit.LOKAL, dateien=[])
        self.auftraege: list[Bildauftrag] = []

    def gewichte(self) -> Gewichtsstand:
        return Gewichtsstand(ordner="/schein", groesse_gb=36.0, fehlen=[])

    def laden(self) -> None:
        pass

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        self.auftraege.append(auftrag)
        return [Image.new("RGB", (auftrag.breite, auftrag.hoehe)) for _ in auftrag.seeds]


def test_nur_mit_vorlage():
    m = ScheinMotor()
    with TestClient(erstellen("qwen-image-edit", m, Einstellungen())) as c:
        bereit(c)
        ohne = c.post("/v1/images/generations", json={"prompt": "Katze"})
        assert ohne.status_code == 400 and "nur mit Bildvorlage" in ohne.json()["detail"]
        leer = c.post("/v1/images/edits", json={"prompt": "Katze", "images": []})
        assert leer.status_code in (400, 422)
        zu_viele = c.post("/v1/images/edits", json={"prompt": "Katze", "images": [vorlage()] * 4})
        assert zu_viele.status_code == 400 and "Höchstens 3" in zu_viele.json()["detail"]
        r = c.post("/v1/images/edits", json={"prompt": "Die Katze im Stil der Vorlage", "images": [vorlage()], "size": "1024x1024",
                                             "seed": 3})
        assert r.status_code == 200, r.text[:300]
        d = r.json()
    assert bild(d["data"][0]).size == (1024, 1024) and d["x-bildwerk"]["schritte"] == 40
    assert len(m.auftraege) == 1 and len(m.auftraege[0].vorlagen) == 1


def test_steckbrief_nennt_vorlagen_und_lizenz():
    with TestClient(erstellen("qwen-image-edit", ScheinMotor(), Einstellungen())) as c:
        s = c.get("/v1/models").json()["data"][0]["steckbrief"]
    assert s["kann_bearbeiten"] and s["vorlagen_min"] == 1 and s["vorlagen_max"] == 3
    assert s["lizenz"] == "Apache-2.0" and s["lizenz_hinweis"] == "Apache-2.0, frei nutzbar."
    assert s["schritte"] == {"vorgabe": 40, "kleinste": 1, "groesste": 100}


def test_eigene_ablage_pruefliste(tmp_path):
    hf = tmp_path / "huggingface"
    hf.mkdir()
    ziel = tmp_path / qwen_image_edit.LOKAL
    (ziel / "transformer").mkdir(parents=True)
    (ziel / "transformer" / "0.safetensors").write_bytes(b"gewichte")
    (ziel / "config.json").write_text("{}")
    quelle = Pruefliste(repo="Qwen/Qwen-Image-Edit-2511", revision="6f3ccc0", lizenz="Apache-2.0", dateien=[])
    liste = aus_ordner(ziel, quelle, qwen_image_edit.LOKAL)
    assert [d.pfad for d in liste.dateien] == ["config.json", "transformer/0.safetensors"] and liste.lokal == qwen_image_edit.LOKAL
    assert liste.ordner(hf) == ziel and pruefen(liste, hf, pruefsummen=True).vollstaendig
    (ziel / "transformer" / "0.safetensors").write_bytes(b"anders!!")
    assert pruefen(liste, hf, pruefsummen=True).falsch == ["transformer/0.safetensors"]


def test_motor_laedt_nie_ohne_eigene_ablage(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_image_edit, "ablage", lambda: tmp_path / "huggingface")
    monkeypatch.setattr(qwen_image_edit, "PRUEFLISTE", tmp_path / "fehlt.json")
    with pytest.raises(GewichteFehlen, match="bildwerk umwandeln qwen-image-edit"):
        qwen_image_edit.Motor(Einstellungen()).laden()


def test_motor_gibt_vorlagen_als_dateien(monkeypatch, tmp_path):
    gesehen = []

    class Ergebnis:
        def __init__(self, groesse):
            self.image = Image.new("RGB", groesse)

    class Modell:
        def generate_image(self, **kw):
            pfade = kw["image_paths"]
            gesehen.append(pfade)
            assert [Image.open(p).size for p in pfade] == [(64, 48)] and kw["guidance"] == 2.5
            return Ergebnis((kw["width"], kw["height"]))

    monkeypatch.setattr(qwen_image_edit, "PRUEFLISTE", tmp_path / "fehlt.json")
    motor = qwen_image_edit.Motor(Einstellungen())
    motor.m = Modell()
    roh = [base64.b64decode(vorlage())]
    bilder = motor.erzeugen(Bildauftrag(beschreibung="Katze", breite=512, hoehe=512, seeds=[1], schritte=40, vorlagen=roh), threading.Event())
    assert [b.size for b in bilder] == [(512, 512)]
    from pathlib import Path
    assert not any(Path(p).exists() for p in gesehen[0])
    with pytest.raises(ValueError, match="mindestens einer Bildvorlage"):
        motor.erzeugen(Bildauftrag(beschreibung="Katze", breite=512, hoehe=512, seeds=[2], schritte=40), threading.Event())


def test_angaben_wie_mflux():
    from mflux.models.common.config import ModelConfig

    c = ModelConfig.qwen_image_edit_2511()
    a = qwen_image_edit.ANGABEN
    assert a.meint(c.model_name) and qwen_image_edit.mflux_herkunft()[0] == "Qwen/Qwen-Image-Edit-2511"
