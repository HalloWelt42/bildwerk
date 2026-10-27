"""Der Dienst moss-image mit nachgestelltem Motor: Schnittstelle im üblichen Bildformat, Steckbrief, Gesundheit,
Probemodus, Fehlerfälle und Abbruch. Kein Modell wird geladen, kein Bild gerechnet."""

from __future__ import annotations

import asyncio
import base64
import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bildwerk.bildformat import SEED_GRENZE, Bildauftrag
from bildwerk.bilddienst import erstellen, startwerte
from bildwerk.gewichte import Gewichtsstand, Pruefliste
from bildwerk.modelle import moss_image
from bildwerk.unterbau import Abgebrochen, Einstellungen, GewichteFehlen, Rechner, verbindung_beobachten


class ScheinMotor:
    """Wie der echte Motor, aber ohne mflux: Bilder sind einfarbige Flächen in der verlangten Größe."""

    angaben = moss_image.ANGABEN

    def __init__(self, gewichte_da: bool = True, laden_fehler: Exception | None = None) -> None:
        self.pruefliste = Pruefliste.lesen(moss_image.PRUEFLISTE)
        self.gewichte_da, self.laden_fehler = gewichte_da, laden_fehler
        self.geladen = 0
        self.auftraege: list[Bildauftrag] = []

    def gewichte(self) -> Gewichtsstand:
        return Gewichtsstand(ordner="/schein", groesse_gb=27.2, fehlen=[] if self.gewichte_da else ["vae/x.safetensors"])

    def laden(self) -> None:
        self.geladen += 1
        if self.laden_fehler:
            raise self.laden_fehler

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        self.auftraege.append(auftrag)
        return [Image.new("RGB", (auftrag.breite, auftrag.hoehe), (s % 256, 0, 0)) for s in auftrag.seeds]


def bereit(c: TestClient) -> dict:
    for _ in range(200):
        r = c.get("/gesundheit")
        if r.status_code == 200:
            return r.json()
        time.sleep(0.01)
    raise AssertionError(c.get("/gesundheit").json())


def bild(daten: dict) -> Image.Image:
    roh = base64.b64decode(daten["b64_json"])
    assert roh[:8] == b"\x89PNG\r\n\x1a\n"
    return Image.open(io.BytesIO(roh))


def test_erzeugen():
    m = ScheinMotor()
    with TestClient(erstellen("moss-image", m, Einstellungen())) as c:
        g = bereit(c)
        assert g["zustand"] == "bereit" and g["dienst"] == "moss-image" and g["gewichte_vollstaendig"] is True
        r = c.post("/v1/images/generations", json={"prompt": "Ein Leuchtturm", "size": "1344x768", "n": 2, "seed": 7, "steps": 20,
                                                   "quality": "hd", "response_format": "b64_json"})
        assert r.status_code == 200, r.text[:300]
        d = r.json()
        assert len(d["data"]) == 2 and [bild(x).size for x in d["data"]] == [(1344, 768)] * 2
        assert d["x-bildwerk"] | {"dauer_ms": 0} == {"modell": "moss-image", "seed": 7, "seeds": [7, 8], "schritte": 20,
                                                     "groesse": "1344x768", "dauer_ms": 0}
        assert m.auftraege[-1] == Bildauftrag(beschreibung="Ein Leuchtturm", breite=1344, hoehe=768, seeds=[7, 8], schritte=20)
        assert m.geladen == 1


def test_vorgaben_und_startwerte():
    m = ScheinMotor()
    with TestClient(erstellen("moss-image", m, Einstellungen())) as c:
        bereit(c)
        r = c.post("/v1/images/generations", json={"prompt": "Katze", "model": "MOSS-Image-SFT"})
        assert r.status_code == 200, r.text[:300]
        a = m.auftraege[-1]
        assert (a.breite, a.hoehe, a.schritte, len(a.seeds)) == (1024, 1024, 50, 1) and 0 <= a.seeds[0] < SEED_GRENZE
        assert r.json()["x-bildwerk"]["seed"] == a.seeds[0]
        for name in ("moss-image", "bildwerk-moss-image", "OpenMOSS-Team/MOSS-Image-SFT"):
            assert c.post("/v1/images/generations", json={"prompt": "Katze", "model": name}).status_code == 200
    assert startwerte(SEED_GRENZE - 1, 2) == [SEED_GRENZE - 1, 0]


@pytest.mark.parametrize("anfrage, code", [
    ({"size": "1000x1000"}, 400),                     # nicht durch 16 teilbar
    ({"size": "2064x1024"}, 400),                     # Kante über 2048
    ({"size": "256x256"}, 400),                       # Fläche unter 512x512
    ({"size": "groß"}, 400),
    ({"steps": 219}, 400),
    ({"steps": 0}, 400),
    ({"n": 5}, 422),
    ({"seed": SEED_GRENZE}, 422),
    ({"response_format": "url"}, 422),
    ({"prompt": "   "}, 422),
    ({"model": "z-image-turbo"}, 409),
])
def test_fehlerfaelle(anfrage: dict, code: int):
    m = ScheinMotor()
    with TestClient(erstellen("moss-image", m, Einstellungen())) as c:
        bereit(c)
        r = c.post("/v1/images/generations", json={"prompt": "Katze"} | anfrage)
        assert r.status_code == code, r.text[:300]
        assert not m.auftraege


def test_bearbeiten_501():
    with TestClient(erstellen("moss-image", ScheinMotor(), Einstellungen())) as c:
        bereit(c)
        r = c.post("/v1/images/edits", files={"image": ("a.png", b"\x89PNG", "image/png")}, data={"prompt": "heller"})
        assert r.status_code == 501 and "Bildvorlage" in r.json()["detail"]


def test_steckbrief():
    m = ScheinMotor()
    with TestClient(erstellen("moss-image", m, Einstellungen(quantisierung="4bit"))) as c:
        bereit(c)
        d = c.get("/v1/models").json()
    assert d["object"] == "list" and len(d["data"]) == 1
    e = d["data"][0]
    s = e["steckbrief"]
    assert e["id"] == "moss-image" and e["object"] == "model" and e["created"] > 1_790_000_000
    assert s["faehigkeiten"] == ["bild_erzeugen"] and s["kann_bearbeiten"] is False and s["lizenz"] == "Apache-2.0"
    assert s["quelle"] == {"repo": m.pruefliste.repo, "revision": m.pruefliste.revision}
    assert s["schritte"] == {"vorgabe": 50, "kleinste": 1, "groesste": 218} and s["groessen"]["vorgabe"] == "1024x1024"
    assert s["quantisierung"] == "4bit" and s["rechenwerk"]["name"] == "mflux" and s["zustand"] == "bereit"
    assert s["anzahl_max"] == 4 and s["beschreibung_max_token"] == 4096


def test_probemodus_laedt_und_rechnet_nie():
    m = ScheinMotor(gewichte_da=False)
    with TestClient(erstellen("moss-image", m, Einstellungen(), probe=True)) as c:
        g = c.get("/gesundheit")
        assert g.status_code == 200 and g.json()["zustand"] == "probe" and g.json()["gewichte_vollstaendig"] is False
        assert c.get("/v1/models").json()["data"][0]["steckbrief"]["zustand"] == "probe"
        r = c.post("/v1/images/generations", json={"prompt": "Katze"})
        assert r.status_code == 503 and "probe" in r.json()["detail"]
    assert m.geladen == 0 and not m.auftraege


def test_gewichte_fehlen():
    m = ScheinMotor(laden_fehler=GewichteFehlen("Gewichte unvollständig in /schein"))
    with TestClient(erstellen("moss-image", m, Einstellungen())) as c:
        for _ in range(200):
            g = c.get("/gesundheit")
            if g.json()["zustand"] != "laedt":
                break
            time.sleep(0.01)
        assert g.status_code == 503 and g.json()["zustand"] == "gestoert" and "unvollständig" in g.json()["meldung"]
        assert c.post("/v1/images/generations", json={"prompt": "Katze"}).status_code == 503


# ── Abbruch ──────────────────────────────────────────────────────────────
async def test_abbruch_vor_dem_start():
    r = Rechner("moss-image", "moss-image", Einstellungen())
    r.zustand = r.zustand.BEREIT
    abbruch = threading.Event()
    abbruch.set()
    gerechnet = []
    with pytest.raises(Abgebrochen):
        await r.rechnen(lambda *a: gerechnet.append(a), abbruch)
    assert not gerechnet


async def test_verbindung_zu_setzt_abbruch():
    class Anfrage:
        def __init__(self):
            self.fragen = 0

        async def is_disconnected(self):
            self.fragen += 1
            return self.fragen >= 2
    abbruch = threading.Event()
    await asyncio.wait_for(verbindung_beobachten(Anfrage(), abbruch), timeout=5)
    assert abbruch.is_set()


def test_motor_bricht_ab_und_raeumt_auf():
    """Der echte Motor von moss-image mit nachgestelltem mflux-Modell: Abbruch zwischen den Bildern und in den Schritten,
    danach ist der Zwischenspeicher der Textkodierung leer."""
    class Ergebnis:
        def __init__(self, groesse):
            self.image = Image.new("RGB", groesse)

    class Modell:
        def __init__(self, abbruch: threading.Event, wache):
            self.prompt_cache, self.aufrufe, self.abbruch, self.wache = {}, [], abbruch, wache

        def generate_image(self, **kw):
            self.aufrufe.append(kw)
            self.prompt_cache[kw["prompt"]] = "kodiert"
            if len(self.aufrufe) == 2:              # während des zweiten Bildes geht der Aufrufer
                self.abbruch.set()
                self.wache.call_in_loop(t=3, seed=kw["seed"], prompt=kw["prompt"], latents=None, config=None, time_steps=None)
            return Ergebnis((kw["width"], kw["height"]))

    motor = moss_image.Motor(Einstellungen())
    abbruch = threading.Event()
    motor.m = Modell(abbruch, motor._wache)
    auftrag = Bildauftrag(beschreibung="Katze", breite=512, hoehe=512, seeds=[1, 2, 3], schritte=8)
    with pytest.raises(Abgebrochen, match="Schritt 4"):
        motor.erzeugen(auftrag, abbruch)
    assert len(motor.m.aufrufe) == 2 and motor.m.prompt_cache == {} and motor._wache.signal is None
    assert motor.m.aufrufe[0] | {"seed": 0} == {"seed": 0, "prompt": "Katze", "num_inference_steps": 8, "width": 512, "height": 512,
                                                "guidance": 3.0}
    bilder = motor.erzeugen(auftrag.model_copy(update={"seeds": [5]}), threading.Event())
    assert [b.size for b in bilder] == [(512, 512)] and motor.m.prompt_cache == {}


def test_motor_laedt_nie_ohne_vollstaendige_gewichte(tmp_path, monkeypatch):
    monkeypatch.setattr(moss_image, "ablage", lambda: tmp_path)
    with pytest.raises(GewichteFehlen, match="bildwerk gewichte moss-image --laden"):
        moss_image.Motor(Einstellungen()).laden()
