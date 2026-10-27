"""Dienst moss-image: MOSS-Image-SFT von OpenMOSS (Apache-2.0) über mflux, ganz in MLX.

MOSS-Image ist ein Diffusions-Transformer mit 4,9 Milliarden Parametern, Qwen3.5-9B als Textkodierer und dem VAE von
FLUX.2 mit kleinem Dekoder. Er nimmt Beschreibungen bis 4096 Token, Schlagwortfolgen und Anordnungen als JSON und
setzt englische und chinesische Schrift ins Bild. Vorgaben der Modellkarte: 50 Schritte, Führung 3, leere negative
Beschreibung, im negativen Zweig wird Schicht 22 übersprungen (mflux setzt das selbst).

Bearbeiten kann der Dienst nicht: mflux nimmt für MOSS-Image keine Bildvorlage an.
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from ..bildformat import Bildauftrag, Groessenregel, Modellangaben, Schrittregel
from ..gewichte import Gewichtsstand, Pruefliste, ablage, pruefen
from ..unterbau import Abgebrochen, Einstellungen, GewichteFehlen, Quantisierung

PORT = 8791
PRUEFLISTE = Path(__file__).with_name("moss_image.json")

ANGABEN = Modellangaben(
    kennung="moss-image",
    name="MOSS-Image-SFT",
    aliase=["OpenMOSS-Team/MOSS-Image-SFT"],
    zweck="Bilder aus Text: lange Beschreibungen bis 4096 Token, klare Schrift im Bild (Englisch und Chinesisch), "
          "Anordnung von Text und Gegenständen per JSON, Fotos wie Zeichnungen",
    kann_bearbeiten=False,
    bearbeiten_grund="mflux nimmt für MOSS-Image keine Bildvorlage an; der Dienst erzeugt nur aus Text.",
    eignung=["gruendlich"],
    sprachen=["en", "zh"],
    parameter_mrd=4.9,
    textkodierer="Qwen3.5-9B",
    lizenz="Apache-2.0",
    erschienen=date(2026, 10, 16),
    # Modellkarte: am besten 1, 1,5 oder 2 Megapixel, möglich 0,25 bis 4 (2K), jedes Seitenverhältnis, Kanten durch 16
    groessen=Groessenregel(raster=16, kante_min=256, kante_max=2048, pixel_min=512 * 512, pixel_max=2048 * 2048, vorgabe="1024x1024",
                           empfohlen=["1024x1024", "1248x1248", "1456x1456", "1344x768", "768x1344", "1152x896", "896x1152"]),
    # mflux: 1 bis 218 (darüber fallen zwei Schritte auf denselben Punkt des Trainingsplans)
    schritte=Schrittregel(vorgabe=50, kleinste=1, groesste=218),
    guidance=3.0,
    beschreibung_max_token=4096,
)


def mflux_herkunft() -> tuple[str, str, list[str]]:
    """Repo, Revision und Dateimuster, wie der gepinnte Stand von mflux sie für MOSS-Image-SFT verlangt."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.resolution.model_revisions import ModelRevisions
    from mflux.models.moss_image.weights.moss_image_weight_definition import MOSSImageWeightDefinition

    repo = ModelConfig.moss_image_sft().model_name
    return repo, ModelRevisions.get(repo), MOSSImageWeightDefinition.get_download_patterns()


class Abbruchwache:
    """Rückruf in jedem Schritt von mflux: ist das Signal des laufenden Auftrags gesetzt, endet die Rechnung."""

    def __init__(self) -> None:
        self.signal: threading.Event | None = None

    def call_in_loop(self, t: int, seed: int, prompt: str, latents: Any, config: Any, time_steps: Any) -> None:
        if self.signal is not None and self.signal.is_set():
            raise Abgebrochen(f"nach Schritt {t + 1}")


class Motor:
    """Hält MOSSImage aus mflux; geladen wird nur aus der Modellablage, nie aus dem Netz."""

    angaben = ANGABEN

    def __init__(self, einstellungen: Einstellungen) -> None:
        self.quantisierung: Quantisierung = einstellungen.quantisierung
        self.pruefliste = Pruefliste.lesen(PRUEFLISTE)
        self.m: Any = None
        self._wache = Abbruchwache()

    def gewichte(self) -> Gewichtsstand:
        return pruefen(self.pruefliste, ablage())

    def laden(self) -> None:
        stand = self.gewichte()
        if not stand.vollstaendig:
            raise GewichteFehlen(f"{stand.meldung}. Laden von Hand: bildwerk gewichte {ANGABEN.kennung} --laden")
        from mflux.models.common.config import ModelConfig
        from mflux.models.moss_image import MOSSImage

        self.m = MOSSImage(quantize=self.quantisierung.bits, model_config=ModelConfig.moss_image_sft())
        self.m.callbacks.register(self._wache)

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        self._wache.signal = abbruch
        try:
            bilder = []
            for seed in auftrag.seeds:
                if abbruch.is_set():
                    raise Abgebrochen(f"vor Bild {len(bilder) + 1}")
                ergebnis = self.m.generate_image(seed=seed, prompt=auftrag.beschreibung, num_inference_steps=auftrag.schritte,
                                                 width=auftrag.breite, height=auftrag.hoehe, guidance=ANGABEN.guidance)
                bilder.append(ergebnis.image)
            return bilder
        finally:
            self._wache.signal = None
            # mflux behält jede Textkodierung (bis 4096 Token je Beschreibung); im Dauerbetrieb wüchse das ohne Grenze
            self.m.prompt_cache.clear()
