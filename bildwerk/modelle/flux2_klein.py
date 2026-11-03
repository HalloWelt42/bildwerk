"""Dienst flux2-klein: FLUX.2 klein 9B von Black Forest Labs über mflux, ganz in MLX.

FLUX.2 klein 9B ist ein destillierter Diffusions-Transformer mit 9 Milliarden Parametern, Qwen3-8B als Textkodierer
und dem VAE von FLUX.2. Er rechnet in 4 Schritten mit Führung 1 und nimmt beim Bearbeiten bis zu 4 Bildvorlagen an, an
deren Stil und Inhalt er sich hält (mflux `Flux2KleinEdit`, Vorlagen als Bezugstoken neben dem Bild). Ohne Vorlage
erzeugt dieselbe Klasse aus Text. Gedacht für schnelle, einfache Aufgaben und Stilvorlagen.

Lizenz: FLUX Non-Commercial License; die Bilder dürfen nur nicht-kommerziell genutzt werden.
"""

from __future__ import annotations

import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from ..bildformat import ANZAHL_MAX, Bildauftrag, Groessenregel, Modellangaben, Schrittregel
from ..gewichte import Gewichtsstand, Pruefliste, ablage, pruefen
from ..unterbau import Abgebrochen, Einstellungen, GewichteFehlen, Quantisierung
from .moss_image import Abbruchwache

PORT = 8792
PRUEFLISTE = Path(__file__).with_name("flux2_klein.json")

ANGABEN = Modellangaben(
    kennung="flux2-klein",
    name="FLUX.2 klein 9B",
    aliase=["black-forest-labs/FLUX.2-klein-9B", "flux2-klein-9b"],
    zweck="Schnelle Bilder aus Text in 4 Schritten, dazu Bilder nach bis zu 4 Vorlagen (Stil, Bezug, Bearbeiten)",
    kann_bearbeiten=True,
    vorlagen_max=ANZAHL_MAX,
    eignung=["schnell"],
    sprachen=["en"],
    parameter_mrd=9.0,
    textkodierer="Qwen3-8B",
    lizenz="FLUX Non-Commercial License",
    lizenz_hinweis="Nur nicht-kommerzielle Nutzung der Bilder.",
    erschienen=date(2026, 1, 14),
    # Black Forest Labs: bis 4 Megapixel, Kanten im Raster von 16
    groessen=Groessenregel(raster=16, kante_min=256, kante_max=2048, pixel_min=512 * 512, pixel_max=2048 * 2048, vorgabe="1024x1024",
                           empfohlen=["1024x1024", "1344x768", "768x1344", "1152x896", "896x1152"]),
    # destilliert auf 4 Schritte; mehr Schritte sind möglich, bringen bei diesem Modell aber kaum mehr
    schritte=Schrittregel(vorgabe=4, kleinste=1, groesste=50),
    guidance=1.0,
    beschreibung_max_token=512,
)


def mflux_herkunft() -> tuple[str, str, list[str]]:
    """Repo, Revision und Dateimuster, wie der gepinnte Stand von mflux sie für FLUX.2 klein 9B verlangt."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.resolution.model_revisions import ModelRevisions
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    repo = ModelConfig.flux2_klein_9b().model_name
    return repo, ModelRevisions.get(repo), Flux2KleinWeightDefinition.get_download_patterns()


class Motor:
    """Hält Flux2KleinEdit aus mflux; geladen wird nur aus der Modellablage, nie aus dem Netz."""

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
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

        self.m = Flux2KleinEdit(quantize=self.quantisierung.bits, model_config=ModelConfig.flux2_klein_9b())
        self.m.callbacks.register(self._wache)

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        self._wache.signal = abbruch
        # mflux liest Vorlagen nur aus Dateien: für die Dauer des Auftrags in einem eigenen Ordner, danach weg
        with tempfile.TemporaryDirectory(prefix="bildwerk-vorlagen-") as ordner:
            pfade = []
            for i, inhalt in enumerate(auftrag.vorlagen):
                pfad = Path(ordner) / f"vorlage-{i}"
                pfad.write_bytes(inhalt)
                pfade.append(pfad)
            try:
                bilder = []
                for seed in auftrag.seeds:
                    if abbruch.is_set():
                        raise Abgebrochen(f"vor Bild {len(bilder) + 1}")
                    ergebnis = self.m.generate_image(seed=seed, prompt=auftrag.beschreibung, num_inference_steps=auftrag.schritte,
                                                     width=auftrag.breite, height=auftrag.hoehe, guidance=ANGABEN.guidance,
                                                     image_paths=pfade or None)
                    bilder.append(ergebnis.image)
                return bilder
            finally:
                self._wache.signal = None
                # mflux behält jede Textkodierung; im Dauerbetrieb wüchse das ohne Grenze
                self.m.prompt_cache.clear()
