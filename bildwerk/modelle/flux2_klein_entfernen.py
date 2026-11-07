"""Dienst bild-entfernen: FLUX.2 klein 4B mit dem LoRA erase-v1 über mflux, Entfernen in einigen Sekunden (ATL-43 des
Ateliers). Die Mittelstufe zwischen LaMa (bild-entfernen-schnell, unter einer Sekunde, verschmiert große Flächen) und
Qwen-Image-Edit (Minuten): das LoRA ersetzt die markierte Fläche durch den umgebenden Hintergrund, laut Modellkarte samt
Schatten und Spiegelung des Objekts.

Ansteuerung nach der Modellkarte von erase-v1: zwei Bezugsbilder, erst das Foto, dann die Maske als RGB (weiß entfernen,
schwarz behalten); mflux legt sie mit den Zeitkoordinaten 10 und 20 neben das Ziel, wie das LoRA es gelernt hat. Der Text
wird nicht neu kodiert: die Karte liefert ihn fertig kodiert (`prompt_embeds.safetensors`), der Dienst setzt ihn an die
Stelle der Textkodierung. 4 Schritte, keine Führung, höchstens 1 Megapixel; die Auswahl wächst um rund 2 % der größeren
Kante (sonst bleiben Reste), und nur innerhalb von ihr wird das Ergebnis eingesetzt.

Schnittstelle wie jeder Bilddienst: `/v1/images/edits` mit genau einer Vorlage, die Fläche deckend in der Signalfarbe
der Aufgabenvorlage (wie bei LaMa und Qwen). Die Anweisung braucht das Modell nicht.

Lizenz: Apache-2.0 (FLUX.2 klein 4B und erase-v1), die Bilder sind frei nutzbar.
"""

from __future__ import annotations

import io
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from ..bildformat import Aufgabe, Bildauftrag, Groessenregel, Modellangaben, Schrittregel, aufgaben_lesen
from ..entfernen import ausschnitt, maske_aus_signalfarbe, wachsen, zurueckmischen
from ..gewichte import Gewichtsstand, Pruefliste, ablage, pruefen
from ..unterbau import Abgebrochen, Einstellungen, GewichteFehlen, Quantisierung
from .moss_image import Abbruchwache

PORT = 8800
#: Name des Dienstes im Maschinenraum und für die Leitstelle: die Fähigkeit (HELF-189), Mittelstufe ohne Zusatz
DIENST = "bild-entfernen"
#: das Original (FLUX.2 klein 4B, wie der gepinnte Stand von mflux es verlangt) und, als Zusatz, LoRA und kodierter Text
PRUEFLISTE_QUELLE = Path(__file__).with_name("flux2_klein_entfernen.json")
ZUSATZ_PRUEFLISTEN = [Path(__file__).with_name("flux2_klein_entfernen_lora.json")]
#: die eigene Ablage (MR-4): 8 Bit mit eingerechnetem LoRA und dem kodierten Text daneben; nur daraus lädt der Dienst
PRUEFLISTE = Path(__file__).with_name("flux2_klein_entfernen_8bit.json")
LOKAL = "bild/mflux/flux2-klein-4b-erase-8bit"
BITS = 8
AUFGABEN = Path(__file__).with_name("flux2_klein_entfernen.aufgaben.toml")
LORA, TEXT = "erase_v1_diffusers.safetensors", "prompt_embeds.safetensors"

#: Arbeitsgröße: höchstens 1 Megapixel, Kanten im Raster von 16 (Modellkarte)
ARBEIT = 1024
#: um diesen Anteil der größeren Kante des Ausschnitts wächst die Auswahl (Modellkarte: rund 2 %)
WACHSEN = 0.02

ANGABEN = Modellangaben(
    kennung="flux2-klein-entfernen",
    name="FLUX.2 klein 4B erase",
    aliase=[DIENST, "slopshop/erase-v1", "flux2-klein-4b-erase"],
    zweck="Markierte Flächen entfernen und mit dem Hintergrund füllen, samt Schatten und Spiegelung, in einigen Sekunden",
    kann_bearbeiten=True,
    vorlagen_min=1,
    vorlagen_max=1,
    eignung=[],
    sprachen=[],
    parameter_mrd=4.0,
    textkodierer="keiner (Text fertig kodiert)",
    lizenz="Apache-2.0",
    lizenz_hinweis="Apache-2.0, frei nutzbar.",
    erschienen=date(2026, 1, 14),
    # Rechengröße frei (gerechnet wird ein Ausschnitt auf der Arbeitsgröße)
    groessen=Groessenregel(raster=8, kante_min=64, kante_max=4096, pixel_min=64 * 64, pixel_max=4096 * 4096, vorgabe="1024x1024"),
    # das LoRA ist auf 4 Schritte gelernt
    schritte=Schrittregel(vorgabe=4, kleinste=4, groesste=4),
    guidance=1.0,
    beschreibung_max_token=0,
    aufgaben=aufgaben_lesen(AUFGABEN),
)


def herkunft() -> tuple[str, str, list[str]]:
    """Repo, Revision und Dateimuster, wie der gepinnte Stand von mflux sie für FLUX.2 klein 4B verlangt."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.resolution.model_revisions import ModelRevisions
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    repo = ModelConfig.flux2_klein_4b().model_name
    return repo, ModelRevisions.get(repo), Flux2KleinWeightDefinition.get_download_patterns()


def bauen(bits: int = BITS) -> Any:
    """FLUX.2 klein 4B aus dem Original, quantisiert, mit eingerechnetem erase-v1 (für die eigene Ablage). Gespeichert
    wird über `Flux2Klein`: dieselben Teilmodule wie `Flux2KleinEdit`, das in mflux 0.24.0 kein `save_model` hat."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    lora = Pruefliste.lesen(ZUSATZ_PRUEFLISTEN[0])
    return Flux2Klein(quantize=bits, model_config=ModelConfig.flux2_klein_4b(), lora_paths=[str(lora.ordner(ablage()) / LORA)],
                          lora_scales=[1.0])


def aus_ablage(ordner: str) -> Any:
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit

    return Flux2KleinEdit(model_path=ordner, model_config=ModelConfig.flux2_klein_4b())


def umwandeln(original_loeschen: bool) -> int:
    """Die eigene 8-Bit-Ablage bauen (bildwerk umwandeln flux2-klein-entfernen); das Original bleibt (MR-4)."""
    from ..eigene_ablage import umwandeln as bauen_und_speichern

    if original_loeschen:
        print("Quellen und Duplikate räumen wir später gemeinsam ab (MR-4); das Original bleibt.")
    lora = Pruefliste.lesen(ZUSATZ_PRUEFLISTEN[0])
    return bauen_und_speichern(ANGABEN.kennung, Pruefliste.lesen(PRUEFLISTE_QUELLE), PRUEFLISTE, LOKAL, bauen, aus_ablage,
                               beilagen=[lora.ordner(ablage()) / TEXT])


def ausschnitt_fuer(maske: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    """Die nach der Modellkarte gewachsene Maske und der Ausschnitt um sie; None ohne Markierung."""
    kasten = ausschnitt(maske)
    if kasten is None:
        return None
    gewachsen = wachsen(maske, round(WACHSEN * max(kasten[2] - kasten[0], kasten[3] - kasten[1])))
    kasten = ausschnitt(gewachsen)
    return (gewachsen, kasten) if kasten else None


class Motor:
    """Hält Flux2KleinEdit aus mflux mit dem eingerechneten LoRA; geladen wird nur aus der eigenen 8-Bit-Ablage."""

    angaben = ANGABEN
    #: die Ablage ist schon quantisiert; die Einstellung des Dienstes gilt hier nicht
    quantisierung = Quantisierung.Q8

    def __init__(self, einstellungen: Einstellungen) -> None:
        self.pruefliste = Pruefliste.lesen(PRUEFLISTE) if PRUEFLISTE.is_file() else Pruefliste(
            repo="black-forest-labs/FLUX.2-klein-4B", revision="", lizenz=ANGABEN.lizenz, lokal=LOKAL, dateien=[])
        self.m: Any = None
        self.signalfarbe: tuple[int, int, int] = tuple(ANGABEN.aufgaben[Aufgabe.ENTFERNEN].signalfarbe or (255, 0, 255))  # type: ignore[assignment]
        self._wache = Abbruchwache()

    def gewichte(self) -> Gewichtsstand:
        stand = pruefen(self.pruefliste, ablage())
        if not self.pruefliste.dateien:
            stand.fehlen.append("eigene Ablage (noch nicht umgewandelt)")
        return stand

    def laden(self) -> None:
        stand = self.gewichte()
        if not stand.vollstaendig:
            raise GewichteFehlen(f"{stand.meldung}. Von Hand: bildwerk gewichte {ANGABEN.kennung} --laden, "
                                 f"dann bildwerk umwandeln {ANGABEN.kennung}")
        import mlx.core as mx

        ordner = Path(stand.ordner)
        self.m = aus_ablage(str(ordner))
        self.m.callbacks.register(self._wache)
        text = mx.load(str(ordner / TEXT))
        # der Text der Modellkarte, fertig kodiert: an die Stelle der Kodierung (keine Gegenanweisung bei Führung 1)
        self.m._encode_prompt_pair = lambda **_: (text["prompt_embeds"], text["text_ids"], None, None)

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        if len(auftrag.vorlagen) != 1:
            raise ValueError("Entfernen rechnet mit genau einer Vorlage")
        bild = Image.open(io.BytesIO(auftrag.vorlagen[0])).convert("RGB")
        if bild.size != (auftrag.breite, auftrag.hoehe):
            bild = bild.resize((auftrag.breite, auftrag.hoehe), Image.LANCZOS)
        gefunden = ausschnitt_fuer(maske_aus_signalfarbe(bild, self.signalfarbe))
        if gefunden is None:
            raise ValueError(f"Keine markierte Fläche: die zu entfernende Fläche muss deckend in der Signalfarbe {self.signalfarbe} stehen")
        maske, kasten = gefunden
        teil = bild.crop(kasten).resize((ARBEIT, ARBEIT), Image.LANCZOS)
        teilmaske = maske.crop(kasten).resize((ARBEIT, ARBEIT), Image.NEAREST)
        # die Signalfarbe nicht ins Modell geben: dort neutral grau
        teil.paste((128, 128, 128), (0, 0, ARBEIT, ARBEIT), teilmaske)
        self._wache.signal = abbruch
        # mflux liest Bezugsbilder nur aus Dateien: für die Dauer des Auftrags in einem eigenen Ordner, danach weg
        with tempfile.TemporaryDirectory(prefix="bildwerk-entfernen-") as ordner:
            foto, weiss = Path(ordner) / "foto.png", Path(ordner) / "maske.png"
            teil.save(foto)
            teilmaske.convert("RGB").save(weiss)
            try:
                bilder = []
                for seed in auftrag.seeds:
                    if abbruch.is_set():
                        raise Abgebrochen(f"vor Bild {len(bilder) + 1}")
                    ergebnis = self.m.generate_image(seed=seed, prompt="erase", num_inference_steps=auftrag.schritte,
                                                     width=ARBEIT, height=ARBEIT, guidance=ANGABEN.guidance, image_paths=[foto, weiss])
                    bilder.append(zurueckmischen(bild, ergebnis.image, maske, kasten))
                return bilder
            finally:
                self._wache.signal = None
