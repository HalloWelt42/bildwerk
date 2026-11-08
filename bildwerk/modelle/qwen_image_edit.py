"""Dienst qwen-image-edit: Qwen-Image-Edit-2511 von Qwen über mflux, ganz in MLX.

Qwen-Image-Edit-2511 bearbeitet Bilder nach einer Anweisung in Worten und nimmt dabei 1 bis 3 Bildvorlagen an (Stil,
Person, Produkt, Szene); laut Modellkarte (Fassung 2509, auf der 2511 aufbaut) rechnet es mit 1 bis 3 Vorlagen am
besten. Ohne Vorlage rechnet es nicht (mflux `QwenImageEdit` verlangt mindestens ein Bild), deshalb ist es ein reines
Bearbeitungsmodell: `/v1/images/generations` lehnt ab, `/v1/images/edits` braucht mindestens eine Vorlage.

Bildteil: MMDiT mit 20 Milliarden Parametern, Textkodierer Qwen2.5-VL 7B (liest auch die Vorlagen). Der Dienst lädt
aus einer eigenen, schon auf 8 Bit gespeicherten Ablage (`bild/mflux/qwen-image-edit-2511-8bit` in der Modellablage,
`bildwerk umwandeln qwen-image-edit`); das Original aus Hugging Face (rund 58 GB) kommt erst in den Papierkorb, wenn eine
Messung über die Leitstelle belegt, dass die eigene Ablage lädt und rechnet (`docs/messungen/qwen-image-edit.json`). Den Textkodierer
lässt mflux auch dort in bfloat16 (rund 15,5 GB mit dem Bildteil des Kodierers).

Lizenz: Apache-2.0, die Bilder sind frei nutzbar.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image

from ..bildformat import Bildauftrag, Groessenregel, Modellangaben, Schrittregel, aufgaben_lesen
from ..eigene_ablage import liste_zu, ort
from ..eigene_ablage import lesen as ablage_lesen
from ..gewichte import Gewichtsstand, Pruefliste, ablage, pruefen
from ..unterbau import Abgebrochen, Einstellungen, GewichteFehlen, Quantisierung
from .moss_image import Abbruchwache

PORT = 8795
#: was aus Hugging Face geladen und umgewandelt wird
PRUEFLISTE_QUELLE = Path(__file__).with_name("qwen_image_edit.json")
#: die eigene Ablage je Bitbreite (MR-5); der Dienst lädt nur daraus, in der Bitbreite seiner Einstellung (Vorgabe 8 Bit)
BASIS, BITS = "qwen-image-edit-2511", 8
LOKAL = ort(BASIS, BITS)
PRUEFLISTE = liste_zu(Path(__file__), "qwen_image_edit", BITS)
#: wie das Modell Aufgaben (Entfernen, Füllen, Erweitern, frei) als Anweisung bekommt; Text, kein Code
AUFGABEN = Path(__file__).with_name("qwen_image_edit.aufgaben.toml")

ANGABEN = Modellangaben(
    kennung="qwen-image-edit",
    name="Qwen-Image-Edit-2511",
    aliase=["Qwen/Qwen-Image-Edit-2511", "qwen-image-edit-2511", "qwen-edit-2511"],
    zweck="Bilder nach Anweisung bearbeiten und nach 1 bis 3 Vorlagen neu zeichnen (Stil, Person, Produkt, Szene), frei nutzbar",
    kann_bearbeiten=True,
    vorlagen_min=1,
    # Modellkarte Qwen-Image-Edit-2509 (2511 baut darauf auf): "Optimal performance is currently achieved with 1 to 3 input images"
    vorlagen_max=3,
    eignung=["gruendlich"],
    sprachen=["en", "zh"],
    parameter_mrd=20.0,
    textkodierer="Qwen2.5-VL-7B",
    lizenz="Apache-2.0",
    lizenz_hinweis="Apache-2.0, frei nutzbar.",
    erschienen=date(2025, 12, 17),
    # mflux rundet jede Kante auf das Raster von 16 ab; Grenzen wie bei den anderen Bilddiensten. Empfohlen sind die
    # Seitenverhältnisse der Modellkarte von Qwen-Image, soweit sie im Raster liegen (1:1, 16:9, 3:2 und hochkant).
    groessen=Groessenregel(raster=16, kante_min=256, kante_max=2048, pixel_min=512 * 512, pixel_max=2048 * 2048, vorgabe="1024x1024",
                           empfohlen=["1024x1024", "1328x1328", "1664x928", "928x1664", "1584x1056", "1056x1584"]),
    # Modellkarte: 40 Schritte; mflux rechnet jede Zahl ab 1, die Obergrenze 100 schützt die Rechenzeit (eigene Festlegung)
    schritte=Schrittregel(vorgabe=40, kleinste=1, groesste=100),
    # mflux: ohne Angabe 2,5 beim Bearbeiten
    guidance=2.5,
    beschreibung_max_token=1024,
    aufgaben=aufgaben_lesen(AUFGABEN),
)


def herkunft() -> tuple[str, str, list[str]]:
    """Repo, Revision und Dateimuster (Gewichte und Tokenizer), wie der gepinnte Stand von mflux sie für Qwen-Image-Edit-2511
    verlangt."""
    from mflux.models.common.config import ModelConfig
    from mflux.models.common.resolution.model_revisions import ModelRevisions
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

    repo = ModelConfig.qwen_image_edit_2511().model_name
    # die Gewichte und dazu der Tokenizer, den mflux über eigene Muster lädt (ohne ihn bricht das Laden ab)
    muster = QwenWeightDefinition.get_download_patterns() + [m for t in QwenWeightDefinition.get_tokenizers() for m in t.download_patterns]
    return repo, ModelRevisions.get(repo), muster


MESSUNG = Path(__file__).resolve().parents[2] / "docs" / "messungen" / "qwen-image-edit.json"


def gerechnet() -> bool:
    """Belegt eine Messung über die Leitstelle, dass die eigene Ablage rechnet (mindestens ein Lauf mit Bild)?"""
    import json
    try:
        laeufe = json.loads(MESSUNG.read_text(encoding="utf-8")).get("laeufe", [])
    except (OSError, ValueError):
        return False
    return any(x.get("vorlage") and x.get("dauer_s", 0) > 0 for x in laeufe)


def original_in_papierkorb(wo: Path, repo: str, papierkorb: Path | None = None) -> Path:
    """Das Original in den Papierkorb legen, nicht endgültig löschen. Die großen Dateien liegen nicht im Ordner des Modells,
    sondern in der gemeinsamen Ablage `blobs/` daneben (der Ordner des Modells hält nur Verweise); verschoben werden nur
    Dateien, auf die kein anderes Modell verweist. Alles kommt in einen Ordner `<Modell> (Original)` im Papierkorb."""
    original = wo / f"models--{repo.replace('/', '--')}"
    eigene = {Path(os.path.realpath(v)) for v in original.rglob("*") if v.is_symlink()}
    andere = {Path(os.path.realpath(v)) for m in wo.glob("models--*") if m != original for v in m.rglob("*") if v.is_symlink()}
    papierkorb = papierkorb or Path.home() / ".Trash"
    ziel = papierkorb / f"{original.name} (Original)"
    zaehler = 2
    while ziel.exists():
        ziel = papierkorb / f"{original.name} (Original) {zaehler}"
        zaehler += 1
    (ziel / "blobs").mkdir(parents=True)
    for datei in sorted(eigene - andere):
        if datei.is_file() and datei.is_relative_to(wo):
            shutil.move(str(datei), str(ziel / "blobs" / datei.relative_to(wo).as_posix().replace("/", "_")))
    shutil.move(str(original), str(ziel / original.name))
    return ziel


def bauen(bits: int = BITS) -> Any:
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

    return QwenImageEdit(quantize=bits, model_config=ModelConfig.qwen_image_edit_2511())


def aus_ablage(ordner: str) -> Any:
    from mflux.models.common.config import ModelConfig
    from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

    return QwenImageEdit(model_path=ordner, model_config=ModelConfig.qwen_image_edit_2511())


def umwandeln(original_loeschen: bool, bits: int = BITS) -> int:
    """Die eigene Ablage einer Bitbreite bauen (bildwerk umwandeln qwen-image-edit [--bits 4]), Prüfliste schreiben,
    Probeladen. Mit `original_loeschen` kommt das Original nur in den Papierkorb, wenn eine Messung belegt, dass die eigene
    Ablage rechnet; sonst sagt der Befehl, was fehlt. Rechnet selbst kein Bild."""
    from ..eigene_ablage import umwandeln as bauen_und_speichern

    quelle = Pruefliste.lesen(PRUEFLISTE_QUELLE)
    erfolg = bauen_und_speichern(ANGABEN.kennung, quelle, liste_zu(Path(__file__), "qwen_image_edit", bits), ort(BASIS, bits),
                                 lambda: bauen(bits), aus_ablage)
    if erfolg or not original_loeschen:
        return erfolg
    if not gerechnet():
        print(f"Original bleibt: erst über die Leitstelle messen (tools/messen.py qwen-image-edit), dann erneut mit --original-loeschen")
        return 3
    print(f"Original im Papierkorb: {original_in_papierkorb(ablage(), quelle.repo)}")
    return 0


class Motor:
    """Hält QwenImageEdit aus mflux; geladen wird nur aus der eigenen Ablage, nie aus dem Netz."""

    angaben = ANGABEN

    def __init__(self, einstellungen: Einstellungen) -> None:
        # die Ablage ist schon quantisiert; die Einstellung des Dienstes wählt nur, aus welcher Ablage geladen wird
        bits = einstellungen.quantisierung.bits or BITS
        self.quantisierung: Quantisierung = Quantisierung(f"{bits}bit")
        self.pruefliste = ablage_lesen(liste_zu(Path(__file__), "qwen_image_edit", bits), "Qwen/Qwen-Image-Edit-2511",
                                       ANGABEN.lizenz, ort(BASIS, bits))
        self.m: Any = None
        self._wache = Abbruchwache()

    def gewichte(self) -> Gewichtsstand:
        stand = pruefen(self.pruefliste, ablage())
        if not self.pruefliste.dateien:
            stand.fehlen.append("eigene Ablage (noch nicht umgewandelt)")
        return stand

    def laden(self) -> None:
        stand = self.gewichte()
        if not stand.vollstaendig:
            raise GewichteFehlen(f"{stand.meldung}. Laden und umwandeln von Hand: bildwerk gewichte {ANGABEN.kennung} --laden, "
                                 f"dann bildwerk umwandeln {ANGABEN.kennung} --bits {self.quantisierung.bits}")
        from mflux.models.common.config import ModelConfig
        from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

        self.m = QwenImageEdit(model_path=stand.ordner, model_config=ModelConfig.qwen_image_edit_2511(), **self.zusatzgewichte())
        self.m.callbacks.register(self._wache)

    def zusatzgewichte(self) -> dict[str, Any]:
        """LoRA, die beim Laden eingerechnet werden (die schnelle Fassung); hier keine."""
        return {}

    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]:
        if not auftrag.vorlagen:
            raise ValueError("Qwen-Image-Edit rechnet nur mit mindestens einer Bildvorlage")
        self._wache.signal = abbruch
        # mflux liest Vorlagen nur aus Dateien: für die Dauer des Auftrags in einem eigenen Ordner, danach weg
        with tempfile.TemporaryDirectory(prefix="bildwerk-vorlagen-") as ordner:
            pfade = []
            for i, inhalt in enumerate(auftrag.vorlagen):
                pfad = Path(ordner) / f"vorlage-{i}"
                pfad.write_bytes(inhalt)
                pfade.append(str(pfad))
            try:
                bilder = []
                for seed in auftrag.seeds:
                    if abbruch.is_set():
                        raise Abgebrochen(f"vor Bild {len(bilder) + 1}")
                    ergebnis = self.m.generate_image(seed=seed, prompt=auftrag.beschreibung, image_paths=pfade,
                                                     num_inference_steps=auftrag.schritte, width=auftrag.breite,
                                                     height=auftrag.hoehe, guidance=self.angaben.guidance)
                    bilder.append(ergebnis.image)
                return bilder
            finally:
                self._wache.signal = None
