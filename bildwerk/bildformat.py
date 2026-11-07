"""Die gemeinsame Schnittstelle aller Bilddienste im üblichen Bildformat: Anfrage, Antwort, Steckbrief.

Jeder Dienst des Bildwerks spricht genau diese Formen, gleich welches Modell er hält. So braucht die Leitstelle einen
einzigen Bildweg für alle Bildmodelle; was ein Modell kann (Größen, Schritte, Bearbeiten), sagt sein Steckbrief.
"""

from __future__ import annotations

import re
import tomllib
from datetime import date
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Größter Startwert: in JSON als ganze Zahl verlustfrei auch für Leser, die Zahlen als Gleitkomma lesen (wie die Leitstelle)
SEED_GRENZE = 2**53
ANZAHL_MAX = 4
VORLAGE_BYTES_MAX = 20 * 1024 * 1024          # je Bildvorlage beim Bearbeiten
GROESSE = re.compile(r"(\d{1,5})x(\d{1,5})")


class Groessenregel(BaseModel):
    """Welche Bildgrößen ein Modell annimmt: Kanten im Raster, je Kante und für die Fläche eine Grenze."""

    raster: int = Field(gt=0, description="jede Kante ist ein Vielfaches davon")
    kante_min: int
    kante_max: int
    pixel_min: int
    pixel_max: int
    vorgabe: str
    empfohlen: list[str] = Field(default_factory=list, description="Größen, in denen das Modell am besten zeichnet")

    def kanten(self, groesse: str) -> tuple[int, int]:
        """Breite und Höhe aus "BxH"; ValueError mit dem Grund, wenn das Modell die Größe nicht annimmt."""
        t = GROESSE.fullmatch(groesse.strip())
        if not t:
            raise ValueError(f"Größe als BreitexHöhe angeben, zum Beispiel {self.vorgabe}, nicht {groesse!r}.")
        breite, hoehe = int(t.group(1)), int(t.group(2))
        if breite % self.raster or hoehe % self.raster:
            raise ValueError(f"Breite und Höhe müssen durch {self.raster} teilbar sein, nicht {breite}x{hoehe}.")
        if not all(self.kante_min <= k <= self.kante_max for k in (breite, hoehe)):
            raise ValueError(f"Jede Kante {self.kante_min} bis {self.kante_max}, nicht {breite}x{hoehe}.")
        if not self.pixel_min <= breite * hoehe <= self.pixel_max:
            raise ValueError(f"Die Fläche muss zwischen {self.pixel_min} und {self.pixel_max} Pixeln liegen, "
                             f"{breite}x{hoehe} hat {breite * hoehe}.")
        return breite, hoehe


class Schrittregel(BaseModel):
    vorgabe: int
    kleinste: int
    groesste: int


class Aufgabe(str, Enum):
    """Was eine Anwendung mit einem Bild vorhat; das Modell sagt je Aufgabe, wie es die Anweisung dafür versteht."""

    ANWEISEN = "anweisen"        # frei nach dem Wunsch des Nutzers
    ENTFERNEN = "entfernen"      # die markierte Fläche verschwindet, die Szene geht dahinter weiter
    FUELLEN = "fuellen"          # die markierte Fläche füllen, mit dem Wunsch oder passend zur Umgebung
    ERWEITERN = "erweitern"      # der markierte Rand um das Bild wird ergänzt
    EINPASSEN = "einpassen"      # eingefügte Objekte an Licht, Farbe, Schatten und Kanten der Szene angleichen


WUNSCH = "{wunsch}"


class Beispiel(BaseModel):
    """Was das Modell an einem festen Motiv aus dem Portfolio macht (ATL-45 des Ateliers): Vorher (was es sah, markiert
    wie in der Bildbearbeitung) und Nachher als kleine Bilder (data-URL), dazu die Dauer im Dienst beim Rechnen."""

    model_config = ConfigDict(extra="forbid")

    motiv: str = Field(default="", description="kurzer Titel des Motivs, etwa: Bank am See")
    vorher: str = Field(pattern=r"^data:image/", description="was das Modell sah, als data-URL")
    nachher: str = Field(pattern=r"^data:image/", description="sein Ergebnis, als data-URL")
    dauer_s: float | None = Field(default=None, description="Dauer im Dienst beim Rechnen des Portfolios")


class Aufgabenvorlage(BaseModel):
    """Wie das Modell eine Aufgabe als Anweisung bekommt (ATL-26 des Ateliers): fester Text in der Sprache, die das Modell
    befolgt, mit dem Platzhalter `{wunsch}` für die Worte des Nutzers (übersetzt in `sprache`). Die Fläche, die die
    Aufgabe betrifft, steht in der Vorlage deckend in `signalfarbe`."""

    model_config = ConfigDict(extra="forbid")

    anweisung: str = Field(min_length=1, description="mit {wunsch}, wenn die Aufgabe einen Wunsch aufnimmt")
    ohne_wunsch: str | None = Field(default=None, description="gilt bei leerem Wunsch; fehlt sie, braucht die Aufgabe einen Wunsch "
                                                               "(steht kein {wunsch} in der Anweisung, nimmt sie keinen)")
    sprache: str = Field(pattern=r"^[a-z]{2}$", description="ISO 639-1; in diese Sprache wird der Wunsch übersetzt")
    signalfarbe: tuple[int, int, int] | None = Field(default=None, description="RGB der markierten Fläche; ohne: die Aufgabe markiert nichts")
    beispiel: Beispiel | None = Field(default=None, description="Vorher und Nachher an einem Motiv des Portfolios, zum Vergleichen")

    @field_validator("signalfarbe")
    @classmethod
    def _farbe(cls, wert: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if wert is not None and not all(0 <= k <= 255 for k in wert):
            raise ValueError("Signalfarbe als drei Werte 0 bis 255")
        return wert

    @field_validator("ohne_wunsch")
    @classmethod
    def _ohne(cls, wert: str | None) -> str | None:
        if wert is not None and WUNSCH in wert:
            raise ValueError("Die Anweisung ohne Wunsch darf keinen Platzhalter {wunsch} enthalten")
        return wert


def aufgaben_lesen(pfad: Path) -> dict[Aufgabe, Aufgabenvorlage]:
    """Die Aufgabenvorlagen eines Modells aus seiner TOML-Datei (je Aufgabe eine Tabelle); fehlt die Datei: keine."""
    if not pfad.is_file():
        return {}
    with pfad.open("rb") as datei:
        roh = tomllib.load(datei)
    return {Aufgabe(name): Aufgabenvorlage.model_validate(werte) for name, werte in roh.items()}


class Modellangaben(BaseModel):
    """Was ein Bildmodell ist und kann; fest je Modell, von Hand nach Modellkarte und Rechenwerk."""

    kennung: str = Field(description="Kurzname des Modells; als `model` in Anfragen")
    name: str
    aliase: list[str] = Field(default_factory=list, description="weitere Namen, die als `model` dasselbe meinen")
    zweck: str
    faehigkeiten: list[str] = Field(default_factory=lambda: ["bild_erzeugen"])
    kann_bearbeiten: bool = False
    bearbeiten_grund: str = Field(default="", description="warum das Modell nicht bearbeitet (Antwort 501)")
    vorlagen_max: int = Field(default=0, ge=0, description="Bildvorlagen je Bearbeitung (Stil, Bezug); 0: keine")
    vorlagen_min: int = Field(default=0, ge=0, description="Bildvorlagen, die das Modell mindestens braucht; über 0: rechnet nur mit Vorlage")
    eignung: list[str] = Field(default_factory=list, description="Wörter der Leitstelle: schnell, gruendlich, ...")
    sprachen: list[str] = Field(default_factory=list, description="Sprachen der Beschreibung, die das Modell gut versteht (ISO 639-1)")
    parameter_mrd: float = Field(description="Parameter des Bildteils (Transformer) in Milliarden")
    textkodierer: str
    lizenz: str
    lizenz_hinweis: str = Field(default="", description="was die Lizenz für die Nutzung der Bilder heißt, in einem Satz")
    erschienen: date
    groessen: Groessenregel
    schritte: Schrittregel
    guidance: float = Field(description="Stärke der Führung durch die Beschreibung (CFG), wie die Modellkarte sie vorgibt")
    beschreibung_max_token: int
    anzahl_max: int = Field(default=ANZAHL_MAX, ge=1, le=ANZAHL_MAX, description="Bilder je Auftrag höchstens (ein Modell, das immer gleich rechnet: 1)")
    aufgaben: dict[Aufgabe, Aufgabenvorlage] = Field(default_factory=dict, description="Anweisung je Aufgabe (nur Modelle, die bearbeiten)")

    def meint(self, model: str) -> bool:
        return model in (self.kennung, self.name, *self.aliase)


class Quelle(BaseModel):
    repo: str
    revision: str


class Rechenwerk(BaseModel):
    name: str = "mflux"
    fassung: str
    commit: str | None = None


class Steckbrief(Modellangaben):
    """Die Angaben des Modells und wie dieser Dienst es hält."""

    quelle: Quelle
    quantisierung: str
    rechenwerk: Rechenwerk
    zustand: str


class Modelleintrag(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "bildwerk"
    steckbrief: Steckbrief


class Modellliste(BaseModel):
    object: str = "list"
    data: list[Modelleintrag]


class Bildanfrage(BaseModel):
    """`POST /v1/images/generations`. Unbekannte Felder des üblichen Formats (quality, style, user) werden überlesen."""

    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1, max_length=20000)
    size: str | None = Field(default=None, description="BreitexHöhe; fehlt sie, gilt die Vorgabe des Modells")
    n: int = Field(default=1, ge=1, le=ANZAHL_MAX)
    seed: int | None = Field(default=None, ge=0, lt=SEED_GRENZE, description="Bild i rechnet mit seed + i; fehlt er, wird einer gezogen")
    steps: int | None = Field(default=None, description="Schritte; fehlen sie, gilt die Vorgabe des Modells")
    model: str | None = None
    response_format: str = "b64_json"

    @field_validator("prompt")
    @classmethod
    def _nicht_leer(cls, wert: str) -> str:
        if not wert.strip():
            raise ValueError("Die Beschreibung ist leer.")
        return wert

    @field_validator("response_format")
    @classmethod
    def _format(cls, wert: str) -> str:
        if wert != "b64_json":
            raise ValueError("Bilder kommen nur als b64_json (PNG), nie als Adresse.")
        return wert


class Bearbeitungsanfrage(Bildanfrage):
    """`POST /v1/images/edits`: wie die Erzeugung, dazu die Bildvorlagen als Base64 (PNG, JPEG oder WebP), in der
    Reihenfolge ihres Gewichts; das Modell hält sich an Stil und Inhalt der Vorlagen."""

    images: list[str] = Field(min_length=1, max_length=ANZAHL_MAX)

    @field_validator("images")
    @classmethod
    def _vorlagen(cls, werte: list[str]) -> list[str]:
        for i, wert in enumerate(werte, start=1):
            if len(wert) * 3 // 4 > VORLAGE_BYTES_MAX:
                raise ValueError(f"Vorlage {i} ist größer als {VORLAGE_BYTES_MAX // (1024 * 1024)} MB.")
        return werte


class Bilddaten(BaseModel):
    b64_json: str
    revised_prompt: str | None = None


class Bildzusatz(BaseModel):
    modell: str
    seed: int                         # Startwert des ersten Bildes
    seeds: list[int]                  # je Bild
    schritte: int
    groesse: str
    dauer_ms: int                     # Rechenzeit im Dienst, ohne Warten auf die Sperre


class Bildantwort(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    created: int
    data: list[Bilddaten]
    x_bildwerk: Bildzusatz = Field(alias="x-bildwerk")


class Bildauftrag(BaseModel):
    """Was der Motor eines Modells rechnet: geprüft, mit festen Startwerten je Bild."""

    beschreibung: str
    breite: int
    hoehe: int
    seeds: list[int]
    schritte: int
    vorlagen: list[bytes] = Field(default_factory=list, description="Bildvorlagen (Dateiinhalt), leer beim Erzeugen aus Text")
