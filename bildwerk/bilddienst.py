"""Ein Bilddienst: dieselbe Schnittstelle für jedes Modell, das Modell selbst steckt im Motor.

    POST /v1/images/generations   Bilder aus Text (übliches Format, Antwort b64_json mit PNG)
    POST /v1/images/edits         Bilder bearbeiten; kann das Modell es nicht, 501 mit Grund
    GET  /v1/models               das eine Modell mit Steckbrief (Fähigkeiten, Größen, Schritte, Lizenz, Quelle)
    GET  /gesundheit              200 bereit (oder Probemodus), sonst 503 mit Zustand

Ein Motor hält genau ein Modell: er prüft dessen Gewichte, lädt es und rechnet einen Bildauftrag. Neue Modelle
brauchen nur einen neuen Motor mit Angaben und Prüfliste (Muster: `modelle/moss_image.py`).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import secrets
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from PIL import Image

from .bildformat import (SEED_GRENZE, Bildanfrage, Bildantwort, Bildauftrag, Bilddaten, Bildzusatz, Modelleintrag, Modellangaben,
                         Modellliste, Quelle, Rechenwerk, Steckbrief)
from .gewichte import Gewichtsstand, Pruefliste
from .unterbau import Einstellungen, Rechner, anwendung, verbindung_beobachten


class Motor(Protocol):
    angaben: Modellangaben
    pruefliste: Pruefliste

    def gewichte(self) -> Gewichtsstand: ...                       # schnelle Prüfung (Dateien und Größen), ohne Laden
    def laden(self) -> None: ...                                    # GewichteFehlen, wenn etwas fehlt; lädt nie aus dem Netz
    def erzeugen(self, auftrag: Bildauftrag, abbruch: threading.Event) -> list[Image.Image]: ...


def rechenwerk() -> Rechenwerk:
    """Fassung und Commit des installierten mflux (der Commit aus der Installation, gepinnt in pyproject.toml)."""
    try:
        fassung = metadata.version("mflux")
    except metadata.PackageNotFoundError:
        return Rechenwerk(fassung="fehlt")
    try:
        herkunft = json.loads(metadata.distribution("mflux").read_text("direct_url.json") or "{}")
    except (OSError, ValueError):
        herkunft = {}
    return Rechenwerk(fassung=fassung, commit=herkunft.get("vcs_info", {}).get("commit_id"))


def als_png(bild: Image.Image) -> bytes:
    puffer = io.BytesIO()
    bild.save(puffer, format="PNG")
    return puffer.getvalue()


def startwerte(seed: int | None, anzahl: int) -> list[int]:
    """Bild i rechnet mit seed + i (im Bereich der Startwerte); ohne seed wird einer gezogen."""
    erster = secrets.randbelow(SEED_GRENZE) if seed is None else seed
    return [(erster + i) % SEED_GRENZE for i in range(anzahl)]


def erstellen(dienst: str, motor: Motor, einstellungen: Einstellungen | None = None, probe: bool = False) -> FastAPI:
    e = einstellungen or Einstellungen()
    a = motor.angaben
    if a.kann_bearbeiten:
        # Bearbeiten bindet das erste Modell an, das es kann; bis dahin darf kein Steckbrief es versprechen
        raise NotImplementedError(f"{a.name}: Bearbeiten ist im Bildwerk noch nicht angebunden")
    rechner = Rechner(dienst, a.kennung, e, probe=probe)
    app = anwendung(dienst, rechner, motor.laden, lambda: motor.gewichte().vollstaendig, a.name)
    erschienen = int(datetime.combine(a.erschienen, datetime.min.time(), tzinfo=timezone.utc).timestamp())

    def auftrag_aus(anfrage: Bildanfrage) -> Bildauftrag:
        if anfrage.model and not a.meint(anfrage.model):
            raise HTTPException(409, f"Dieser Dienst hält {a.kennung}; für {anfrage.model} den Dienst dieses Modells nehmen.")
        try:
            breite, hoehe = a.groessen.kanten(anfrage.size or a.groessen.vorgabe)
        except ValueError as f:
            raise HTTPException(400, str(f)) from None
        schritte = a.schritte.vorgabe if anfrage.steps is None else anfrage.steps
        if not a.schritte.kleinste <= schritte <= a.schritte.groesste:
            raise HTTPException(400, f"Schritte {a.schritte.kleinste} bis {a.schritte.groesste}, nicht {schritte}.")
        return Bildauftrag(beschreibung=anfrage.prompt, breite=breite, hoehe=hoehe, seeds=startwerte(anfrage.seed, anfrage.n),
                           schritte=schritte)

    def rechnen(auftrag: Bildauftrag, abbruch: threading.Event) -> tuple[list[bytes], int]:
        beginn = time.monotonic()
        bilder = [als_png(b) for b in motor.erzeugen(auftrag, abbruch)]
        return bilder, int((time.monotonic() - beginn) * 1000)

    @app.post("/v1/images/generations", response_model=Bildantwort, response_model_by_alias=True)
    async def generations(anfrage: Bildanfrage, request: Request) -> Bildantwort:
        auftrag = auftrag_aus(anfrage)
        abbruch = threading.Event()
        waechter = asyncio.create_task(verbindung_beobachten(request, abbruch))
        try:
            bilder, dauer_ms = await rechner.rechnen(rechnen, abbruch, auftrag)
        finally:
            waechter.cancel()
        zusatz = Bildzusatz(modell=a.kennung, seed=auftrag.seeds[0], seeds=auftrag.seeds, schritte=auftrag.schritte,
                            groesse=f"{auftrag.breite}x{auftrag.hoehe}", dauer_ms=dauer_ms)
        return Bildantwort(created=int(time.time()), data=[Bilddaten(b64_json=base64.b64encode(b).decode("ascii")) for b in bilder],
                           x_bildwerk=zusatz)

    @app.post("/v1/images/edits")
    async def edits() -> None:
        raise HTTPException(501, f"{a.name} kann keine Bilder bearbeiten: {a.bearbeiten_grund}")

    @app.get("/v1/models", response_model=Modellliste)
    async def modelle() -> Modellliste:
        steckbrief = Steckbrief(**a.model_dump(), quelle=Quelle(repo=motor.pruefliste.repo, revision=motor.pruefliste.revision),
                                quantisierung=e.quantisierung.value, rechenwerk=rechenwerk(), zustand=rechner.zustand.value)
        return Modellliste(data=[Modelleintrag(id=a.kennung, created=erschienen, steckbrief=steckbrief)])

    return app
