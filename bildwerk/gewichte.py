"""Gewichte eines Modells: Prüfliste, Prüfung und Laden in die Modellablage.

Die Prüfliste (eine JSON-Datei je Modell neben seinem Modul) nennt Repo, Revision und jede Datei, die mflux für das
Modell braucht, mit Größe und Prüfsumme. Repo, Revision und Dateimuster stammen aus mflux (eine Wahrheit: der gepinnte
Stand von mflux); die Liste hält fest, was dieser Stand verlangt, damit die Prüfung ohne Netz gelingt.

Die Gewichte liegen im Zwischenspeicher von Hugging Face (`HF_HUB_CACHE`), den der Maschinenraum jedem Dienst auf die
Modellablage setzt (`~/Maschinenraum/modelle/huggingface`). Ein Dienst lädt nie selbst aus dem Netz: fehlt etwas,
meldet er es und bleibt gestört. Geladen wird nur von Hand mit `bildwerk gewichte <modell> --laden`.

Heruntergeladene Dateien sind fremde Daten: geladen werden nur Gewichte, Konfiguration und Tokenizer, nie Code des
Modell-Repos (dort liegende Python-Dateien passen auf kein Muster).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

BLOCK = 16 * 1024 * 1024


class Datei(BaseModel):
    pfad: str
    groesse: int
    sha256: str | None = Field(default=None, description="große Dateien (LFS): Prüfsumme des Inhalts")
    git_sha1: str | None = Field(default=None, description="kleine Dateien: Kennung des Inhalts im Git des Modell-Repos")


class Pruefliste(BaseModel):
    repo: str
    revision: str
    lizenz: str
    dateien: list[Datei]

    @classmethod
    def lesen(cls, datei: Path) -> "Pruefliste":
        return cls.model_validate_json(datei.read_text(encoding="utf-8"))

    def schreiben(self, datei: Path) -> None:
        datei.write_text(self.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")

    @property
    def groesse(self) -> int:
        return sum(d.groesse for d in self.dateien)

    def ordner(self, ablage: Path) -> Path:
        """Schnappschuss dieser Revision im Zwischenspeicher von Hugging Face."""
        return ablage / f"models--{self.repo.replace('/', '--')}" / "snapshots" / self.revision


class Gewichtsstand(BaseModel):
    ordner: str
    groesse_gb: float
    fehlen: list[str] = Field(default_factory=list, description="fehlt oder hat die falsche Größe")
    falsch: list[str] = Field(default_factory=list, description="Prüfsumme stimmt nicht (nur bei der Prüfung mit Prüfsummen)")

    @property
    def vollstaendig(self) -> bool:
        return not self.fehlen and not self.falsch

    @property
    def meldung(self) -> str:
        if self.vollstaendig:
            return f"vollständig ({self.groesse_gb:.1f} GB) in {self.ordner}"
        teile = []
        if self.fehlen:
            teile.append(f"{len(self.fehlen)} Dateien fehlen oder sind unvollständig ({', '.join(self.fehlen[:3])})")
        if self.falsch:
            teile.append(f"{len(self.falsch)} Dateien mit falscher Prüfsumme ({', '.join(self.falsch[:3])})")
        return f"Gewichte unvollständig in {self.ordner}: {'; '.join(teile)}"


def ablage() -> Path:
    """Der Zwischenspeicher von Hugging Face, wie ihn die Umgebung setzt (im Maschinenraum die Modellablage); derselbe
    Ort, an dem mflux sucht."""
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE).expanduser()


def _sha256(datei: Path) -> str:
    h = hashlib.sha256()
    with open(datei, "rb") as f:
        while stueck := f.read(BLOCK):
            h.update(stueck)
    return h.hexdigest()


def _git_sha1(datei: Path) -> str:
    inhalt = datei.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(inhalt) + inhalt).hexdigest()


def pruefen(liste: Pruefliste, wo: Path, pruefsummen: bool = False) -> Gewichtsstand:
    """Jede Datei der Liste da und gleich groß; mit `pruefsummen` auch gleich im Inhalt (liest alles, dauert)."""
    ordner = liste.ordner(wo)
    stand = Gewichtsstand(ordner=str(ordner), groesse_gb=round(liste.groesse / 1e9, 2))
    for d in liste.dateien:
        pfad = ordner / d.pfad
        if not pfad.is_file() or pfad.stat().st_size != d.groesse:
            stand.fehlen.append(d.pfad)
            continue
        if pruefsummen:
            ist = _sha256(pfad) if d.sha256 else _git_sha1(pfad)
            if ist != (d.sha256 or d.git_sha1):
                stand.falsch.append(d.pfad)
    return stand


def laden(liste: Pruefliste, wo: Path) -> Path:
    """Genau die Dateien der Liste in dieser Revision in die Ablage laden; Vorhandenes bleibt."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=liste.repo, revision=liste.revision, allow_patterns=[d.pfad for d in liste.dateien],
                                  cache_dir=wo, max_workers=4))


def erstellen(repo: str, revision: str, muster: list[str], lizenz: str) -> Pruefliste:
    """Prüfliste aus Hugging Face: alle Dateien der Revision, die auf die Muster von mflux passen (braucht Netz)."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.utils import filter_repo_objects
    eintraege = [e for e in HfApi().list_repo_tree(repo, revision=revision, recursive=True, expand=True) if isinstance(e, RepoFile)]
    gewaehlt = set(filter_repo_objects([e.path for e in eintraege], allow_patterns=muster))
    dateien = [Datei(pfad=e.path, groesse=e.size, sha256=e.lfs.sha256 if e.lfs else None, git_sha1=None if e.lfs else e.blob_id)
               for e in sorted(eintraege, key=lambda e: e.path) if e.path in gewaehlt]
    return Pruefliste(repo=repo, revision=revision, lizenz=lizenz, dateien=dateien)
