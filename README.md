# Bildwerk

Bilddienste, je Modell ein eigener Prozess. Jeder Dienst lädt beim Start genau sein Bildmodell; laufen heißt geladen,
anhalten heißt der Speicher ist wieder frei. Alle Dienste sprechen dieselbe Schnittstelle im üblichen Bildformat, so
braucht die Leitstelle einen einzigen Bildweg, gleich welches Modell rechnet.

## Zusammenspiel

Das Bildwerk gehört zu [Leitstelle](https://github.com/HalloWelt42/leitstelle),
[Maschinenraum](https://github.com/HalloWelt42/maschinenraum) und [Stellwerk](https://github.com/HalloWelt42/stellwerk).
Ein Bilddienst ist eine Rechenressource wie jedes andere Modell: er startet nie von selbst. Der Maschinenraum des
Rechners startet, hält an und misst ihn, aber nur, wenn die Leitstelle es verlangt; sie entscheidet über Laden,
Entladen und Vorrang. Die Dienste lauschen nur auf ihrem Rechner (`127.0.0.1`); Anwendungen, auch der Assistent,
erreichen sie nur über die Leitstelle.

```
Anwendungen  ->  Leitstelle  ->  Maschinenraum  ->  Bildwerk moss-image
```

## Die Dienste

| Dienst | Port | Fähigkeit | Modell | Bearbeiten | Lizenz des Modells |
|---|---|---|---|---|---|
| `moss-image` | 8791 | bild_erzeugen | MOSS-Image-SFT (4,9 Mrd. Parameter, Textkodierer Qwen3.5-9B), Vorgabe 8 Bit | nein | Apache-2.0 |

Gerechnet wird mit [mflux](https://github.com/mflux-community/mflux) ganz in MLX auf der Grafik des Macs. Innerhalb
eines Dienstes rechnet immer nur ein Auftrag zugleich; der Zwischenspeicher für Grafikpuffer ist begrenzt
(`BILDWERK_MLX_ZWISCHENSPEICHER_GB`, Vorgabe 2) und wird nach jeder Rechnung geleert. Feste Zeitgrenzen gibt es nicht;
schließt der Aufrufer die Verbindung (die Leitstelle bricht einen Auftrag ab), endet die Rechnung beim nächsten Schritt.

## Schnittstelle

Jeder Dienst, gleich welches Modell:

| Aufruf | Bedeutung |
|---|---|
| `POST /v1/images/generations` | Bilder aus Text; Antwort `data[].b64_json` (PNG) |
| `POST /v1/images/edits` | Bilder bearbeiten; kann das Modell es nicht, `501` mit Grund |
| `GET /v1/models` | das eine Modell des Dienstes mit Steckbrief |
| `GET /gesundheit` | `200` bereit (oder Probemodus), sonst `503` mit `zustand` laedt oder gestoert |

Felder von `POST /v1/images/generations` (unbekannte Felder werden überlesen):

| Feld | Pflicht | Bedeutung |
|---|---|---|
| `prompt` | ja | Beschreibung |
| `size` | nein | `BreitexHöhe`; fehlt sie, gilt die Vorgabe des Modells (Grenzen im Steckbrief) |
| `n` | nein | Anzahl 1 bis 4, Vorgabe 1 |
| `seed` | nein | Startwert 0 bis 2^53 - 1; Bild i rechnet mit `seed + i`; fehlt er, wird einer gezogen |
| `steps` | nein | Schritte; fehlen sie, gilt die Vorgabe des Modells |
| `model` | nein | wenn gesetzt, muss es das Modell des Dienstes nennen (`moss-image`, `MOSS-Image-SFT` oder die Kennung des Dienstes im Maschinenraum, `bildwerk-moss-image`), sonst `409` |
| `response_format` | nein | nur `b64_json` |

```json
{"created": 1793125734,
 "data": [{"b64_json": "iVBORw0KGgo...", "revised_prompt": null}],
 "x-bildwerk": {"modell": "moss-image", "seed": 42, "seeds": [42], "schritte": 50, "groesse": "1024x1024", "dauer_ms": 1234}}
```

Gleicher Startwert, gleiche Beschreibung, Größe und Schritte ergeben beim selben Dienst dasselbe Bild. `dauer_ms` ist
die Rechenzeit im Dienst ohne das Warten auf einen freien Platz. Fehler: `400` (Größe oder Schritte passen nicht zum
Modell), `409` (anderes Modell verlangt), `422` (Form der Anfrage), `501` (Bearbeiten), `503` (Modell nicht bereit).

Der Steckbrief unter `GET /v1/models` (`data[0].steckbrief`) nennt Zweck, Fähigkeiten, `kann_bearbeiten`, Eignung,
Sprachen der Beschreibung, Parameter, Textkodierer, Lizenz, Größenregel (Raster, Kanten, Fläche, Vorgabe, empfohlene
Größen), Schrittregel (Vorgabe, kleinste, größte), Führung, längste Beschreibung in Token, Quelle (Repo und Revision
der Gewichte), Quantisierung, Rechenwerk (mflux mit Fassung und Commit) und Zustand.

MOSS-Image: Größen mit Kanten durch 16, je 256 bis 2048, Fläche 512x512 bis 2048x2048, am besten 1 bis 2 Megapixel;
Schritte 1 bis 218, Vorgabe 50; Führung 3; Beschreibungen bis 4096 Token, auch als Schlagwortfolge oder als Anordnung in
JSON; Schrift im Bild auf Englisch und Chinesisch.

## Gewichte

Die Gewichte liegen im Zwischenspeicher von Hugging Face, den der Maschinenraum jedem Dienst auf die Modellablage setzt
(`HF_HUB_CACHE`, `~/Maschinenraum/modelle/huggingface`). Ein Dienst lädt nie selbst aus dem Netz: fehlt etwas, meldet
er `gestoert` mit dem Grund. Je Modell hält eine Prüfliste (`bildwerk/modelle/<modell>.json`) Repo, Revision und jede
nötige Datei mit Größe und Prüfsumme fest; Repo, Revision und Dateimuster stammen aus dem gepinnten Stand von mflux.
Code aus dem Modell-Repo wird nie geladen.

```bash
HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte moss-image --laden
HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte moss-image --pruefsummen
```

Der erste Aufruf lädt fehlende Dateien und prüft danach alles mit Prüfsummen, der zweite prüft nur. MOSS-Image-SFT:
16 Dateien, 27,2 GB.

## Installieren

Python 3.11 oder neuer, ein Mac, auf dem MLX läuft.

```bash
./installieren.sh
```

Das Skript legt `.venv` mit den erprobten Paketständen aus `einschraenkungen.txt` und mflux auf dem Commit aus
`pyproject.toml` an und trägt, wenn es einen Maschinenraum gibt, je Modell die Dienstdatei aus `deploy/` unter
`~/Maschinenraum/dienste/` ein (vorhandene bleiben). Gestartet und geladen wird nichts.

## Einstellungen

Je Dienst über Umgebungsvariablen (in der Dienstdatei des Maschinenraums unter `[umgebung]`):

| Variable | Vorgabe | Bedeutung |
|---|---|---|
| `BILDWERK_QUANTISIERUNG` | 8bit | `bf16`, `8bit`, `6bit`, `5bit`, `4bit` oder `3bit` (Transformer und Textkodierer, beim Laden) |
| `BILDWERK_ADRESSE` | 127.0.0.1 | nur dieser Rechner |
| `BILDWERK_MLX_ZWISCHENSPEICHER_GB` | 2 | Grenze für behaltene Grafikpuffer |
| `HF_HUB_CACHE` | vom Maschinenraum | Ort der Gewichte |
| `HF_HUB_OFFLINE` | 1 (Dienstdatei) | nie aus dem Netz laden |

## Stand von mflux

mflux ist auf genau einen Commit gepinnt, an einer Stelle: `pyproject.toml`, Zusatz `mflux`. Ein neuer Stand: Commit
dort ändern, `./installieren.sh`, `bildwerk pruefliste <modell>` (schreibt die Prüfliste aus der Revision, die mflux
verlangt; braucht Netz), dann die Tests. Sie schlagen fehl, solange installiertes mflux, Pin und Prüflisten nicht
zusammenpassen.

## Entwickeln

```bash
.venv/bin/python -m pytest
```

Die Tests laden kein Modell und rechnen kein Bild: der Dienst läuft mit nachgestelltem Motor, der echte Motor mit
nachgestelltem mflux-Modell. Dazu prüfen sie Prüflisten und Steckbrief gegen den gepinnten Stand von mflux.

Probemodus ohne Modell, etwa für einen freien Port:

```bash
.venv/bin/bildwerk moss-image --probe --port 18791
```

Er beantwortet `/gesundheit` (`zustand` probe, mit `gewichte_vollstaendig`) und `/v1/models`, lehnt Rechnen mit `503` ab.

---

## Lizenz

**Nicht-kommerzielle Nutzung** - Siehe [LICENSE](LICENSE)

Erlaubt: Private Nutzung, Installation, persönliche Anpassungen, Teilen mit Quellenangabe

Verboten: Kommerzielle Nutzung, Verkauf, Einbindung in kommerzielle Produkte

---

## Unterstützen

Bildwerk ist ein privates Hobby-Projekt. Kein Tracking, keine Werbung, keine Kompromisse.

Wenn dir das Projekt gefällt, kannst du ihm einen Stern geben - oder direkt hier:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/HalloWelt42)

**Crypto:**

| Coin | Adresse |
|------|---------|
| BTC | `bc1qnd599khdkv3v3npmj9ufxzf6h4fzanny2acwqr` |
| DOGE | `DL7tuiYCqm3xQjMDXChdxeQxqUGMACn1ZV` |
| ETH | `0x8A28fc47bFFFA03C8f685fa0836E2dBe1CA14F27` |

Copyright (c) 2026 HalloWelt42
