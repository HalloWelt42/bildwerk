---
zweck: "Die Bilddienste des Hauses, je Bildmodell ein eigener Prozess mit derselben Schnittstelle: Bilder aus Text für die Apps, vermittelt von der Leitstelle."
geraet: [mac]
abhaengigkeiten: [maschinenraum]
daten:
  - "Mac: die Gewichte in der Modellablage `~/Maschinenraum/modelle/huggingface` (MOSS-Image-SFT, 27,2 GB)"
  - "Mac: Dienstdateien `~/Maschinenraum/dienste/bildwerk-*.toml` (je Modell eine)"
sicherung:
  - "nichts Eigenes zu sichern: die Dienstdateien schreibt die Installation neu, die Gewichte lassen sich neu laden; die Modellablage gehört zur Sicherung des Macs (siehe Grundlage Mac)"
  - "das Repo liegt bisher nur auf dem Mac (keine entfernte Ablage): der Ordner gehört zur Sicherung des Macs"
schluessel_orte: []
einrichten: einrichten.sh
stand: 60ff107
---

# Bildwerk

## Zweck

Das Bildwerk rechnet Bilder für die Apps: je Bildmodell ein Dienst, ein eigener Prozess mit genau einem Modell, der es
beim Start lädt und den Speicher beim Anhalten wieder frei gibt. Ein Bilddienst ist eine Rechenressource wie jedes
andere Modell: er startet nie von selbst. Der [[Maschinenraum]] startet und hält ihn an, wenn die [[Leitstelle]] es
verlangt; sie entscheidet über Laden, Entladen und Vorrang, und Apps (auch der Assistent) erreichen ihn nur über sie.

| Dienst | Fähigkeit | Modell |
|---|---|---|
| moss-image | bild_erzeugen | MOSS-Image-SFT (Apache-2.0), 8 Bit, gerechnet mit mflux (MLX); Bilder aus Text, kein Bearbeiten |

Sichtbar heißt der Dienst nach seinem Modell ("MOSS-Image", `name` der Dienstdatei); Maschinenraum, Leitstelle und
Apps setzen die Aufgabe klein dahinter ("MOSS-Image · Bilderzeugung"). Die Kennung `bildwerk-moss-image` bleibt.

Alle Dienste sprechen dieselbe Schnittstelle im üblichen Bildformat (`POST /v1/images/generations`,
`POST /v1/images/edits`, `GET /v1/models` mit Steckbrief, `GET /gesundheit`); Einzelheiten im README des Repos. Port
und Prüfadresse stehen unten im Block "Auf einen Blick" (aus den Dienstdateien).

## Was sie braucht

- Mac nach [[Grundlage Mac]]: Python ab 3.11, die Modellablage.
- [[Maschinenraum]]: die Installation trägt die Dienste in `~/Maschinenraum/dienste/` ein; gestartet werden sie dort
  auf Geheiß der [[Leitstelle]], nie von der Installation selbst.
- Die Pakete kommen mit festen Ständen aus `einschraenkungen.txt`, mflux auf dem Commit aus `pyproject.toml`.
- Die Gewichte je Modell in der Modellablage, vollständig nach der Prüfliste des Modells; für MOSS-Image rund 27 GB.

## Neu einrichten

Zwei Wege. Aus der Sicherung: den Ordner {{ordner}} und die Modellablage zurückspielen (siehe [[Grundlage Mac]]); dann
fehlen nur Umgebung und Dienstdatei. Frisch: Ordner zurückspielen, Installation, die Gewichte einmal laden.

### 1. Ordner und Installation

```
git clone {{repo}} {{ordner}}
cd {{ordner}} && ./einrichten.sh
```

`./einrichten.sh` ruft `./installieren.sh`: `.venv` im Ordner mit den Ständen aus `einschraenkungen.txt` und mflux auf
dem gepinnten Commit, dann je Modell eine Datei `~/Maschinenraum/dienste/bildwerk-<modell>.toml` aus `deploy/`, aber nur,
wenn es sie noch nicht gibt (Änderungen von Hand bleiben). Gestartet und geladen wird nichts.

Probe: `ls ~/Maschinenraum/dienste/bildwerk-*.toml` nennt `bildwerk-moss-image.toml`, und der Maschinenraum zeigt den
Dienst unter Dienste als aus.

### 2. Gewichte

Aus der Sicherung liegen sie in der Modellablage. Frisch einmal laden (rund 27 GB aus dem Netz, nur Gewichte,
Konfiguration und Tokenizer, nie Code des Modell-Repos):
`HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte moss-image --laden`. Die Dienstdatei sperrt
danach das Netz (`HF_HUB_OFFLINE=1`); der Dienst lädt nie selbst.

Probe: `HF_HUB_CACHE=~/Maschinenraum/modelle/huggingface .venv/bin/bildwerk gewichte moss-image --pruefsummen` meldet
"vollständig" und "Prüfsummen stimmen".

### 3. Starten über die Leitstelle

Die [[Leitstelle]] startet den Dienst über den Maschinenraum, sobald eine App ein Bild mit diesem Modell verlangt. Ohne
Modell lässt sich die Schnittstelle im Probemodus prüfen: `.venv/bin/bildwerk moss-image --probe --port 18791`, danach
mit Strg-C beenden.

Probe: im Probemodus meldet `curl -s http://127.0.0.1:18791/gesundheit` den Zustand `probe` und
`"gewichte_vollstaendig":true`; nach einem Start durch die Leitstelle meldet `curl -s {{pruefadresse:bildwerk-moss-image}}`
den Zustand `bereit`.

## Sicherung und Rückspielen

Das Bildwerk hält keine eigenen Daten: die Dienstdateien schreibt die Installation neu, erzeugte Bilder gehen direkt in
der Antwort an den Aufrufer. Was eine Neueinrichtung dauern lässt, sind die Gewichte in der Modellablage (rund 27 GB je
Modell); sie gehören zur Sicherung des Macs (siehe [[Grundlage Mac]]). Das Repo liegt bisher nur auf dem Mac und ist
damit nur über die Sicherung des Macs wiederherstellbar.

## Wenn etwas klemmt

- Der Dienst meldet `gestoert` mit "Gewichte unvollständig": die Gewichte fehlen in der Modellablage oder ein Download
  brach ab; Schritt 2.
- Der Dienst bleibt bei `laedt`: das Laden liest rund 27 GB und quantisiert auf 8 Bit; das dauert. Endet er nach der
  Startzeit der Dienstdatei, steht der Grund im Protokoll des Dienstes im Maschinenraum.
- Antwort `400`: Größe oder Schritte passen nicht zum Modell (Grenzen im Steckbrief unter `/v1/models`); `501` bei
  Bearbeiten: das Modell kann es nicht.
- Speicher knapp: der Dienst braucht in 8 Bit rund 24 GB (Schätzung bis zur ersten Messung); weniger mit
  `BILDWERK_QUANTISIERUNG = "4bit"` unter `[umgebung]` der Dienstdatei. Die Leitstelle hält Dienste an, die sie nicht
  braucht.
