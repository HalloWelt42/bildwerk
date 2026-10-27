#!/bin/sh
# Bildwerk installieren oder aktualisieren: Umgebung mit den erprobten Paketständen und mflux auf dem Commit aus
# pyproject.toml, dann je Modell die Dienstdatei aus deploy/ im Maschinenraum (~/Maschinenraum/dienste/bildwerk-<modell>.toml),
# sofern es den Maschinenraum gibt und die Datei noch nicht existiert. Gestartet und geladen wird nichts: starten lässt
# die Leitstelle über den Maschinenraum, die Gewichte lädt von Hand "bildwerk gewichte <modell> --laden".
set -eu

HIER="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
DIENSTE="${MASCHINENRAUM_ORDNER:-$HOME/Maschinenraum}/dienste"

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || { echo "Python 3.11 oder neuer nötig" >&2; exit 1; }
[ -d "$HIER/.venv" ] || "$PYTHON" -m venv "$HIER/.venv"
"$HIER/.venv/bin/pip" install --quiet --upgrade pip
"$HIER/.venv/bin/pip" install --quiet -c "$HIER/einschraenkungen.txt" -e "$HIER[mflux,test]"
echo "Umgebung bereit: $HIER/.venv"

if [ -d "$DIENSTE" ]; then
  for vorlage in "$HIER"/deploy/bildwerk-*.toml; do
    datei="$DIENSTE/$(basename "$vorlage")"
    if [ -f "$datei" ]; then echo "bleibt: $datei"; continue; fi
    # Arbeitsordner ist dieser Klon; alles andere wie in deploy/
    sed "s|^ordner = .*|ordner = \"$HIER\"|" "$vorlage" > "$datei"
    echo "eingetragen: $datei"
  done
else
  echo "Kein Maschinenraum unter $DIENSTE; Dienste von Hand starten: $HIER/.venv/bin/bildwerk <modell>"
fi
