#!/bin/bash

# Name der requirements-Datei
REQUIREMENTS_FILE="requirements.txt"

# Überprüfen, ob die requirements.txt-Datei existiert
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE nicht gefunden."
    exit 1
fi

# Überprüfen, ob uv installiert ist
if ! command -v uv &> /dev/null; then
    echo "Error: uv ist nicht installiert. Bitte installieren Sie uv zuerst."
    echo "Sie können uv mit folgendem Befehl installieren:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Installieren der Pakete
echo "Installiere Pakete aus $REQUIREMENTS_FILE mit uv..."
uv pip install -r "$REQUIREMENTS_FILE"

if [ $? -eq 0 ]; then
    echo "Installation erfolgreich abgeschlossen."
else
    echo "Error: Installation fehlgeschlagen."
    exit 1
fi