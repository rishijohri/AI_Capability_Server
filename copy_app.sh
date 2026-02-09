#!/bin/bash
set -e

SRC="dist/visarc_ai_server.app"
DEST="/Users/rishijohri/Documents/Projects/data_storage_pc/macos/Runner/Resources/"

if [ ! -d "$SRC" ]; then
    echo "❌ Source not found: $SRC"
    exit 1
fi

echo "Removing old bundle..."
rm -rf "${DEST}visarc_ai_server.app"

echo "Copying $SRC → $DEST"
cp -R "$SRC" "$DEST"

echo "✅ Done ($(du -sh "${DEST}visarc_ai_server.app" | cut -f1))"
