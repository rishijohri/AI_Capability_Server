#!/bin/bash
# Direct launch test with immediate stderr visibility

set -e

APP_PATH="dist/visarc_ai_server.app"
EXECUTABLE="$APP_PATH/Contents/MacOS/visarc_ai_server"

echo "=========================================="
echo "DIRECT LAUNCH TEST (stderr visible)"
echo "=========================================="
echo ""

if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: Executable not found at $EXECUTABLE"
    echo "Run './build.sh' first"
    exit 1
fi

echo "Launching: $EXECUTABLE"
echo ""
echo "=========================================="
echo "OUTPUT:"
echo "=========================================="
echo ""

# Run directly - all stderr/stdout will be visible immediately
$EXECUTABLE

echo ""
echo "=========================================="
echo "App exited"
echo "=========================================="
