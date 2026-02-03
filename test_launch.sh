#!/bin/bash
# Quick test script to check if the app launches and view logs

set -e

echo "========================================"
echo "Testing visarc_ai_server launch"
echo "========================================"
echo ""

# Find the app bundle
APP_PATH="dist/visarc_ai_server.app"

if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: App not found at $APP_PATH"
    echo "Run './build.sh' first to build the app"
    exit 1
fi

# Get the executable path
EXECUTABLE="$APP_PATH/Contents/MacOS/visarc_ai_server"

if [ ! -f "$EXECUTABLE" ]; then
    echo "ERROR: Executable not found at $EXECUTABLE"
    exit 1
fi

echo "Found app at: $APP_PATH"
echo "Executable: $EXECUTABLE"
echo ""

echo "Note: All logs go to stderr/stdout (no file logging)"
echo "      Visible in Console.app or this terminal"
echo ""

echo "========================================"
echo "Starting app in background..."
echo "========================================"
echo ""

# Launch the app in background
$EXECUTABLE &
APP_PID=$!

echo "App launched with PID: $APP_PID"
echo ""
echo "Waiting 3 seconds for startup..."
sleep 3

# Check if process is still running
if ps -p $APP_PID > /dev/null 2>&1; then
    echo "✅ SUCCESS: App is still running (PID: $APP_PID)"
    echo "Stopping app..."
    kill $APP_PID 2>/dev/null || true
else
    echo "❌ FAILURE: App crashed or exited early"
    echo ""
fi

echo ""
echo "========================================"
echo "macOS Console Logs (last 2 minutes):"
echo "========================================"
log show --predicate 'process == "visarc_ai_server"' --last 2m --style compact

echo ""
echo "========================================"
echo "Check for crash reports:"
echo "========================================"
CRASH_DIR="$HOME/Library/Logs/DiagnosticReports"
CRASH_FILES=$(ls -t "$CRASH_DIR"/visarc_ai_server*.crash 2>/dev/null | head -1)
if [ -n "$CRASH_FILES" ]; then
    echo "Found crash report:"
    ls -lh "$CRASH_FILES"
    echo ""
    echo "Content:"
    cat "$CRASH_FILES" | head -100
else
    echo "No crash reports found"
fi

echo ""
echo "========================================"
echo "Test complete"
echo "========================================"
