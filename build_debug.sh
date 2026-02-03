#!/bin/bash
set -e

echo "=============================================="
echo "Building AI Server with Debug File Logging"
echo "=============================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf build dist macos/Runner/Resources/visarc_ai_server.app

# Build with PyInstaller
echo "Building with PyInstaller..."
pyinstaller ai_capability.spec --clean --noconfirm

# Copy to Resources
echo "Copying to Flutter Resources..."
mkdir -p macos/Runner/Resources
cp -R dist/visarc_ai_server.app macos/Runner/Resources/

echo ""
echo "✅ Build complete"
echo ""
echo "Next step: Sign with debug entitlements"
echo "  Update your signing script to use:"
echo "  ENTITLEMENTS=\"/Users/rishijohri/Documents/Projects/AI_Capability/entitlements-debug-file-logging.plist\""
echo ""
echo "After signing, the debug log will be at:"
echo "  /Users/rishijohri/Documents/visarc_ai_server_debug.log"
echo ""
