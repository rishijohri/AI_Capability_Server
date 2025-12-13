#!/bin/bash

# AI Capability Server - Build Script (macOS/Linux)
# This script builds the standalone executable using PyInstaller

set -e  # Exit on error

echo "============================================================"
echo "AI Capability Server - Build Script (macOS/Linux)"
echo "============================================================"

# Detect if running on Windows (Git Bash, WSL, etc.)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo ""
    echo "⚠️  Windows detected!"
    echo ""
    echo "This is a Unix shell script. On Windows, please use:"
    echo "  - PowerShell: .\build.ps1"
    echo "  - Command Prompt: build.bat"
    echo ""
    echo "If you're in Git Bash or WSL, this script may work but"
    echo "native Windows scripts are recommended."
    echo ""
    read -p "Continue with bash script anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Verify PyInstaller is installed
if ! command -v pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Check for required directories
echo ""
echo "Checking required directories..."
if [ ! -d "binary" ]; then
    echo "❌ ERROR: binary/ directory not found"
    echo "   Please create it and add llama binaries (llama-server, llama-cli, etc.)"
    exit 1
fi

if [ ! -d "model" ]; then
    echo "❌ ERROR: model/ directory not found"
    echo "   Please create it and add model files (.gguf)"
    exit 1
fi

# Count files
binary_count=$(find binary -type f | wc -l | tr -d ' ')
model_count=$(find model -type f -name "*.gguf" | wc -l | tr -d ' ')

echo "✅ Found $binary_count binaries in binary/"
echo "✅ Found $model_count model files in model/"

# Check disk space (models can be large)
echo ""
echo "Checking directory sizes..."
model_size=$(du -sh model | cut -f1)
binary_size=$(du -sh binary | cut -f1)
echo "  Model directory: $model_size"
echo "  Binary directory: $binary_size"

echo ""
echo "Building AI Capability Server executable..."
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/

# Build with PyInstaller
echo "Running PyInstaller..."
pyinstaller --clean ai_capability.spec

# Check if build was successful
if [ -f "dist/ai_capability_server/ai_capability_server" ]; then
    echo ""
    echo "============================================================"
    echo "✅ Build successful!"
    echo "============================================================"
    echo ""
    echo "Output location: dist/ai_capability_server/"
    echo ""
    echo "To run the application:"
    echo "  cd dist/ai_capability_server"
    echo "  ./ai_capability_server"
    echo ""
    echo "To distribute:"
    echo "  cd dist"
    echo "  zip -r ai_capability_server.zip ai_capability_server/"
    echo ""
    
    # Show size
    dist_size=$(du -sh dist/ai_capability_server | cut -f1)
    echo "Total packaged size: $dist_size"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "❌ Build failed!"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above."
    exit 1
fi
