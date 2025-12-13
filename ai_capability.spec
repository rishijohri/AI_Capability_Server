# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Capability Server (macOS/Linux).

This spec file is designed for Unix-based systems where binaries
have no file extensions (e.g., llama-server, llama-cli).

For Windows, use ai_capability_windows.spec instead, which handles
.exe extensions and Windows-specific requirements.

Build with:
    pyinstaller ai_capability.spec
Or use the build script:
    ./build.sh
"""

import sys
from pathlib import Path

# Get project root
project_root = Path(SPECPATH)

# Collect all binary files
binaries = []
binary_dir = project_root / 'binary'
if binary_dir.exists():
    for binary_file in binary_dir.iterdir():
        if binary_file.is_file():
            binaries.append((str(binary_file), 'binary'))

# Collect all model files
datas = []
model_dir = project_root / 'model'
if model_dir.exists():
    # Add GGUF model files
    for model_file in model_dir.iterdir():
        if model_file.is_file() and model_file.suffix in ['.gguf', '.bin']:
            datas.append((str(model_file), 'model'))
    
    # Add InsightFace models directory recursively (includes all subdirectories)
    models_subdir = model_dir / 'models'
    if models_subdir.exists():
        for root, dirs, files in models_subdir.walk():
            for file in files:
                src_path = root / file
                # Calculate relative path from model_dir to preserve directory structure
                rel_path = src_path.relative_to(model_dir)
                dest_dir = 'model' / rel_path.parent
                datas.append((str(src_path), str(dest_dir)))



# Hidden imports that PyInstaller might miss
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'aiohttp',
    'multipart',
    'pydantic',
    'fastapi',
    'insightface',
    'insightface.app',
    'insightface.app.face_analysis',
    'insightface.model_zoo',
    'insightface.model_zoo.landmark',
    'insightface.utils',
    'insightface.utils.transform',
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.capi.onnxruntime_pybind11_state',
]

# Analysis
a = Analysis(
    ['run_server.py'],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# PYZ (Python zip archive)
pyz = PYZ(a.pure)

# EXE - Creates the executable
# NOTE: exclude_binaries=True ensures folder-based distribution (not single file)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # CRITICAL: Must be True for folder-based distribution
    name='ai_capability_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# COLLECT - Bundles everything into a folder
# This creates dist/ai_capability_server/ directory with:
#   - ai_capability_server (executable)
#   - binary/ (llama binaries)
#   - model/ (model files)
#   - All Python dependencies
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ai_capability_server',
)
