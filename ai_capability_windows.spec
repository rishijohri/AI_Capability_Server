# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AI Capability Server (Windows).

This spec file is specifically designed for Windows systems where
binaries have .exe extensions (e.g., llama-server.exe, llama-cli.exe).

For macOS/Linux, use ai_capability.spec instead, which handles
binaries without extensions.

Build with:
    pyinstaller ai_capability_windows.spec
Or use a build script:
    .\build.ps1  (PowerShell)
    build.bat    (Command Prompt)
"""

import sys
from pathlib import Path

# Get project root
project_root = Path(SPECPATH)

# Collect all binary files including llama_binaries subdirectories (.exe files on Windows)
binaries = []
datas = []  # Initialize datas here for binary directory structure
binary_dir = project_root / 'binary'

if binary_dir.exists():
    # First, collect any binaries in the root binary/ directory (legacy support)
    for binary_file in binary_dir.iterdir():
        if binary_file.is_file():
            # On Windows, we expect .exe files
            if binary_file.suffix.lower() in ['.exe', '.dll']:
                binaries.append((str(binary_file), 'binary'))
            elif binary_file.suffix == '':
                # Also include files without extension (in case)
                binaries.append((str(binary_file), 'binary'))
    
    # Then, collect binaries from llama_binaries subdirectories (new structure)
    llama_binaries_dir = binary_dir / 'llama_binaries'
    if llama_binaries_dir.exists():
        for config_dir in llama_binaries_dir.iterdir():
            if config_dir.is_dir():
                # Collect all files in each configuration directory
                for binary_file in config_dir.iterdir():
                    if binary_file.is_file():
                        # Preserve directory structure: binary/llama_binaries/[config]/[binary]
                        rel_path = binary_file.relative_to(binary_dir)
                        dest_dir = 'binary' / rel_path.parent
                        datas.append((str(binary_file), str(dest_dir)))

# Collect all model files (datas already initialized above)
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

# EXE - Creates the Windows executable (.exe)
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
    console=True,  # Show console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one: 'icon.ico'
)

# COLLECT - Bundles everything into a folder
# This creates dist/ai_capability_server/ directory with:
#   - ai_capability_server.exe (Windows executable)
#   - binary/ (llama .exe binaries)
#   - model/ (model files)
#   - All Python dependencies and DLLs
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ai_capability_server',
)
