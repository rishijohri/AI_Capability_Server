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

# Collect all binary files including llama_binaries subdirectories
binaries = []
datas = []  # Initialize datas here for binary directory structure
binary_dir = project_root / 'binary'

if binary_dir.exists():
    # First, collect any binaries in the root binary/ directory (legacy support)
    for binary_file in binary_dir.iterdir():
        if binary_file.is_file():
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

# Bundle certifi CA certificates (critical for HTTPS in sandbox)
try:
    import certifi
    cert_file = certifi.where()
    if Path(cert_file).exists():
        # Bundle as certifi/cacert.pem (expected by runtime hook)
        datas.append((cert_file, 'certifi'))
        print(f"[SPEC] Bundling certifi certificate: {cert_file}")
    else:
        print(f"[SPEC] WARNING: certifi.where() returned non-existent file: {cert_file}")
except ImportError as e:
    print(f"[SPEC] ERROR: certifi not found - {e}")
    print("[SPEC] Install with: pip install certifi")
except Exception as e:
    print(f"[SPEC] ERROR: Failed to bundle certifi: {e}")
    import traceback
    traceback.print_exc()


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
    # SSL certificate management for HTTPS in sandbox
    'certifi',
    'ssl',
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
    runtime_hooks=['hook-ssl-certifi.py'],  # Configure SSL before any imports
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
    name='visarc_ai_server',
    debug=True,  # TEMPORARY: Enable debug mode to capture early failures
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file='entitlements-minimal.plist',  # Minimal sandbox entitlements (uses bundled certifi)
    bundle_identifier='com.memoin.visarcpc.visarc-ai-server',
    info_plist={
        'CFBundleIdentifier': 'com.memoin.visarcpc.visarc-ai-server',
        'CFBundleName': 'visarc_ai_server',
        'CFBundleDisplayName': 'Visarc AI Server',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '11.0',
    },
)

# COLLECT - Bundles everything into a folder
# This creates dist/visarc_ai_server/ directory with:
#   - visarc_ai_server (executable)
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
    name='visarc_ai_server',
)

app = BUNDLE(
    coll,
    name='visarc_ai_server.app', # THIS is where the .app extension goes
    icon=None,
    bundle_identifier='com.memoin.visarcpc.visarc-ai-server',
    info_plist={
        'CFBundleIdentifier': 'com.memoin.visarcpc.visarc-ai-server',
        'CFBundleName': 'visarc_ai_server',
        'CFBundleDisplayName': 'Visarc AI Server',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '11.0',
        'NSHighResolutionCapable': 'True',
        'LSBackgroundOnly': 'True', # Crucial: Tells OS this is a background helper
    }
)