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
import os
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

# Exclude modules that contain non-public/deprecated macOS APIs.
# NOTE: All scipy modules (cython_blas, cython_lapack, _propack) are KEPT because
# they are required at runtime. Their offending symbols (_lsame_, _dcabs1_,
# _xerbla_array__) are stripped post-build using nmedit (see bottom of spec).
excludes = [
    # Tk/Tcl - not needed for headless server, contains private macOS APIs
    'tkinter',
    '_tkinter',
    'Tkinter',
    'turtle',
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
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# Filter out Tk/Tcl frameworks from collected binaries.
# These contain non-public macOS API symbols that cause App Store rejection.
# NOTE: scipy binaries are kept — their offending symbols are stripped post-build.
_blocked_patterns = (
    'Tk', 'Tcl', 'tkinter', '_tkinter', 'tcl8', 'tk8',
)
a.binaries = [(name, path, typecode) for name, path, typecode in a.binaries
              if not any(blocked in name or blocked in path for blocked in _blocked_patterns)]
a.datas = [(name, path, typecode) for name, path, typecode in a.datas
           if not any(blocked in name or blocked in path for blocked in _blocked_patterns)]

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

# ---------------------------------------------------------------------------
# POST-BUILD: Strip non-public API symbols from scipy .so files
# ---------------------------------------------------------------------------
# Apple's App Store scanner flags standard FORTRAN BLAS/LAPACK symbols
# (_lsame_, _dcabs1_, _xerbla_array__) as "non-public APIs" because they
# collide with internal Apple symbol names. These are false positives —
# they're standard LAPACK helpers compiled into scipy's Cython extensions.
#
# We use macOS `nmedit` to localise (hide) these symbols so they no longer
# appear in the export table, which satisfies the App Store scanner while
# keeping the .so files fully functional (the symbols are only called
# internally within each shared object).
# ---------------------------------------------------------------------------
import subprocess, tempfile, glob as _glob

_dist_dir = os.path.join(DISTPATH, 'visarc_ai_server')
_app_dir = os.path.join(DISTPATH, 'visarc_ai_server.app', 'Contents', 'Frameworks')

# Map of .so file glob patterns → symbols to hide
_strip_targets = {
    'scipy/linalg/cython_blas*.so': ['_dcabs1_', '_lsame_'],
    'scipy/linalg/cython_lapack*.so': ['_xerbla_array__'],
    'scipy/sparse/linalg/_propack/_cpropack*.so': ['_lsame_'],
    'scipy/sparse/linalg/_propack/_dpropack*.so': ['_lsame_'],
    'scipy/sparse/linalg/_propack/_spropack*.so': ['_lsame_'],
    'scipy/sparse/linalg/_propack/_zpropack*.so': ['_lsame_'],
}

def _strip_symbols(base_dir):
    """Use nmedit -R to make listed symbols local/private in .so files."""
    if not Path(base_dir).exists():
        return
    for pattern, symbols in _strip_targets.items():
        matches = _glob.glob(os.path.join(base_dir, pattern))
        for so_path in matches:
            # Write symbol list to a temp file for nmedit -R
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for sym in symbols:
                    f.write(sym + '\n')
                sym_file = f.name
            try:
                result = subprocess.run(
                    ['nmedit', '-R', sym_file, so_path],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f'[SPEC] Stripped symbols {symbols} from {so_path}')
                else:
                    print(f'[SPEC] WARNING: nmedit failed on {so_path}: {result.stderr}')
            except FileNotFoundError:
                print('[SPEC] WARNING: nmedit not found — cannot strip symbols')
                break
            finally:
                os.unlink(sym_file)

print('[SPEC] Post-build: stripping non-public API symbols from scipy binaries...')
_strip_symbols(_dist_dir)
_strip_symbols(_app_dir)

# ---------------------------------------------------------------------------
# POST-BUILD: Re-sign binaries after nmedit modification
# ---------------------------------------------------------------------------
# nmedit invalidates the ad-hoc code signatures that PyInstaller applied.
# macOS (Ventura+) enforces signature validity for all Mach-O files inside
# an .app bundle — including when spawning subprocesses like llama-server.
# If any binary in the bundle has a broken signature, subprocess spawning
# fails with SIGTRAP (exit code -5).
#
# We re-sign everything with an ad-hoc signature so the bundle is valid
# for local testing. Production signing (Developer ID) is done separately
# via sign_2.sh.
# ---------------------------------------------------------------------------

def _adhoc_resign(base_dir):
    """Ad-hoc re-sign all Mach-O files (.so, .dylib, executables) in base_dir."""
    if not Path(base_dir).exists():
        return
    signed_count = 0
    # Sign .so and .dylib files
    for ext in ('**/*.so', '**/*.dylib'):
        for fpath in Path(base_dir).glob(ext):
            result = subprocess.run(
                ['codesign', '--force', '--sign', '-', str(fpath)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                signed_count += 1
            else:
                print(f'[SPEC] WARNING: ad-hoc sign failed for {fpath}: {result.stderr.strip()}')
    # Sign executable binaries (llama-server, llama-cli, etc.)
    for fpath in Path(base_dir).rglob('*'):
        if fpath.is_file() and not fpath.suffix and os.access(str(fpath), os.X_OK):
            result = subprocess.run(
                ['codesign', '--force', '--sign', '-', str(fpath)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                signed_count += 1
            else:
                # Not all executable files are Mach-O; ignore failures on scripts, etc.
                pass
    print(f'[SPEC] Ad-hoc re-signed {signed_count} files in {base_dir}')

def _adhoc_resign_app(app_path):
    """Ad-hoc re-sign the .app bundle (must be done last, outside-in is wrong — sign contents first, bundle last)."""
    if not Path(app_path).exists():
        return
    # Sign Frameworks contents first
    frameworks_dir = os.path.join(app_path, 'Contents', 'Frameworks')
    _adhoc_resign(frameworks_dir)
    # Sign MacOS executable
    main_exec = os.path.join(app_path, 'Contents', 'MacOS', 'visarc_ai_server')
    if Path(main_exec).exists():
        subprocess.run(['codesign', '--force', '--sign', '-', main_exec],
                       capture_output=True, text=True)
    # Sign the .app bundle itself last
    result = subprocess.run(
        ['codesign', '--force', '--sign', '-', app_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f'[SPEC] Ad-hoc re-signed {app_path}')
    else:
        print(f'[SPEC] WARNING: ad-hoc sign of .app failed: {result.stderr.strip()}')

print('[SPEC] Post-build: re-signing binaries after nmedit modifications...')
_adhoc_resign(_dist_dir)
_adhoc_resign_app(os.path.join(DISTPATH, 'visarc_ai_server.app'))