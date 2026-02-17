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
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

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


# Collect all insightface, onnxruntime, and cv2 submodules, data files, and binaries
# This ensures PyInstaller bundles the complete packages (hidden imports,
# data files like images/.pkl, and compiled .so/.dylib files).
_if_datas, _if_binaries, _if_hiddenimports = collect_all('insightface')
_ort_datas, _ort_binaries, _ort_hiddenimports = collect_all('onnxruntime')
_cv2_datas, _cv2_binaries, _cv2_hiddenimports = collect_all('cv2')

datas += _if_datas + _ort_datas + _cv2_datas
binaries += _if_binaries + _ort_binaries + _cv2_binaries

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
    # SSL certificate management for HTTPS in sandbox
    'certifi',
    'ssl',
    # OpenCV - ensure all submodules are bundled for insightface
    'cv2',
] + _if_hiddenimports + _ort_hiddenimports + _cv2_hiddenimports

# Exclude modules that contain non-public/deprecated macOS APIs.
# NOTE: All scipy modules (cython_blas, cython_lapack, _propack) are KEPT because
# they are required at runtime. Their offending symbols (_lsame_, _dcabs1_,
# _xerbla_array__) are binary-patched post-build to use renamed symbols and
# redirected from Accelerate to a shim dylib (see bottom of spec).
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
    runtime_hooks=['hook-cv2-fix.py', 'hook-ssl-certifi.py'],  # Fix cv2 recursion, then configure SSL
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# Filter out Tk/Tcl frameworks from collected binaries.
# These contain non-public macOS API symbols that cause App Store rejection.
# NOTE: scipy binaries are kept — their offending symbols are binary-patched post-build.
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
# POST-BUILD: Patch non-public API symbols in scipy .so files
# ---------------------------------------------------------------------------
# Apple's App Store scanner flags certain LAPACK/BLAS symbol REFERENCES
# (_lsame_, _dcabs1_, _xerbla_array__) in scipy's .so files as "non-public
# APIs" because those names collide with internal Accelerate symbols.
#
# These symbols are UNDEFINED references (type 'U' in nm output) — the .so
# files import them from Apple's Accelerate framework. The previous nmedit
# approach cannot work because nmedit only operates on DEFINED symbols.
#
# Solution:
#   1. Build a shim dylib (libscipy_blas_shim.dylib) that re-exports
#      Accelerate and additionally defines renamed versions of the three
#      flagged functions (lsamZ_, dcabZ1_, xerblZ_array__).
#   2. Binary-patch the .so files to reference the renamed symbol names
#      (same byte length, so all offsets stay valid).
#   3. Use install_name_tool to redirect each .so's Accelerate link to
#      the shim dylib (which re-exports everything else from Accelerate).
#   4. Ad-hoc re-sign everything.
# ---------------------------------------------------------------------------
import subprocess, shutil, glob as _glob

_dist_dir = os.path.join(DISTPATH, 'visarc_ai_server')
_app_dir = os.path.join(DISTPATH, 'visarc_ai_server.app', 'Contents', 'Frameworks')
_shim_src = os.path.join(str(project_root), 'scipy_blas_shim.c')
_shim_name = 'libscipy_blas_shim.dylib'

# Symbol rename map: original bytes → replacement bytes (MUST be same length)
_SYMBOL_RENAMES = {
    b'_lsame_\x00':          b'_lsamZ_\x00',          # 8 bytes
    b'_dcabs1_\x00':         b'_dcabZ1_\x00',         # 9 bytes
    b'_xerbla_array__\x00':  b'_xerblZ_array__\x00',  # 16 bytes
}

# .so files that need patching (glob patterns relative to base_dir)
_PATCH_TARGETS = [
    'scipy/linalg/cython_blas*.so',
    'scipy/linalg/cython_lapack*.so',
    'scipy/sparse/linalg/_propack/_cpropack*.so',
    'scipy/sparse/linalg/_propack/_dpropack*.so',
    'scipy/sparse/linalg/_propack/_spropack*.so',
    'scipy/sparse/linalg/_propack/_zpropack*.so',
]

_ACCELERATE_PATH = '/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate'

def _build_shim_dylib(output_path):
    """Compile scipy_blas_shim.c into a dylib that re-exports Accelerate."""
    if not Path(_shim_src).exists():
        print(f'[SPEC] ERROR: {_shim_src} not found — cannot build shim dylib')
        return False
    install_name = f'@executable_path/../Frameworks/{_shim_name}'
    result = subprocess.run([
        'clang', '-dynamiclib',
        '-o', output_path,
        _shim_src,
        '-Wl,-reexport_framework,Accelerate',
        '-install_name', install_name,
        '-arch', 'arm64',
        '-mmacosx-version-min=11.0',
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(f'[SPEC] Built shim dylib: {output_path}')
        return True
    else:
        print(f'[SPEC] ERROR: Failed to build shim dylib: {result.stderr}')
        return False

def _binary_patch_symbols(so_path):
    """Replace flagged symbol name bytes in a .so file with renamed versions."""
    data = Path(so_path).read_bytes()
    original = data
    patched_syms = []
    for old_bytes, new_bytes in _SYMBOL_RENAMES.items():
        if old_bytes in data:
            data = data.replace(old_bytes, new_bytes)
            patched_syms.append(old_bytes.rstrip(b'\x00').decode())
    if data != original:
        Path(so_path).write_bytes(data)
        print(f'[SPEC] Patched symbols {patched_syms} in {os.path.basename(so_path)}')
        return True
    return False

def _redirect_accelerate_to_shim(so_path):
    """Change the Accelerate load command to point to our shim dylib."""
    shim_install_name = f'@executable_path/../Frameworks/{_shim_name}'
    result = subprocess.run([
        'install_name_tool', '-change',
        _ACCELERATE_PATH, shim_install_name,
        so_path,
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(f'[SPEC] Redirected Accelerate → shim in {os.path.basename(so_path)}')
    else:
        print(f'[SPEC] WARNING: install_name_tool failed on {so_path}: {result.stderr.strip()}')

def _patch_scipy_in_dir(base_dir, shim_dylib_path):
    """Patch all scipy .so files in base_dir and install the shim dylib."""
    if not Path(base_dir).exists():
        return
    # Copy shim dylib into the base directory
    dest_shim = os.path.join(base_dir, _shim_name)
    shutil.copy2(shim_dylib_path, dest_shim)
    print(f'[SPEC] Installed shim dylib at {dest_shim}')
    # Patch each target .so
    for pattern in _PATCH_TARGETS:
        matches = _glob.glob(os.path.join(base_dir, pattern))
        for so_path in matches:
            _binary_patch_symbols(so_path)
            _redirect_accelerate_to_shim(so_path)

# --- Build the shim dylib ---
import tempfile as _tempfile
_shim_build_dir = _tempfile.mkdtemp(prefix='scipy_shim_')
_shim_dylib_path = os.path.join(_shim_build_dir, _shim_name)

print('[SPEC] Post-build: building scipy BLAS shim dylib...')
if _build_shim_dylib(_shim_dylib_path):
    print('[SPEC] Post-build: patching scipy .so files (binary rename + redirect)...')
    _patch_scipy_in_dir(_dist_dir, _shim_dylib_path)
    _patch_scipy_in_dir(_app_dir, _shim_dylib_path)
else:
    print('[SPEC] ERROR: Shim dylib build failed — scipy symbols NOT patched!')

# Clean up temp build directory
shutil.rmtree(_shim_build_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# POST-BUILD: Re-sign binaries after symbol patching
# ---------------------------------------------------------------------------
# Binary patching and install_name_tool invalidate code signatures.
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

print('[SPEC] Post-build: re-signing binaries after symbol patching...')
_adhoc_resign(_dist_dir)
_adhoc_resign_app(os.path.join(DISTPATH, 'visarc_ai_server.app'))