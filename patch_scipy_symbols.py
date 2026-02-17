#!/usr/bin/env python3
"""
Verify the shim + binary-patch approach on the existing built .app bundle.
This script simulates what the spec file post-build step does.
Run it INSTEAD of a full rebuild to validate the fix.
"""
import os, shutil, subprocess, glob, sys
from pathlib import Path

APP_DIR = 'dist/visarc_ai_server.app/Contents/Frameworks'
DIST_DIR = 'dist/visarc_ai_server'
SHIM_SRC = 'scipy_blas_shim.c'
SHIM_NAME = 'libscipy_blas_shim.dylib'

ACCELERATE_PATH = '/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate'

SYMBOL_RENAMES = {
    b'_lsame_\x00':          b'_lsamZ_\x00',
    b'_dcabs1_\x00':         b'_dcabZ1_\x00',
    b'_xerbla_array__\x00':  b'_xerblZ_array__\x00',
}

PATCH_TARGETS = [
    'scipy/linalg/cython_blas*.so',
    'scipy/linalg/cython_lapack*.so',
    'scipy/sparse/linalg/_propack/_cpropack*.so',
    'scipy/sparse/linalg/_propack/_dpropack*.so',
    'scipy/sparse/linalg/_propack/_spropack*.so',
    'scipy/sparse/linalg/_propack/_zpropack*.so',
]

def build_shim(output_path):
    install_name = f'@executable_path/../Frameworks/{SHIM_NAME}'
    result = subprocess.run([
        'clang', '-dynamiclib', '-o', output_path, SHIM_SRC,
        '-Wl,-reexport_framework,Accelerate',
        '-install_name', install_name,
        '-arch', 'arm64', '-mmacosx-version-min=11.0',
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'ERROR building shim: {result.stderr}')
        sys.exit(1)
    print(f'Built {output_path}')

def patch_dir(base_dir, shim_path):
    if not Path(base_dir).exists():
        print(f'Skipping {base_dir} (not found)')
        return
    # Install shim
    dest = os.path.join(base_dir, SHIM_NAME)
    shutil.copy2(shim_path, dest)
    subprocess.run(['codesign', '--force', '--sign', '-', dest], capture_output=True)
    print(f'Installed shim at {dest}')
    
    for pattern in PATCH_TARGETS:
        for so_path in glob.glob(os.path.join(base_dir, pattern)):
            # Binary patch
            data = Path(so_path).read_bytes()
            patched = []
            for old, new in SYMBOL_RENAMES.items():
                if old in data:
                    data = data.replace(old, new)
                    patched.append(old.rstrip(b'\x00').decode())
            if patched:
                Path(so_path).write_bytes(data)
            # Redirect Accelerate
            shim_install = f'@executable_path/../Frameworks/{SHIM_NAME}'
            subprocess.run([
                'install_name_tool', '-change', ACCELERATE_PATH, shim_install, so_path
            ], capture_output=True, text=True)
            # Re-sign
            subprocess.run(['codesign', '--force', '--sign', '-', so_path], capture_output=True)
            print(f'  Patched {os.path.basename(so_path)}: {patched}')

def verify_dir(base_dir):
    if not Path(base_dir).exists():
        return True
    clean = True
    for pattern in PATCH_TARGETS:
        for so_path in glob.glob(os.path.join(base_dir, pattern)):
            result = subprocess.run(['nm', so_path], capture_output=True, text=True)
            for sym in ['_lsame_', '_dcabs1_', '_xerbla_array__']:
                if sym in result.stdout:
                    # Check it's not just a substring of a renamed symbol
                    for line in result.stdout.splitlines():
                        if sym in line and 'lsamZ' not in line and 'dcabZ' not in line and 'xerblZ' not in line:
                            print(f'  FAIL: {os.path.basename(so_path)} still has {sym}')
                            clean = False
    return clean

if __name__ == '__main__':
    print('=== Building shim dylib ===')
    shim_path = '/tmp/libscipy_blas_shim.dylib'
    build_shim(shim_path)
    
    print('\n=== Patching dist folder ===')
    patch_dir(DIST_DIR, shim_path)
    
    print('\n=== Patching .app bundle ===')
    patch_dir(APP_DIR, shim_path)
    
    print('\n=== Verification ===')
    ok1 = verify_dir(DIST_DIR)
    ok2 = verify_dir(APP_DIR)
    if ok1 and ok2:
        print('ALL CLEAN - no flagged symbols remain')
    else:
        print('WARNING: some flagged symbols still present')
        sys.exit(1)
