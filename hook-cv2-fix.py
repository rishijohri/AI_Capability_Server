"""
PyInstaller runtime hook to fix OpenCV (cv2) recursion error.

Problem: OpenCV's cv2/__init__.py bootstrap() function does:
  1. Sets sys.OpenCV_LOADER = True (recursion guard)
  2. Pops 'cv2' from sys.modules (the wrapper package)
  3. Calls importlib.import_module("cv2") to load the native .so extension

In a PyInstaller bundle, step 3 re-imports the cv2 wrapper __init__.py
(instead of finding cv2.abi3.so directly), which triggers bootstrap() again,
hits the recursion guard, and raises ImportError.

Fix: Patch importlib.import_module so that the recursive "import cv2"
call inside bootstrap() loads the native .so extension directly (using
spec_from_file_location with the correct module name "cv2") instead of
going through PyInstaller's importer which would re-enter the wrapper.
"""

import sys
import os


def _fix_cv2_for_pyinstaller():
    """Patch importlib.import_module to prevent cv2 bootstrap recursion."""
    if not getattr(sys, 'frozen', False) or not hasattr(sys, '_MEIPASS'):
        return  # Not running in PyInstaller bundle

    meipass = sys._MEIPASS

    # Find the cv2 directory and native extension in the bundle
    cv2_dir = os.path.join(meipass, 'cv2')
    if not os.path.isdir(cv2_dir):
        print("[HOOK-CV2] WARNING: cv2 directory not found in bundle", file=sys.stderr)
        return

    # Look for the native extension file (cv2.abi3.so on macOS/Linux, cv2.pyd on Windows)
    native_ext = None
    for fname in os.listdir(cv2_dir):
        if fname.startswith('cv2') and (fname.endswith('.so') or fname.endswith('.pyd')):
            native_ext = os.path.join(cv2_dir, fname)
            break

    if native_ext is None:
        print("[HOOK-CV2] WARNING: cv2 native extension (.so/.pyd) not found", file=sys.stderr)
        return

    # Patch importlib.import_module to intercept the recursive call from bootstrap()
    import importlib
    import importlib.util

    _original_import_module = importlib.import_module

    def _patched_import_module(name, package=None):
        # Only intercept when bootstrap() is running (OpenCV_LOADER flag is set)
        # and the import is for "cv2" (the recursive call)
        if name == "cv2" and getattr(sys, 'OpenCV_LOADER', False):
            # Load the native .so directly with module name "cv2" so that
            # the export function PyInit_cv2 matches
            spec = importlib.util.spec_from_file_location("cv2", native_ext)
            if spec is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        return _original_import_module(name, package)

    importlib.import_module = _patched_import_module

    print(f"[HOOK-CV2] Patched importlib.import_module for cv2 recursion fix "
          f"(native ext: {os.path.basename(native_ext)})", file=sys.stderr)


_fix_cv2_for_pyinstaller()
