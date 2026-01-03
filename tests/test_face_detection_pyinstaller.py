#!/usr/bin/env python3
"""Test face detection in PyInstaller environment."""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.resource_paths import get_face_models_path, get_base_path

print(f"Base path: {get_base_path()}")
print(f"Face models path: {get_face_models_path()}")
print(f"Face models path exists: {get_face_models_path().exists()}")

models_dir = get_face_models_path() / "models"
print(f"Models directory: {models_dir}")
print(f"Models directory exists: {models_dir.exists()}")

if models_dir.exists():
    buffalo_dir = models_dir / "buffalo_l"
    print(f"Buffalo_l directory: {buffalo_dir}")
    print(f"Buffalo_l directory exists: {buffalo_dir.exists()}")
    
    if buffalo_dir.exists():
        onnx_files = list(buffalo_dir.glob("*.onnx"))
        print(f"Found {len(onnx_files)} ONNX files:")
        for f in onnx_files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")

print("\n--- Testing InsightFace initialization ---")

try:
    from insightface.app import FaceAnalysis
    print("InsightFace imported successfully")
    
    model_root = str(get_face_models_path())
    print(f"Initializing FaceAnalysis with root: {model_root}")
    
    face_app = FaceAnalysis(
        name='buffalo_l',
        root=model_root,
        providers=['CPUExecutionProvider']
    )
    print("FaceAnalysis created successfully")
    
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("FaceAnalysis prepared successfully")
    
    print("\nInsightFace initialization successful!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
