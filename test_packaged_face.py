#!/usr/bin/env python3
"""Minimal test for face detection in packaged app."""

import sys
from pathlib import Path

print(f"Python executable: {sys.executable}")
print(f"sys.frozen: {getattr(sys, 'frozen', False)}")
if hasattr(sys, '_MEIPASS'):
    print(f"sys._MEIPASS: {sys._MEIPASS}")

# Test resource paths
if getattr(sys, 'frozen', False):
    base_path = Path(sys._MEIPASS)
else:
    base_path = Path(__file__).parent

print(f"\nBase path: {base_path}")

model_path = base_path / "model"
print(f"Model path: {model_path}")
print(f"Model path exists: {model_path.exists()}")

if model_path.exists():
    models_dir = model_path / "models" / "buffalo_l"
    print(f"Buffalo_l path: {models_dir}")
    print(f"Buffalo_l exists: {models_dir.exists()}")
    
    if models_dir.exists():
        onnx_files = list(models_dir.glob("*.onnx"))
        print(f"ONNX files found: {len(onnx_files)}")
        for f in onnx_files:
            print(f"  - {f.name}")

print("\n--- Testing InsightFace ---")
try:
    from insightface.app import FaceAnalysis
    print("✓ InsightFace imported")
    
    model_root = str(model_path)
    print(f"Creating FaceAnalysis with root: {model_root}")
    
    face_app = FaceAnalysis(
        name='buffalo_l',
        root=model_root,
        providers=['CPUExecutionProvider']
    )
    print("✓ FaceAnalysis created")
    
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("✓ FaceAnalysis prepared")
    
    print("\n✅ SUCCESS: Face detection is working!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
