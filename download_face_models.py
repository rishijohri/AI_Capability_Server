#!/usr/bin/env python3
"""
Download InsightFace models to a local directory.

This script downloads the buffalo_l model pack to a specified directory,
allowing you to use face recognition without downloading models at runtime.

Usage:
    python download_face_models.py [target_directory]
    
    If no directory is specified, models will be downloaded to ./face_models

After downloading, update your config:
    POST /api/config
    {
        "face_models_dir": "/path/to/face_models"
    }
"""

import sys
import os
from pathlib import Path

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("Error: InsightFace not installed.")
    print("Install with: pip install insightface onnxruntime")
    sys.exit(1)


def download_models(target_dir: str = None):
    """
    Download InsightFace models to specified directory.
    
    Args:
        target_dir: Target directory path. If None, uses ./face_models
    """
    if target_dir is None:
        target_dir = "./face_models"
    
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading InsightFace buffalo_l model to: {target_path}")
    print("This may take a few minutes (~400MB download)...")
    print()
    
    try:
        # Initialize FaceAnalysis with target directory
        # This will download the model if it doesn't exist
        app = FaceAnalysis(
            name='buffalo_l',
            root=str(target_path),
            providers=['CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        print()
        print("✓ Download complete!")
        print()
        print("Model files are located at:")
        print(f"  {target_path}")
        print()
        print("To use these models, update your server configuration:")
        print('  POST http://localhost:8000/api/config')
        print('  {')
        print(f'    "face_models_dir": "{target_path}"')
        print('  }')
        print()
        print("Or set it in your config before starting the server.")
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    # Handle help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("=" * 70)
        print("InsightFace Model Downloader")
        print("=" * 70)
        print()
        print(__doc__)
        return
    
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("=" * 70)
    print("InsightFace Model Downloader")
    print("=" * 70)
    print()
    
    download_models(target_dir)


if __name__ == "__main__":
    main()
