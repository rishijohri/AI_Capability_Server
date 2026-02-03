# Face Models Setup Guide

## Quick Start

### Download Models (Recommended for offline use)

```bash
# Download to default location (./face_models)
python download_face_models.py

# Or specify custom directory
python download_face_models.py /path/to/your/models
```

### Configure Server

After downloading, configure the server to use the local models:

```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "face_models_dir": "/absolute/path/to/face_models"
  }'
```

**Important:** Use absolute paths, not relative paths.

## Model Information

- **Model Name:** buffalo_l
- **Size:** ~400MB
- **Provider:** InsightFace
- **Embedding Size:** 512 dimensions
- **Default Location:** `~/.insightface/models/buffalo_l/`

## Directory Structure

After downloading, the directory will contain:

```
face_models/
└── models/
    └── buffalo_l/
        ├── 1k3d68.onnx          # 3D face alignment
        ├── 2d106det.onnx        # 106-point face landmarks
        ├── det_10g.onnx         # Face detection model
        ├── genderage.onnx       # Gender and age prediction
        └── w600k_r50.onnx       # Face recognition model (main)
```

## Usage Examples

### Example 1: Default Location (Auto-download on first use)

```python
# No configuration needed
# Models will be downloaded to ~/.insightface on first use
```

### Example 2: Custom Location

```python
# 1. Download models
python download_face_models.py ./my_models

# 2. Configure before starting server
from app.config import update_config
update_config(face_models_dir="./my_models")

# 3. Start server
python run_server.py
```

### Example 3: Runtime Configuration

```python
import requests

# Start server first
# Then configure model location
response = requests.post(
    "http://localhost:8000/api/config",
    json={"face_models_dir": "/absolute/path/to/models"}
)
print(response.json())
```

## Troubleshooting

### "Model not found" Error

**Problem:** Server can't find the model files.

**Solution:**
1. Verify the directory path is correct and absolute
2. Check that the directory contains `models/buffalo_l/` subdirectories
3. Ensure all .onnx files are present

### "Permission denied" Error

**Problem:** Server can't access the model directory.

**Solution:**
```bash
# Make directory readable
chmod -R 755 /path/to/face_models

# Or move to a location with proper permissions
```

### Models Re-downloading

**Problem:** Models download every time despite being present.

**Solution:**
1. Ensure `face_models_dir` is set to the correct path
2. Use absolute paths, not relative paths
3. Verify directory structure matches InsightFace expectations

## Offline Usage

For completely offline environments:

1. **On a machine with internet:**
   ```bash
   python download_face_models.py ./offline_models
   tar -czf face_models.tar.gz offline_models/
   ```

2. **On the offline machine:**
   ```bash
   tar -xzf face_models.tar.gz
   # Configure server to use ./offline_models
   ```

## Performance Notes

- **First Detection:** 2-5 seconds (model loading)
- **Subsequent Detections:** <1 second per image
- **GPU vs CPU:** GPU is 5-10x faster but requires onnxruntime-gpu

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `face_models_dir` | string/null | null | Absolute path to models directory |

**Get current config:**
```bash
curl http://localhost:8000/api/config | jq '.face_models_dir'
```

**Set config:**
```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"face_models_dir": "/path/to/models"}'
```

**Reset to default:**
```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"face_models_dir": null}'
```
