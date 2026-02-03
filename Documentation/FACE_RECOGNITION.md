# Face Recognition Feature

This document describes the face recognition feature using InsightFace library.

## Overview

The face recognition system detects faces in images, creates unique identifiers for each person, and maintains a mapping of face IDs to face embeddings. It can identify the same person across multiple images.

## Installation

Install the required InsightFace library:

```bash
pip install insightface onnxruntime
```

For GPU support (optional):
```bash
pip install onnxruntime-gpu
```

### Pre-downloading Models

**Note:** Face models are now bundled with the application in the `model/` directory and will be automatically used. No additional configuration is needed.

If you need to use a custom model location or update the models:

#### Option 1: Use the Download Script

```bash
# Download to default location (./face_models)
python download_face_models.py

# Or specify a custom directory
python download_face_models.py /path/to/models
```

#### Option 2: Use Bundled Models (Default)

The InsightFace buffalo_l models are already included in the `model/models/buffalo_l/` directory and will be automatically used by the face recognition service.

#### Configure Custom Model Location (Optional)

If you want to use models from a different location:

**Via API:**
```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "face_models_dir": "/path/to/custom/models"
  }'
```

**Via config.py (before starting server):**
```python
from app.config import update_config

update_config(
    face_models_dir="/path/to/custom/models"
)
```

**Note:** 
- If `face_models_dir` is `null` or not set, the system automatically uses the bundled `model/` directory
- When building with PyInstaller, models in `model/` are automatically packaged with the executable

## Features

### 1. Face Detection and Identification

Detects all faces in images and assigns unique face IDs. If a face matches an existing person (based on embedding similarity), it uses the same face ID.

**Endpoint:** `POST /detect-faces`

**Request:**
```json
{
  "file_paths": ["photo1.jpg", "photo2.jpg"],
  "similarity_threshold": 0.6
}
```

**Response:**
```json
{
  "results": {
    "photo1.jpg": [
      {
        "face_id": "face_001",
        "bbox": [100, 150, 200, 250],
        "confidence": 0.99,
        "is_new": false
      },
      {
        "face_id": "face_002",
        "bbox": [400, 200, 180, 220],
        "confidence": 0.97,
        "is_new": true
      }
    ],
    "photo2.jpg": [
      {
        "face_id": "face_001",
        "bbox": [120, 180, 190, 240],
        "confidence": 0.98,
        "is_new": false
      }
    ]
  }
}
```

**Parameters:**
- `file_paths`: List of image filenames (must be in metadata store)
- `similarity_threshold`: Threshold for face matching (0.0-1.0, default: 0.6)
  - Higher values = stricter matching (more likely to create new face IDs)
  - Lower values = looser matching (more likely to reuse existing face IDs)

**Response Fields:**
- `face_id`: Unique identifier for the person (e.g., "face_001")
- `bbox`: Bounding box as [x, y, width, height]
- `confidence`: Detection confidence (0.0-1.0)
- `is_new`: Whether this is a newly registered face

### 2. Get Face Crop

Extracts and returns a cropped image of a specific face from an image.

**Endpoint:** `POST /get-face-crop`

**Request:**
```json
{
  "image_name": "photo.jpg",
  "face_id": "face_001",
  "padding": 20
}
```

**Response:**
```json
{
  "face_crop_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "face_id": "face_001",
  "image_name": "photo.jpg"
}
```

**Parameters:**
- `image_name`: Image filename (must be in metadata store)
- `face_id`: Face ID to extract
- `padding`: Pixels to add around face bbox (default: 20)

**Response Fields:**
- `face_crop_base64`: Base64-encoded JPEG image of the face
- `face_id`: Requested face ID
- `image_name`: Source image name

## Data Storage

### Face Embeddings File

Face embeddings are stored in: `{rag_directory}/face_embeddings.pkl`

The file contains a pickled dictionary mapping:
```python
{
  "face_001": numpy_array([...]),  # 512-dimensional embedding
  "face_002": numpy_array([...]),
  ...
}
```

### Location

The file is stored in the same RAG directory as other RAG data, determined by the metadata store location.

## How It Works

### Face Detection
1. Uses InsightFace's `buffalo_l` model for face detection
2. Extracts 512-dimensional face embeddings
3. Detection size: 640x640 pixels

### Face Matching
1. Computes cosine similarity between new face and all stored faces
2. If similarity > threshold, assigns existing face ID
3. If no match found, generates new face ID (face_XXX format)
4. Automatically saves new faces to the embeddings file

### Face Cropping
1. Re-detects faces in the source image
2. Finds the face matching the requested face ID (by embedding similarity)
3. Crops the face region with padding
4. Returns as JPEG with 95% quality

## Usage Examples

### Example 1: Detect Faces in Multiple Images

```python
import requests

response = requests.post(
    "http://localhost:8000/detect-faces",
    json={
        "file_paths": ["family_photo.jpg", "vacation.jpg"],
        "similarity_threshold": 0.6
    }
)

results = response.json()["results"]
for image_name, faces in results.items():
    print(f"\n{image_name}:")
    for face in faces:
        status = "NEW" if face["is_new"] else "EXISTING"
        print(f"  - {face['face_id']} [{status}] at {face['bbox']}")
```

### Example 2: Extract All Faces from an Image

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# First, detect faces
detect_response = requests.post(
    "http://localhost:8000/detect-faces",
    json={"file_paths": ["group_photo.jpg"]}
)

faces = detect_response.json()["results"]["group_photo.jpg"]

# Then, get crop for each face
for face in faces:
    crop_response = requests.post(
        "http://localhost:8000/get-face-crop",
        json={
            "image_name": "group_photo.jpg",
            "face_id": face["face_id"],
            "padding": 30
        }
    )
    
    # Decode and save the face crop
    face_data = base64.b64decode(crop_response.json()["face_crop_base64"])
    img = Image.open(BytesIO(face_data))
    img.save(f"{face['face_id']}.jpg")
    print(f"Saved {face['face_id']}.jpg")
```

### Example 3: Find Same Person Across Multiple Images

```python
import requests
from collections import defaultdict

# Detect faces in all images
response = requests.post(
    "http://localhost:8000/detect-faces",
    json={
        "file_paths": ["photo1.jpg", "photo2.jpg", "photo3.jpg"],
        "similarity_threshold": 0.65
    }
)

# Group images by face ID
face_appearances = defaultdict(list)
for image_name, faces in response.json()["results"].items():
    for face in faces:
        face_appearances[face["face_id"]].append(image_name)

# Print people who appear in multiple images
for face_id, images in face_appearances.items():
    if len(images) > 1:
        print(f"{face_id} appears in: {', '.join(images)}")
```

## Configuration

### Similarity Threshold

The `similarity_threshold` parameter controls face matching sensitivity:

- **0.4-0.5**: Very loose matching (may incorrectly match different people)
- **0.6**: Recommended default (good balance)
- **0.7-0.8**: Stricter matching (may create duplicate IDs for same person)
- **0.9+**: Very strict (will likely create many duplicates)

### Detection Size

The face detection size is set to 640x640. You can modify this in `face_service.py`:

```python
self.face_app.prepare(ctx_id=0, det_size=(640, 640))
```

Larger sizes = better detection but slower processing.

## Performance Considerations

- **CPU vs GPU**: By default uses CPU. For faster processing on machines with GPU, modify the providers in `face_service.py`:
  ```python
  providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
  ```

- **Model Loading**: The face analysis model is loaded on first use and kept in memory

- **Processing Speed**: ~1-2 seconds per image on CPU (varies by image size and number of faces)

## Limitations

1. Face embeddings are 512-dimensional vectors (not reduced)
2. No automatic cleanup of orphaned face IDs
3. Face matching is based purely on visual similarity (no name labels)
4. Requires images to be in the metadata store
5. Works best with frontal face images

## Troubleshooting

### ImportError: No module named 'insightface'
Install InsightFace: `pip install insightface onnxruntime`

### Model download errors
InsightFace downloads models on first use. Ensure internet connectivity or manually download the `buffalo_l` model.

### Low detection accuracy
- Increase detection size in the code
- Ensure images have good lighting and clear faces
- Try adjusting the similarity threshold

### Too many duplicate face IDs
Lower the similarity_threshold (e.g., from 0.6 to 0.5)

### Same person gets multiple IDs
Raise the similarity_threshold (e.g., from 0.6 to 0.7)
