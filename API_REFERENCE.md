# API Reference

Complete API documentation for the AI Server with detailed request/response examples.

## Base URL

```
http://localhost:8000
```

## Important Notes

### LLM Backend Behavior

The `llm_mode` (or `backend`) configuration controls which LLM backend is used for **all** LLM tasks:

- **`server` mode**: Uses `llama-server` (persistent process) for chat, embeddings, **and vision tasks**
  - Vision tasks send base64-encoded images to llama-server with multimodal message format
  - Better performance with persistent process
  
- **`cli` mode**: Uses specialized binaries per-request for chat, embeddings, **and vision tasks**
  - Chat/embeddings use `llama-cli`
  - Vision tasks use `llama-mtmd-cli` with image file input
  - Lower memory footprint

All LLM parameters (ctx_size, temp, top_p, top_k, presence_penalty, mirostat, batch_size, ubatch_size) are applied to all LLM operations including vision tasks.

### File Paths

For `/api/tag` and `/api/describe` endpoints, the `file_paths` array should contain **filenames only** (not absolute paths). The server resolves full paths using the configured storage metadata directory.

**Example**: `["image1.jpg", "image2.png"]` not `["/full/path/to/image1.jpg"]`

### File Storage Structure

**All photos and videos must be located in a `files` subdirectory** at the same level as the `storage_metadata.json` file:

```
/path/to/your/data/
├── storage_metadata.json    # Metadata file
└── files/                   # All media files go here
    ├── image1.jpg
    ├── video1.mp4
    ├── image2.png
    └── ...
```

The `fileName` field in `storage_metadata.json` should contain only the filename (e.g., `"image1.jpg"`), and the server will automatically look for it in the `files/` subdirectory.

## Endpoints Overview

| Endpoint | Type | Description |
|----------|------|-------------|
| `/api/config` | GET | Get current configuration |
| `/api/config` | POST | Update configuration |
| `/api/available-models` | GET | Get available models filtered by task type |
| `/api/set-storage-metadata` | POST | Set metadata file path |
| `/api/load-rag` | POST | Load RAG database |
| `/api/kill` | POST | Shutdown server and all processes |
| `/api/detect-faces` | POST | Detect and identify faces in images |
| `/api/get-face-crop` | POST | Extract face crop by face ID |
| `/api/rename-face-id` | POST | Rename a face ID in the database |
| `/api/vector-embeddings` | WebSocket | Generate or regenerate embeddings for files |
| `/api/generate-rag` | WebSocket | Build RAG database |
| `/api/tag` | WebSocket | Generate tags for media |
| `/api/describe` | WebSocket | Generate descriptions for media |
| `/api/chat` | WebSocket | Chat with RAG context |

---

## REST API

### GET /api/config

Get the current server configuration, including editable and read-only settings.

**Request:**
```bash
curl http://localhost:8000/api/config
```

**Response:** `200 OK`
```json
{
  "reduced_embedding_size": null,
  "chat_rounds": 3,
  "image_quality": 1.0,
  "llm_mode": "server",
  "top_k": 5,
  "recency_bias": 1.0,
  "enable_visual_chat": true,
  "chat_model": "Qwen3-0.6B-Q4_K_M.gguf",
  "embedding_model": "embeddinggemma-300M-Q8_0.gguf",
  "vision_model": "gemma-3-4b-it-UD-IQ1_S.gguf",
  "mmproj_model": "gemma_3_mmproj-F16.gguf",
  "chat_system_prompt": "You are Persona, a helpful AI assistant. Provide concise, factual answers...",
  "tag_prompt": "Analyze this image and generate descriptive tags...",
  "describe_prompt": "Describe this image in detail...",
  "vision_binary": "auto",
  "backend": "server",
  "model_timeout": 300,
  "llm_timeout": 300,
  "llm_params": {
    "ctx_size": 12192,
    "temp": 0.35,
    "top_p": 0.9,
    "top_k": 40,
    "presence_penalty": 0.2,
    "mirostat": 0,
    "batch_size": 1024,
    "ubatch_size": 512,
    "n_gpu_layers": 999
  },
  "rag_directory_name": "rag",
  "storage_metadata_path": null
}
```

**Field Descriptions:**

| Field | Type | Editable | Description |
|-------|------|----------|-------------|
| `reduced_embedding_size` | int/null | ✅ | Target dimension for PCA reduction (null = no reduction) |
| `chat_rounds` | int | ✅ | Number of conversation rounds to maintain (1-10) |
| `image_quality` | float | ✅ | Image scale multiplier (0.0-1.0): 1.0 = original dimensions, <1.0 = scale down (e.g., 0.5 = half size) |
| `llm_mode` | string | ✅ | LLM backend: `server` (persistent) or `cli` (per-request) |
| `top_k` | int | ✅ | Number of RAG results to retrieve (1-50) |
| `recency_bias` | float | ✅ | Recency weight in search (≥0.1, where 1.0 = no bias, >1.0 = favor recent) |
| `enable_visual_chat` | bool | ✅ | Enable visual conversation mode (uses vision model for chat with images) |
| `chat_model` | string | ✅ | Chat model filename |
| `embedding_model` | string | ✅ | Embedding model filename |
| `vision_model` | string | ✅ | Vision model filename |
| `mmproj_model` | string | ✅ | MMProj model filename for vision |
| `chat_system_prompt` | string | ✅ | System prompt for chat conversations (XML format with `<think>`, `<conclusion>`, `<files>` tags) |
| `tag_prompt` | string | ✅ | Prompt template for generating tags (XML format with `<think>` and `<conclusion>` tags) |
| `describe_prompt` | string | ✅ | Prompt template for generating descriptions (XML format with `<think>` and `<conclusion>` tags) |
| `vision_binary` | string | ✅ | Override vision binary: "auto" (default, auto-detect), "llama-mtmd-cli", or "llama-qwen2vl-cli" |
| `backend` | string | ✅ | Same as `llm_mode` |
| `model_timeout` | int | ✅ | Seconds before unloading inactive model |
| `llm_timeout` | int | ✅ | Timeout for LLM operations in seconds (10-3600) |
| `llm_params` | object | ✅ | LLM execution parameters |
| `rag_directory_name` | string | ❌ | RAG directory name (read-only) |
| `storage_metadata_path` | string/null | ❌ | Current metadata path (read-only) |

---

### POST /api/config

Update server configuration. Only editable fields can be changed.

**Request:**
```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "chat_model": "Qwen3-8B-Q4_K_M.gguf",
    "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf",
    "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
    "mmproj_model": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf",
    "chat_system_prompt": "You are Persona, a helpful AI assistant...",
    "tag_prompt": "List relevant tags for this image...",
    "describe_prompt": "Describe this image in detail...",
    "reduced_embedding_size": 512,
    "image_quality": 0.75,
    "top_k": 10,
    "recency_bias": 0.5,
    "backend": "server",
    "llm_params": {
      "ctx_size": 8192,
      "temp": 0.7
    }
  }'
```

**Response:** `200 OK` - Same structure as GET /config with updated values

**Error Response:** `422 Unprocessable Entity`
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "chat_rounds"],
      "msg": "Value must be between 1 and 10",
      "input": 15
    }
  ]
}
```

---

### GET /api/available-models

Get a list of available models filtered by task type. This endpoint checks which models actually exist in the model folder and returns their availability status.

**Query Parameters:**
- `task_type` (optional): Filter models by task type. Valid values: `vision`, `chat`, `embedding`
  - If not specified, returns all models

**Request:**
```bash
# Get all available models
curl http://localhost:8000/api/available-models

# Get only vision models
curl "http://localhost:8000/api/available-models?task_type=vision"

# Get only chat models
curl "http://localhost:8000/api/available-models?task_type=chat"

# Get only embedding models
curl "http://localhost:8000/api/available-models?task_type=embedding"
```

**Response:** `200 OK`
```json
{
  "models": [
    {
      "name": "Gemma 3 4B Vision",
      "type": "vision",
      "model_file": "gemma-3-4b-it-Q4_K_M.gguf",
      "model_exists": true,
      "mmproj_file": "gemma_3_mmproj-F16.gguf",
      "mmproj_exists": true,
      "llm_params": {
        "temperature": 0.7,
        "top_p": 0.9
      }
    },
    {
      "name": "Qwen3 8B Chat",
      "type": "chat",
      "model_file": "Qwen3-8B-Q4_K_M.gguf",
      "model_exists": true,
      "mmproj_file": null,
      "mmproj_exists": null,
      "llm_params": null
    },
    {
      "name": "Qwen3 Embedding 8B",
      "type": "embedding",
      "model_file": "qwen3-embedding-8b-q4_k_m.gguf",
      "model_exists": true,
      "mmproj_file": null,
      "mmproj_exists": null,
      "llm_params": null
    }
  ],
  "total_count": 3,
  "task_type": null
}
```

**Response Fields:**
- `models`: Array of model information objects
  - `name`: Human-readable model name/identifier
  - `type`: Task type (`vision`, `chat`, or `embedding`)
  - `model_file`: Model filename
  - `model_exists`: Boolean indicating if the model file exists in the model folder
  - `mmproj_file`: Path to multimodal projector file (null if not applicable)
  - `mmproj_exists`: Boolean indicating if the mmproj file exists (null if not applicable)
  - `llm_params`: Optional model-specific LLM parameters (null if not defined)
- `total_count`: Total number of models returned
- `task_type`: The task type filter applied (null if no filter)

**Error Responses:**

`400 Bad Request` - Invalid task type:
```json
{
  "detail": "Invalid task_type 'invalid_type'. Must be one of: vision, chat, embedding"
}
```

**Use Cases:**
- Check which models are currently available before making requests
- Display available models to users in a UI
- Validate configuration before starting long-running tasks
- Debug model file installation issues

---

### POST /api/set-storage-metadata

Set the path to the storage metadata JSON file. This file contains information about all files to be indexed. Must be called before using other endpoints.

**Request:**
```bash
curl -X POST http://localhost:8000/api/set-storage-metadata \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/Users/username/data/storage-metadata.json"
  }'
```

**Response:** `200 OK`
```json
{
  "status": "success",
  "message": "Storage metadata set to /Users/username/data/storage-metadata.json",
  "data": {
    "metadata_count": 156,
    "rag_directory": "/Users/username/data/rag",
    "embeddings_loaded": true,
    "embeddings_count": 156
  }
}
```

**Error Responses:**

`404 Not Found` - File doesn't exist:
```json
{
  "detail": "File not found: /path/to/file.json"
}
```

`400 Bad Request` - Not a file:
```json
{
  "detail": "Not a file: /path/to/directory"
}
```

---

### POST /api/load-rag

Load an existing RAG database from disk. The RAG directory is determined from the storage metadata path.

**Request:**
```bash
curl -X POST http://localhost:8000/api/load-rag
```

**Response:** `200 OK`
```json
{
  "status": "success",
  "message": "RAG database loaded successfully"
}
```

**Error Response:** `400 Bad Request`
```json
{
  "detail": "Storage metadata not set. Call /set-storage-metadata first."
}
```

---
---
### POST /api/kill

Shutdown the server and terminate all associated processes including llama-server, llama-cli, and the Python application.

**Request:**
```bash
curl -X POST http://localhost:8000/api/kill
```

**Response:** `200 OK`
```json
{
  "status": "success",
  "message": "Server shutdown initiated. All processes will be terminated."
}
```

**Behavior:**
1. Unloads all active models (LLM, embedding, vision)
2. Kills all llama-server and llama-cli processes
3. Sends response to client
4. Terminates the Python application after 0.5 seconds

**Note:** This endpoint is useful for clean shutdown when running the server as a background service or in automated scripts.

**Error Response:** `200 OK` (with error status)
```json
{
  "status": "error",
  "message": "Error during shutdown: <error details>"
}
```

---

### POST /api/detect-faces

Detect and identify faces in images using InsightFace. Automatically matches faces against stored embeddings or creates new face IDs.

**Request:**
```bash
curl -X POST http://localhost:8000/api/detect-faces \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["family_photo.jpg", "birthday_party.jpg"],
    "similarity_threshold": 0.5
  }'
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file_paths` | array[string] | ✅ | - | Filenames to process (not full paths) |
| `similarity_threshold` | float | ❌ | 0.5 | Minimum cosine similarity for face matching (0.0-1.0). Recommended: 0.4 (loose), 0.5 (balanced), 0.6 (strict) |

**Response:** `200 OK`
```json
{
  "status": "success",
  "results": [
    {
      "filename": "family_photo.jpg",
      "faces": [
        {
          "face_id": "face_001",
          "bbox": [120, 85, 230, 195],
          "similarity": 0.92,
          "is_new": false
        },
        {
          "face_id": "face_002",
          "bbox": [340, 90, 450, 200],
          "similarity": 0.78,
          "is_new": false
        },
        {
          "face_id": "face_003",
          "bbox": [550, 100, 660, 210],
          "similarity": null,
          "is_new": true
        }
      ]
    },
    {
      "filename": "birthday_party.jpg",
      "faces": [
        {
          "face_id": "face_001",
          "bbox": [200, 150, 310, 260],
          "similarity": 0.89,
          "is_new": false
        }
      ]
    }
  ]
}
```

**Field Descriptions:**

| Field | Description |
|-------|-------------|
| `face_id` | Unique identifier for the face |
| `bbox` | Bounding box [x, y, x+w, y+h] in pixels |
| `similarity` | Cosine similarity score (0.0-1.0) for matched faces, null for new faces |
| `is_new` | True if this is a newly detected face, false if matched to existing |

**Error Response:** `400 Bad Request`
```json
{
  "detail": "Storage metadata not set. Call /set-storage-metadata first."
}
```

**Error Response:** `500 Internal Server Error`
```json
{
  "detail": "Face detection error: <error details>"
}
```

**Notes:**
- Face embeddings are stored in `{rag_directory}/face_embeddings.pkl`
- Uses buffalo_l model from InsightFace (512-dimensional embeddings)
- Lower threshold (e.g., 0.4) increases false positives but finds more matches
- Higher threshold (e.g., 0.6) reduces false positives but may miss valid matches
- Default 0.5 provides balanced accuracy
- **Multiple embeddings per person**: Each time a face is detected and matched to an existing face ID, a new embedding is automatically added to that person's collection. This improves recognition accuracy over time as it captures different angles, expressions, and lighting conditions.
- The system maintains a list of embeddings for each face ID, allowing better matching against various appearances of the same person

---

### POST /api/get-face-crop

Extract a cropped image of a specific face by face ID from an image.

**Request:**
```bash
curl -X POST http://localhost:8000/api/get-face-crop \
  -H "Content-Type: application/json" \
  -d '{
    "image_name": "family_photo.jpg",
    "face_id": "face_001",
    "padding": 20,
    "min_similarity": 0.4
  }'
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image_name` | string | ✅ | - | Filename (not full path) |
| `face_id` | string | ✅ | - | Face ID to extract |
| `padding` | int | ❌ | 20 | Pixels to add around face bbox |
| `min_similarity` | float | ❌ | 0.4 | Minimum similarity threshold for matching the face |
| `min_similarity` | float | ❌ | 0.4 | Minimum similarity threshold |

**Response:** `200 OK`
```json
{
  "face_crop_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "face_id": "face_001",
  "image_name": "family_photo.jpg"
}
```

**Field Descriptions:**

| Field | Description |
|-------|-------------|
| `face_crop_base64` | Base64-encoded JPEG image of the face crop with padding |
| `face_id` | The face ID that was requested |
| `image_name` | The image filename |

**Error Responses:**

`400 Bad Request` - Storage not set:
```json
{
  "detail": "Storage metadata not set. Call /set-storage-metadata first."
}
```

`404 Not Found` - Face not found:
```json
{
  "detail": "Face 'face_999' not found in 'family_photo.jpg'"
}
```

`500 Internal Server Error` - Processing error:
```json
{
  "detail": "Face crop error: <error details>"
}
```

**Usage Example:**
```python
import requests
import base64
from PIL import Image
import io

response = requests.post(
    "http://localhost:8000/api/get-face-crop",
    json={
        "image_name": "family_photo.jpg",
        "face_id": "face_001",
        "padding": 30
    }
)

if response.status_code == 200:
    data = response.json()
    # Decode base64 image
    img_data = base64.b64decode(data['face_crop_base64'])
    img = Image.open(io.BytesIO(img_data))
    img.save(f"{data['face_id']}.jpg")
```

---

### POST /api/rename-face-id

Rename a face ID in the face embeddings database. This updates all references to the face ID.

**Important:** If the new face ID already exists, all embeddings from both face IDs will be merged. This allows:
- Consolidating duplicate face IDs that represent the same person
- Building a richer embedding set for better recognition (multiple angles, expressions, lighting)
- Correcting misidentifications from model imperfections

**Request:**
```bash
curl -X POST http://localhost:8000/api/rename-face-id \
  -H "Content-Type: application/json" \
  -d '{
    "old_face_id": "face_001",
    "new_face_id": "john_doe"
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `old_face_id` | string | ✅ | Current face ID to rename |
| `new_face_id` | string | ✅ | New face ID name |

**Response:** `200 OK`
```json
{
  "status": "success",
  "message": "Face ID renamed from face_001 to john_doe",
  "old_face_id": "face_001",
  "new_face_id": "john_doe"
}
```

**Error Responses:**

`400 Bad Request` - Missing required fields:
```json
{
  "detail": "old_face_id and new_face_id are required"
}
```

`404 Not Found` - Old face ID not found:
```json
{
  "detail": "Face ID 'face_001' not found"
}
```

`500 Internal Server Error` - Processing error:
```json
{
  "detail": "<error details>"
}
```

**Usage Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/rename-face-id",
    json={
        "old_face_id": "face_001",
        "new_face_id": "john_doe"
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Successfully renamed: {data['message']}")
else:
    print(f"Error: {response.json()['detail']}")
```

**Notes:**
- The new face ID can be any string (e.g., person names, custom identifiers)
- The rename operation updates the face embeddings file immediately
- All future face detections will use the new face ID when matching this person
- If the new face ID already exists, all embeddings are merged (not discarded), creating a richer embedding set
- Multiple embeddings per person improve recognition accuracy across different angles, expressions, and lighting
- Each time a face is detected and matched to an existing face ID, a new embedding is added to that person's collection
- This behavior allows handling cases where the model incorrectly assigned the same ID to two different people, and also builds stronger recognition profiles over time

---

## WebSocket API

All WebSocket endpoints follow a consistent message format:

```typescript
interface WebSocketMessage {
  type: "status" | "progress" | "result" | "error" | "confirmation_needed" | "thinking" | "conclusion" | "files";
  message: string;
  data?: any;
}
```

**Message Types:**
- `status` - General status updates
- `progress` - Progress updates with partial data
- `result` - Final result or completion
- `error` - Error messages
- `confirmation_needed` - Requires user confirmation to continue
- `thinking` - Model's analysis/reasoning process (tag and describe only; sanitized in chat)
- `conclusion` - Final answer (chat only)
- `files` - Relevant files section (chat only)

### Connection Example

```javascript
const ws = new WebSocket('ws://localhost:8000/api/generate-embeddings');

ws.onopen = () => {
  ws.send(JSON.stringify({
    embedding_model: "qwen3-embedding-8b-q4_k_m.gguf"
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.type, msg.message, msg.data);
};
```

---

### WS /api/vector-embeddings

Generate or regenerate vector embeddings for files in the storage metadata. This unified endpoint handles both initial embedding generation and selective regeneration. Uses the specified embedding model to create numerical representations of file content.

**Automatic Metadata Reload:** Always reloads `storage-metadata.json` if it has been modified since last load, ensuring embeddings reflect the latest metadata.

**Three Modes of Operation:**
1. **New files only** (default): Generate embeddings only for files without existing embeddings
2. **Specific files**: Regenerate embeddings for a list of filenames
3. **All files**: Regenerate embeddings for all files in storage metadata

**Connection:** `ws://localhost:8000/api/vector-embeddings`

**1. Client Connects and Sends Configuration:**

**Generate for new files only (default):**
```json
{
  "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf"
}
```

**Regenerate specific files:**
```json
{
  "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf",
  "file_names": ["image1.jpg", "image2.jpg", "video1.mp4"]
}
```

**Regenerate all files:**
```json
{
  "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf",
  "regenerate_all": true
}
```

**Configuration Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `embedding_model` | string | ❌ | From config | Embedding model filename |
| `file_names` | array[string] | ❌ | null | List of specific filenames to process. If provided, only these files are processed |
| `regenerate_all` | boolean | ❌ | false | If true, regenerate embeddings for all files. If true, `file_names` is ignored |

**2. Server Responds with Status:**

**New files mode:**
```json
{
  "type": "status",
  "message": "Found 100 existing embeddings, processing 56 new file(s)..."
}
```

**Specific files mode:**
```json
{
  "type": "status",
  "message": "Processing 3 specific file(s)..."
}
```

**Regenerate all mode:**
```json
{
  "type": "status",
  "message": "Regenerating embeddings for all 156 file(s)..."
}
```

**3. Server Sends Progress Updates:**
```json
{
  "type": "progress",
  "message": "Processing vacation/beach.jpg",
  "data": {
    "current": 15,
    "total": 56,
    "filename": "vacation/beach.jpg"
  }
}
```

**4. Server May Send Status Messages for Connection Issues:**

If the LLM server disconnects during processing, the system automatically restarts it and retries:

```json
{
  "type": "status",
  "message": "Server disconnected, restarting model (attempt 1/2)..."
}
```

```json
{
  "type": "status",
  "message": "Model restarted successfully, retrying..."
}
```

**5. Server May Send Error Messages for Individual File Failures:**

If a file fails during processing (even after retries), the server sends an error message but continues processing remaining files:

```json
{
  "type": "error",
  "message": "Failed to generate embedding for corrupted.jpg: Invalid file format",
  "data": {
    "filename": "corrupted.jpg",
    "error": "Invalid file format",
    "continue": true
  }
}
```

**Note:** The system automatically handles server disconnections by restarting the model and retrying the failed file up to 2 times before reporting an error.

**6. Server Sends Final Result:**

**Success - all files processed:**
```json
{
  "type": "result",
  "message": "Embeddings generated successfully",
  "data": {
    "count": 156,
    "processed": 56,
    "successful": 56,
    "failed": 0
  }
}
```

**Partial success - some files failed:**
```json
{
  "type": "result",
  "message": "Embeddings generated with 2 failure(s)",
  "data": {
    "count": 154,
    "processed": 56,
    "successful": 54,
    "failed": 2,
    "failed_files": [
      {
        "filename": "corrupted.jpg",
        "error": "Invalid file format"
      },
      {
        "filename": "missing.mp4",
        "error": "File not accessible"
      }
    ]
  }
}
```

**7. Connection Closes**

**Special Cases:**

All files already have embeddings (new files mode):
```json
{
  "type": "status",
  "message": "All files already have embeddings"
}
```
```json
{
  "type": "result",
  "message": "Embeddings generated successfully",
  "data": {
    "count": 156,
    "processed": 0
  }
}
```

Metadata file was updated:
```json
{
  "type": "status",
  "message": "Storage metadata file was updated. Reloaded metadata."
}
```

**Error Cases:**

File not found in metadata:
```json
{
  "type": "error",
  "message": "File not found in metadata: unknown.jpg"
}
```

Model not found:
```json
{
  "type": "error",
  "message": "Model file not found: model.gguf"
}
```

**Python Examples:**

**Generate for new files only:**
```python
import asyncio
import websockets
import json

async def generate_embeddings():
    async with websockets.connect('ws://localhost:8000/api/vector-embeddings') as ws:
        # Send configuration
        await ws.send(json.dumps({
            "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf"
        }))
        
        # Receive messages
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            
            if data['type'] == 'progress':
                print(f"  Progress: {data['data']['current']}/{data['data']['total']}")
            elif data['type'] in ['result', 'error']:
                break

asyncio.run(generate_embeddings())
```

**Regenerate specific files:**
```python
async def regenerate_files():
    async with websockets.connect('ws://localhost:8000/api/vector-embeddings') as ws:
        await ws.send(json.dumps({
            "file_names": ["image1.jpg", "video2.mp4"]
        }))
        
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            if data['type'] in ['result', 'error']:
                break

asyncio.run(regenerate_files())
```

**Regenerate all files:**
```python
async def regenerate_all():
    async with websockets.connect('ws://localhost:8000/api/vector-embeddings') as ws:
        await ws.send(json.dumps({
            "regenerate_all": True
        }))
        
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            if data['type'] in ['result', 'error']:
                break

asyncio.run(regenerate_all())
```

**Behavior:**
1. Reloads `storage-metadata.json` if it has been modified since last load
2. Determines which files to process based on parameters
3. Loads the embedding model (uses `embedding_model` from config)
4. Generates embeddings for selected files with progress updates
5. Updates embeddings.json file (preserves existing embeddings not being regenerated)
6. Applies PCA reduction if `reduced_embedding_size` is configured
7. Unloads the embedding model
8. Returns success with total count and processed count

**Use Cases:**
- **Initial setup**: Connect without parameters to generate embeddings for all new files
- **After metadata updates**: Use `file_names` to regenerate specific files you edited
- **Bulk regeneration**: Use `regenerate_all: true` after widespread metadata changes
- **Keeping up-to-date**: Run periodically without parameters to process newly added files

---

### WS /api/generate-rag

Build the RAG (Retrieval Augmented Generation) database from embeddings. Creates a FAISS index for fast similarity search and saves it to disk.

**Connection:** `ws://localhost:8000/api/generate-rag`

**1. Client Connects (No Initial Message Required)**

**2. Server Sends Status Updates:**
```json
{
  "type": "status",
  "message": "Building RAG database..."
}
```

```json
{
  "type": "status",
  "message": "Adding vectors to FAISS index..."
}
```

```json
{
  "type": "status",
  "message": "Saving RAG database to disk..."
}
```

**3. Server Sends Result:**
```json
{
  "type": "result",
  "message": "RAG database created and loaded successfully"
}
```

**4. Connection Closes**

**Error Case:**

No embeddings available:
```json
{
  "type": "error",
  "message": "No embeddings available. Generate embeddings first."
}
```

**Note:** The RAG database is saved to `{metadata_directory}/rag/` as:
- `faiss.index` - FAISS vector index
- `metadata.json` - File metadata mapping

---

### WS /api/tag

Generate AI tags for images and videos. Tags are automatically saved to the storage metadata file.

**Connection:** `ws://localhost:8000/api/tag`

**1. Client Connects and Sends Request:**
```json
{
  "file_paths": [
    "beach_sunset.jpg",
    "birthday_party.mp4",
    "mountain_hike.jpg"
  ],
  "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
  "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
}
```

**Note:** `file_paths` should contain only the filenames as they appear in your metadata file, not absolute paths. The server will resolve the full paths automatically using the storage metadata.

**2. Server Processes First File Automatically:**
```json
{
  "type": "status",
  "message": "Generating tags for beach_sunset.jpg..."
}
```

**3. Server Sends Image Info (if available):**
```json
{
  "type": "status",
  "message": "Image Info - Original: 4032x3024, Processed: 2016x1512, Size: 324.5 KB",
  "data": {
    "original_width": 4032,
    "original_height": 3024,
    "processed_width": 2016,
    "processed_height": 1512,
    "size_bytes": 332288
  }
}
```

**4. Server Sends Thinking Process:**
```json
{
  "type": "thinking",
  "message": "Analysis for beach_sunset.jpg",
  "data": {
    "filename": "beach_sunset.jpg",
    "thinking": "I can see a beautiful coastal scene with several elements. The main focus is the sunset with vibrant colors in the sky. The ocean is visible with waves, and there are palm trees framing the composition. The lighting suggests golden hour photography."
  }
}
```

**5. Server Returns Tags:**
```json
{
  "type": "result",
  "message": "Tags generated for beach_sunset.jpg",
  "data": {
    "filename": "beach_sunset.jpg",
    "tags": ["beach", "sunset", "ocean", "vacation", "nature", "sky", "water", "palm trees", "golden hour"]
  }
}
```

**6. Server Asks for Confirmation (Subsequent Files):**
```json
{
  "type": "confirmation_needed",
  "message": "Ready to tag birthday_party.mp4. Send 'continue' to proceed.",
  "data": {
    "current": 2,
    "total": 3
  }
}
```

**7. Client Confirms:**
```json
{
  "action": "continue"
}
```

**8. Repeat Steps 2-7 for Each File**

**9. Server Sends Completion:**
```json
{
  "type": "status",
  "message": "Tagging complete"
}
```

**10. Connection Closes**

**Error Cases:**

File not in metadata:
```json
{
  "type": "error",
  "message": "Metadata not found for unknown_file.jpg"
}
```

File doesn't exist:
```json
{
  "type": "error",
  "message": "File not found: /path/to/file.jpg"
}
```

Vision processing error (with detailed logging):
```json
{
  "type": "error",
  "message": "Failed to generate tags for beach.jpg: RuntimeError: Vision model failed",
  "data": {
    "filename": "beach.jpg",
    "error_type": "RuntimeError",
    "error_message": "Vision model failed",
    "traceback": "Full Python traceback...",
    "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
    "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf",
    "file_type": "image",
    "file_exists": true,
    "file_size": 2458734
  }
}
```

---

### WS /api/describe

Generate detailed AI descriptions for images and videos. Descriptions are automatically saved to the storage metadata file.

**Connection:** `ws://localhost:8000/api/describe`

**Message Flow:** Similar to `/tag` endpoint

**1. Client Request:**
```json
{
  "file_paths": [
    "beach_sunset.jpg",
    "concert.mp4"
  ],
  "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
  "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
}
```

**Note:** `file_paths` should contain only the filenames as they appear in your metadata file, not absolute paths.

**2. Server Status:**
```json
{
  "type": "status",
  "message": "Generating description for beach_sunset.jpg..."
}
```

**3. Server Sends Image Info (if available):**
```json
{
  "type": "status",
  "message": "Image Info - Original: 4032x3024, Processed: 2016x1512, Size: 324.5 KB",
  "data": {
    "original_width": 4032,
    "original_height": 3024,
    "processed_width": 2016,
    "processed_height": 1512,
    "size_bytes": 332288
  }
}
```

**4. Server Sends Thinking Process:**
```json
{
  "type": "thinking",
  "message": "Analysis for beach_sunset.jpg",
  "data": {
    "filename": "beach_sunset.jpg",
    "thinking": "Let me examine the key elements in this image. I can identify a coastal landscape during sunset with distinctive features that should be described in detail."
  }
}
```

**5. Server Returns Description:**
```json
{
  "type": "result",
  "message": "Description generated for beach_sunset.jpg",
  "data": {
    "filename": "beach_sunset.jpg",
    "description": "A breathtaking sunset over the ocean with vibrant orange and pink hues painting the sky. The sun sits just above the horizon, casting a golden reflection on the calm water. Silhouettes of palm trees frame the scene on the left side, creating a classic tropical composition."
  }
}
```

**6. Confirmation (for subsequent files):**
```json
{
  "type": "confirmation_needed",
  "message": "Ready to describe concert.mp4. Send 'continue' to proceed.",
  "data": {
    "current": 2,
    "total": 2
  }
}
```

**7. Client Confirms:**
```json
{
  "action": "continue"
}
```

**8. Repeat Steps 2-7 for Each File**

**9. Completion:**
```json
{
  "type": "status",
  "message": "Description generation complete"
}
```

**10. Connection Closes**

---

### WS /api/chat

Interactive chat with RAG context. Uses the RAG database to provide context-aware responses about your files. The server automatically selects the appropriate models based on configuration.

**Important:** Each WebSocket connection handles a **single request-response cycle**. The connection automatically closes after the response is complete. For follow-up questions, initiate a new WebSocket connection and provide the conversation history via the `history` parameter.

**Connection:** `ws://localhost:8000/api/chat`

**1. Client Connects (No Initial Configuration Required)**

The server automatically uses the models configured in `/api/config`:
- `chat_model` for text conversations
- `vision_model` + `mmproj_model` for visual conversations (if `enable_visual_chat` is true)
- `embedding_model` for RAG search

**2. Server Loads Models:**
```json
{
  "type": "status",
  "message": "Loading RAG database..."
}
```

```json
{
  "type": "status",
  "message": "Loading chat model Qwen3-8B-Q4_K_M.gguf..."
}
```

```json
{
  "type": "status",
  "message": "Chat ready. Send your message."
}
```

**3. Client Sends Message:**

**First Message (No History):**
```json
{
  "message": "What beach photos do I have?"
}
```

**Follow-up Message (With History from Previous Connection):**
```json
{
  "message": "Show me the sunset ones",
  "history": [
    {
      "role": "user",
      "content": "What beach photos do I have?"
    },
    {
      "role": "assistant",
      "content": "You have 12 beach photos in your collection..."
    }
  ]
}
```

**Message Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | ✅ | The current user message to process |
| `history` | array | ❌ | Optional chat history in OpenAI format. If provided, the server uses this history instead of its internal conversation state. Each item must have `role` ("user" or "assistant") and `content` fields |
| `image_name` | string | ❌ | Optional image filename for visual conversations (when `enable_visual_chat` is true). If provided with `enable_visual_chat: true`, the vision model is used instead of the chat model |

**Note:** 
- **One message per connection**: Each WebSocket connection handles exactly one request-response cycle and then closes automatically
- **Multi-turn conversations**: For follow-up questions, open a new connection and provide the previous conversation via the `history` parameter
- The `history` parameter is **required** for maintaining context across multiple connections
- **RAG Search with History**: When searching the knowledge base, the system includes the conversation history in the embedding vector. The search query is formatted as: `user: "query1", assistant: "response1", user: "query2", ...`, ending with the latest user message. This provides better context-aware retrieval.

**4. Server Searches and Generates:**
```json
{
  "type": "status",
  "message": "Searching knowledge base..."
}
```

```json
{
  "type": "status",
  "message": "Loading chat model Qwen3-8B-Q4_K_M.gguf..."
}
```

```json
{
  "type": "status",
  "message": "Generating response..."
}
```

**5. Server Streams Response with Progress Messages:**

During response generation, the server sends real-time progress updates as the model generates text:

```json
{
  "type": "progress",
  "message": "Let me",
  "data": {
    "partial_response": "Let me"
  }
}
```

```json
{
  "type": "progress",
  "message": " search through",
  "data": {
    "partial_response": "Let me search through"
  }
}
```

```json
{
  "type": "progress",
  "message": " your photo collection",
  "data": {
    "partial_response": "Let me search through your photo collection"
  }
}
```

**Note:** Progress messages contain text chunks as they are generated, allowing clients to display streaming responses. The `message` field contains the new chunk, while `data.partial_response` contains the accumulated response so far.

**6. Server Returns Structured Response:**

After streaming completes, the chat endpoint returns responses in structured sections. The server automatically sanitizes the response to remove internal reasoning (`<think>` tags) and returns only the user-facing content:

**Conclusion Section:**
```json
{
  "type": "conclusion",
  "message": "You have 12 beach photos in your collection. These include beautiful sunset scenes from your summer vacation, family gatherings at the beach, surfing activities, and coastal landscapes. The photos were taken between June and August 2024, mostly at Santa Monica and Malibu beaches."
}
```

**Files Section:**
```json
{
  "type": "files",
  "message": "vacation/beach_sunset.jpg, summer/surfing_day.jpg, family/beach_picnic.jpg",
  "data": {
    "relevant_files": [
      "vacation/beach_sunset.jpg",
      "summer/surfing_day.jpg",
      "family/beach_picnic.jpg"
    ]
  }
}
```

**Result (Completion):**
```json
{
  "type": "result",
  "message": "Response complete"
}
```

**Note:** If the LLM doesn't use the structured format (no XML tags), the response falls back to a single `result` message with the full response.

**7. Connection Closes Automatically**

After sending the result, the WebSocket connection closes automatically. For follow-up questions, create a new connection and include the conversation history.

**Follow-up Question Example:**

To ask a follow-up question, open a new WebSocket connection:

```json
{
  "message": "Show me the ones with sunsets",
  "history": [
    {
      "role": "user",
      "content": "What beach photos do I have?"
    },
    {
      "role": "assistant",
      "content": "<think>...</think><conclusion>You have 12 beach photos...</conclusion><files>...</files>"
    }
  ]
}
```

The server will respond with the same message flow (steps 4-6), then close the connection automatically.

**Error Cases:**

RAG not available:
```json
{
  "type": "error",
  "message": "RAG not available. Generate RAG first."
}
```

Model not found:
```json
{
  "type": "error",
  "message": "Model file not found: chat.gguf"
}
```

Invalid history format:
```json
{
  "type": "error",
  "message": "history parameter must be a list of message objects"
}
```

Invalid history structure:
```json
{
  "type": "error",
  "message": "Each history item must be a dict with 'role' and 'content' keys"
}
```

Invalid history role:
```json
{
  "type": "error",
  "message": "History role must be 'user' or 'assistant'"
}
```

**Features:**
- **One message per connection**: Each WebSocket connection handles a single request-response cycle
- **Client-managed history**: Client must provide conversation history via `history` parameter for multi-turn conversations
- **Automatic disconnection**: Connection closes automatically after response is complete
- **Context-aware RAG search**: Includes full conversation history in the embedding vector for better retrieval
- Uses RAG to provide context from your files
- Returns relevant files with similarity scores
- Supports both text and visual conversations (when `enable_visual_chat` is enabled)

---

## Complete Workflow Example

Here's a complete workflow from setup to chat:

```python
import asyncio
import websockets
import json
import requests

BASE_URL = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"

# 1. Set storage metadata
response = requests.post(f"{BASE_URL}/api/set-storage-metadata", json={
    "path": "/Users/username/data/metadata.json"
})
print(response.json())

# 2. Generate embeddings
async def generate_embeddings():
    async with websockets.connect(f"{WS_BASE}/api/vector-embeddings") as ws:
        await ws.send(json.dumps({
            "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf"
        }))
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            if data['type'] in ['result', 'error']:
                break

asyncio.run(generate_embeddings())

# 3. Build RAG database
async def build_rag():
    async with websockets.connect(f"{WS_BASE}/api/generate-rag") as ws:
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            if data['type'] in ['result', 'error']:
                break

asyncio.run(build_rag())

# 4. Generate tags for some files
async def tag_files():
    async with websockets.connect(f"{WS_BASE}/api/tag") as ws:
        await ws.send(json.dumps({
            "file_paths": ["beach_sunset.jpg", "vacation_photo.jpg"],
            "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
            "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
        }))
        
        file_count = 0
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            
            if data['type'] == 'thinking':
                print(f"  Thinking: {data['data']['thinking'][:100]}...")
            elif data['type'] == 'confirmation_needed':
                # Continue to next file
                await ws.send(json.dumps({"action": "continue"}))
            elif data['type'] == 'result':
                file_count += 1
                print(f"  Tags: {', '.join(data['data']['tags'])}")
            elif data['type'] == 'status' and 'complete' in data['message'].lower():
                break

asyncio.run(tag_files())

# 5. Chat with RAG (Single Message)
async def chat():
    async with websockets.connect(f"{WS_BASE}/api/chat") as ws:
        # Wait for ready
        while True:
            message = await ws.recv()
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            if "ready" in data['message'].lower():
                break
        
        # Send message
        await ws.send(json.dumps({
            "message": "What photos do I have from 2024?"
        }))
        
        # Get structured response
        thinking = ""
        conclusion = ""
        files = []
        
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            
            if data['type'] == 'thinking':
                thinking = data['message']
            elif data['type'] == 'conclusion':
                conclusion = data['message']
            elif data['type'] == 'files':
                files = data['data'].get('relevant_files', [])
            elif data['type'] == 'result':
                print(f"\nThinking: {thinking}")
                print(f"\nConclusion: {conclusion}")
                print(f"\nRelevant files: {', '.join(files)}")
                break
        
        # Connection closes automatically after result

asyncio.run(chat())

# 5b. Multi-turn Chat with History (Separate Connections)
async def chat_with_history():
    # First message - new connection
    first_response = ""
    async with websockets.connect(f"{WS_BASE}/api/chat") as ws:
        # Wait for ready
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        # Send first message
        await ws.send(json.dumps({
            "message": "What photos do I have from 2024?"
        }))
        
        # Collect response
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'progress':
                first_response += data['message']
            elif data['type'] == 'result':
                break
        # Connection closes automatically
    
    # Follow-up message - new connection with history
    async with websockets.connect(f"{WS_BASE}/api/chat") as ws:
        # Wait for ready
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        # Send second message with history
        conversation_history = [
            {
                "role": "user",
                "content": "What photos do I have from 2024?"
            },
            {
                "role": "assistant",
                "content": first_response
            }
        ]
        
        await ws.send(json.dumps({
            "message": "Show me the beach ones",
            "history": conversation_history
        }))
        
        # Get second response
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            if data['type'] == 'result':
                break
        # Connection closes automatically

asyncio.run(chat_with_history())

# 6. Detect faces (optional)
response = requests.post(f"{BASE_URL}/api/detect-faces", json={
    "file_paths": ["family_photo.jpg"],
    "similarity_threshold": 0.5
})
print("Face detection:", response.json())

# 7. Shutdown server (optional)
response = requests.post(f"{BASE_URL}/api/kill")
print(response.json())
```

---

## Error Handling

All endpoints follow consistent error handling:

**HTTP Errors:**
- `400 Bad Request` - Invalid input or precondition not met
- `404 Not Found` - Resource doesn't exist
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

**WebSocket Errors:**
All errors are sent as messages before closing:
```json
{
  "type": "error",
  "message": "Detailed error message"
}
```

---

## Rate Limits

Currently no rate limits are enforced. Future versions may add:
- Request rate limiting
- Concurrent WebSocket connection limits
- Model loading throttling

---

## Best Practices

1. **Always set metadata first:** Call `/set-storage-metadata` before other endpoints
2. **Check configuration:** Use `GET /config` to verify current settings
3. **Handle confirmations:** The `/tag` and `/describe` endpoints require confirmations for control
4. **Monitor progress:** WebSocket endpoints provide detailed progress updates with thinking and result messages
5. **Error recovery:** Check for `type: "error"` messages and handle appropriately
6. **Model management:** Models auto-unload after `model_timeout` seconds of inactivity
7. **Dimension reduction:** Use `reduced_embedding_size` to reduce memory usage by 87%
8. **Connection cleanup:** Always close WebSocket connections properly
9. **Face recognition:** Use similarity thresholds appropriately (0.4-0.6 range recommended)
10. **Structured output:** Parse XML tags (`<think>`, `<conclusion>`, `<files>`) from chat, tag, and describe responses

---

## Version

API Version: 2.0  
Documentation Last Updated: October 25, 2025
