# API Reference

Complete API documentation for the AI Server with detailed request/response examples.

## Base URL

```
http://localhost:8000
```

## Important Notes

### 🔥 Hot Reload - No Restart Required

The server supports **hot reload** - all configuration changes, model directory updates, and new model downloads take effect **immediately without restarting the server**:

- ✅ Configuration changes via `POST /api/config` are instant
- ✅ Model directory changes immediately affect model path resolution
- ✅ Downloaded models become available instantly
- ✅ No downtime or service interruption

See [HOT_RELOAD_GUIDE.md](HOT_RELOAD_GUIDE.md) for details.

### LLM Backend Behavior

The `llm_mode` (or `backend`) configuration controls which LLM backend is used for **all** LLM tasks:

- **`server` mode**: Uses `llama-server` (persistent process) for chat, embeddings, **and vision tasks**
  - Vision tasks send base64-encoded images to llama-server with multimodal message format
  - Better performance with persistent process
  
- **`cli` mode**: Uses specialized binaries per-request for chat, embeddings, **and vision tasks**
  - Chat uses `llama-cli`
  - Embeddings use `llama-embedding`
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
| `/api/model-options` | GET | Get all downloadable models (with repo_id configuration) |
| `/api/download-models` | WebSocket | Download models from Hugging Face repositories |
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
| `/api/deep-chat` | WebSocket | Multi-round thinking chat with RAG function access |
| `/api/cloud-chat` | WebSocket | Get RAG context for external cloud LLM |
| `/api/compact-conversations` | WebSocket | Compact (summarize) conversations for RAG memory |
| `/api/cloud-compact` | WebSocket | Embed client-provided conversation summaries for RAG (cloud AI workflow) |
| `/api/mcp` | WebSocket | Cloud AI Deep Chat — MCP tool calling endpoint (multi-message, persistent connection) |

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
    "n_gpu_layers": 999,
    "enable_thinking": true
  },
  "rag_directory_name": "rag",
  "storage_metadata_path": null,
  "enable_conversation_compaction": true,
  "max_compaction_tokens": 2000,
  "min_compaction_relevance": 0.4,
  "tool_history_max_tags": 7,
  "tool_history_max_results": 5,
  "max_tags_per_scope": 100,
  "max_dates_per_scope": 10
}
```

**Field Descriptions:**

| Field | Type | Editable | Description |
|-------|------|----------|-------------|
| `reduced_embedding_size` | int/null | ✅ | Target dimension for PCA reduction (null = no reduction) |
| `chat_rounds` | int | ✅ | Total MCP tool calls allowed per deep chat session (the tool budget). When 1 call remains, only `scoped_rag_search` is offered. At 0 calls the agent must produce a final answer. Default: 10 |
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
| `llm_params` | object | ✅ | LLM execution parameters: `ctx_size`, `temp`, `top_p`, `top_k`, `presence_penalty`, `mirostat`, `batch_size`, `ubatch_size`, `n_gpu_layers`, `enable_thinking` (bool, default `true` — enables thinking mode when deep chat loads the model) |
| `enable_conversation_compaction` | bool | ✅ | Enable conversation compaction (dreaming mechanism) for summarizing conversations into RAG |
| `max_compaction_tokens` | int | ✅ | Token budget for compacted conversation context injected into chat (100-8000) |
| `min_compaction_relevance` | float | ✅ | Minimum similarity score for compacted conversation retrieval (0.0-1.0) |
| `tool_history_max_tags` | int | ✅ | Number of tags retained when truncating `get_scoped_tags` results in tool call history (default: 7) |
| `tool_history_max_results` | int | ✅ | Number of results retained when truncating other MCP tool results in tool call history (default: 5) |
| `max_tags_per_scope` | int | ✅ | Maximum tags returned by `get_scoped_tags` per call (default: 100) |
| `max_dates_per_scope` | int | ✅ | Maximum date ranges returned by `get_scoped_dates` per call (default: 10) |
| `rag_directory_name` | string | ❌ | RAG directory name (read-only) |
| `storage_metadata_path` | string/null | ❌ | Current metadata path (read-only) |

---

### POST /api/config

Update server configuration. Only editable fields can be changed. **Changes take effect immediately without restarting the server** (hot reload).

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
    "model_directory": "/custom/path/to/models",
    "llm_params": {
      "ctx_size": 8192,
      "temp": 0.7
    }
  }'
```

**Editable Fields:**
- `model_directory`: Custom model directory path (absolute path). Set to null to use default saved_llm location. **Changes take effect immediately** - all subsequent model operations use the new directory.
- All other fields from GET /api/config marked as editable

**Hot Reload Behavior:**
- Config changes propagate immediately to all endpoints
- `model_directory` changes affect `/api/available-models` and model loading instantly
- No server restart required for any configuration change

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

Get a list of available models filtered by task type. This endpoint checks which models actually exist in the model folder and returns their availability status. **Model availability reflects the current `model_directory` setting immediately** (hot reload).

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

**Hot Reload Behavior:**
- Reflects current `model_directory` immediately after changes
- Shows newly downloaded models without server restart
- File existence checked at request time, not cached

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

### GET /api/model-options

Get all models that can be downloaded, regardless of whether they exist locally. This endpoint shows all models defined in `model_options` with their download configuration status.

**Difference from `/api/available-models`:**
- `/api/available-models`: Returns only models with files that exist on disk
- `/api/model-options`: Returns all models in configuration, showing which have `repo_id` configured for download

**Request:**
```bash
curl http://localhost:8000/api/model-options

# Filter by task type
curl http://localhost:8000/api/model-options?task_type=vision
```

**Query Parameters:**
- `task_type` (optional): Filter by `vision`, `chat`, or `embedding`

**Response:** `200 OK`
```json
{
  "models": [
    {
      "model_id": "qwen_3_0.6B",
      "name": "qwen_3_0.6B",
      "type": "chat",
      "model_file": "Qwen3-0.6B-Q4_K_M.gguf",
      "mmproj_file": null,
      "repo_id": "Qwen/Qwen3-0.6B-GGUF",
      "repo_id_configured": true,
      "llm_params": null
    },
    {
      "model_id": "gemma3_4b_q4_k_m",
      "name": "gemma3_4b_q4_k_m",
      "type": "vision",
      "model_file": "gemma-3-4b-it-Q4_K_M.gguf",
      "mmproj_file": "gemma_3_mmproj-F16.gguf",
      "repo_id": "",
      "repo_id_configured": false,
      "llm_params": null
    }
  ],
  "total_count": 2,
  "configured_count": 1,
  "task_type": null
}
```

**Response Fields:**
- `models`: Array of model configuration objects
  - `model_id`: Model identifier (key in `model_options`, used for download requests)
  - `name`: Model name
  - `type`: Model type (`vision`, `chat`, or `embedding`)
  - `model_file`: Model filename
  - `mmproj_file`: MMProj file for vision models (null for non-vision)
  - `repo_id`: Hugging Face repository ID
  - `repo_id_configured`: Boolean indicating if `repo_id` is configured (non-empty)
  - `llm_params`: Model-specific LLM parameters (if configured)
- `total_count`: Total number of models returned
- `configured_count`: Number of models with `repo_id` configured (ready to download)
- `task_type`: Filter applied (if any)

**Use Cases:**
- Check which models can be downloaded before calling `/api/download-models`
- Identify models missing `repo_id` configuration
- List all available model IDs for download requests
- Filter downloadable models by type

**Python Example:**
```python
import requests

# Get all downloadable models
response = requests.get("http://localhost:8000/api/model-options")
data = response.json()

print(f"Total models: {data['total_count']}")
print(f"Configured for download: {data['configured_count']}")

# Show models ready to download
for model in data['models']:
    if model['repo_id_configured']:
        print(f"✓ {model['model_id']}: {model['repo_id']}")
    else:
        print(f"✗ {model['model_id']}: repo_id not configured")
```

**JavaScript Example:**
```javascript
fetch('http://localhost:8000/api/model-options?task_type=vision')
  .then(res => res.json())
  .then(data => {
    console.log(`Found ${data.total_count} vision models`);
    console.log(`${data.configured_count} ready for download`);
    
    data.models.forEach(model => {
      const status = model.repo_id_configured ? '✓' : '✗';
      console.log(`${status} ${model.model_id}`);
    });
  });
```

---

### WebSocket /api/download-models

Download models from Hugging Face repositories. This endpoint downloads model files and mmproj files (for vision models) directly from configured Hugging Face repositories.

**Prerequisites:**
1. Install `huggingface_hub`: `pip install huggingface_hub`
2. Configure `repo_id` for models in `app/config/settings.py` (see [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md))

**WebSocket Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/download-models');
```

**Request Message:**
```json
{
  "model_ids": ["qwen_3_0.6B", "gemma3_4b_q4_k_m"],
  "force_redownload": false,
  "download_location": null
}
```

**Request Fields:**
- `model_ids` (required): Array of model IDs from `model_options` to download
- `force_redownload` (optional, default: false): If true, re-downloads files even if they already exist
- `download_location` (optional, default: null): Custom download location (absolute path). If null, uses configured `model_directory` or default location

**Response Messages:**

The endpoint sends multiple WebSocket messages with different types:

**1. Status Message** - General progress updates:
```json
{
  "type": "status",
  "message": "Starting download for 2 model(s)...",
  "data": null
}
```

**2. Progress Message** - Download progress with file information:
```json
{
  "type": "progress",
  "message": "Downloaded Qwen3-0.6B-Q4_K_M.gguf (512.5 MB)",
  "data": {
    "filename": "Qwen3-0.6B-Q4_K_M.gguf",
    "bytes_downloaded": 537395200,
    "total_bytes": 537395200
  }
}
```

**3. Result Message** - Completion status for each model:
```json
{
  "type": "result",
  "message": "Completed processing qwen_3_0.6B",
  "data": {
    "model_id": "qwen_3_0.6B",
    "overall_status": "completed",
    "files": [
      {
        "filename": "Qwen3-0.6B-Q4_K_M.gguf",
        "status": "completed",
        "error": null,
        "bytes_downloaded": 537395200,
        "total_bytes": 537395200
      }
    ]
  }
}
```

**4. Error Message** - Download errors:
```json
{
  "type": "error",
  "message": "Failed to download model_file.gguf: Connection timeout",
  "data": {
    "filename": "model_file.gguf",
    "error": "Connection timeout",
    "repo_id": "username/repo"
  }
}
```

**File Status Values:**
- `pending`: Download not yet started
- `downloading`: Currently downloading
- `completed`: Successfully downloaded
- `failed`: Download failed with error
- `skipped`: File already exists (use `force_redownload: true` to re-download)

**Overall Status Values:**
- `completed`: All files downloaded successfully (or skipped)
- `partial`: Some files succeeded, some failed
- `failed`: All files failed to download

**Python Example:**
```python
import asyncio
import json
import websockets

async def download_models():
    uri = "ws://localhost:8000/api/download-models"
    
    async with websockets.connect(uri) as websocket:
        # Send download request with custom location
        await websocket.send(json.dumps({
            "model_ids": ["qwen_3_0.6B"],
            "force_redownload": False,
            "download_location": "/custom/models/path"  # Optional
        }))
        
        # Receive progress updates
        async for message in websocket:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            
            if data['type'] == 'progress' and data.get('data'):
                bytes_dl = data['data']['bytes_downloaded']
                total = data['data']['total_bytes']
                percent = (bytes_dl / total) * 100 if total > 0 else 0
                print(f"  Progress: {percent:.1f}%")

asyncio.run(download_models())
```

**JavaScript Example:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/download-models');

ws.onopen = () => {
    ws.send(JSON.stringify({
        model_ids: ["qwen_3_0.6B", "gemma3_4b_q4_k_m"],
        force_redownload: false,
        download_location: "/custom/models/path"  // Optional
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`[${data.type}] ${data.message}`);
    
    if (data.type === 'progress' && data.data) {
        const percent = (data.data.bytes_downloaded / data.data.total_bytes) * 100;
        console.log(`  ${data.data.filename}: ${percent.toFixed(1)}%`);
    }
};

ws.onerror = (error) => console.error('Error:', error);
ws.onclose = () => console.log('Download complete');
```

**Common Errors:**

**Missing repo_id:**
```json
{
  "type": "error",
  "message": "Model qwen_3 does not have a repo_id configured. Please add repo_id to model_options in settings.py"
}
```
Solution: Add `"repo_id": "username/repo-name"` to the model in `app/config/settings.py`

**Invalid model_id:**
```json
{
  "type": "error",
  "message": "Invalid model_id: unknown_model. Not found in model_options."
}
```
Solution: Use a valid model ID from `model_options`

**Library not installed:**
```json
{
  "type": "error",
  "message": "huggingface_hub library not installed. Please install it with: pip install huggingface_hub"
}
```
Solution: `pip install huggingface_hub`

**File not found in repository:**
```json
{
  "type": "error",
  "message": "Failed to download: File not found",
  "data": {
    "filename": "model-file.gguf",
    "error": "File not found in repository",
    "repo_id": "username/repo"
  }
}
```
Solution: Verify the `model_file` name matches the exact filename in the Hugging Face repository

**Use Cases:**
- Initial setup: Download all required models
- Model updates: Re-download models when new versions are available
- Automated deployment: Download models as part of server initialization
- Testing: Download small models for testing purposes

**Important - Model Storage Location:**
- **Custom `download_location`**: If provided in the request, downloads to this specific path
- **Custom `model_directory`**: If set via POST /api/config, downloads to this path (when download_location is null)
- **Default behavior**: Downloads to `saved_llm/` alongside storage-metadata.json
- **Fallback**: Downloads to project `model/` directory if storage metadata not set
- Example: If your storage metadata is at `/Users/you/data/storage-metadata.json`, LLM models will be stored in `/Users/you/data/saved_llm/`
- **Note:** Face recognition models (buffalo_l) always remain in the static `model/models/` directory and are not moved to `saved_llm/`

**Priority Order for Download Location:**
1. `download_location` parameter (if provided in request)
2. `model_directory` config (if set via POST /api/config)
3. `saved_llm/` folder (if storage_metadata_path is set)
4. Default `model/` directory (fallback)

**Notes:**
- Downloads are resumable - interrupted downloads can be resumed automatically
- For vision models, both `model_file` and `mmproj_file` are downloaded from the same repository
- Large models (2-8 GB) may take significant time depending on connection speed
- Downloads work correctly in both development and PyInstaller-packaged modes

**See Also:**
- [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md) - Complete download feature documentation
- [REPO_ID_EXAMPLES.md](REPO_ID_EXAMPLES.md) - Example repository IDs for all models

---

### POST /api/set-storage-metadata

Set the path to the storage metadata JSON file. This file contains information about all files to be indexed. Must be called before using other endpoints.

**Important:** When you set the storage metadata path, the following directories are automatically created in the same location:
- `rag_db/` - For RAG database files
- `saved_llm/` - For downloaded LLM models (GGUF files only)

**Note:** Face recognition models (buffalo_l) remain in the static `model/models/` directory in the project root and are not affected by this setting.

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
    "saved_llm_directory": "/Users/username/data/saved_llm",
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

Load an existing RAG database from disk. The RAG directory is determined from the storage metadata path. Also loads conversation compaction data if available.

**Request:**
```bash
curl -X POST http://localhost:8000/api/load-rag
```

**Response:** `200 OK`
```json
{
  "status": "success",
  "message": "RAG database loaded successfully",
  "data": {
    "dimension": 768,
    "indexed_files": 156
  }
}
```

**Response Fields:**
- `dimension`: The embedding dimension of the loaded RAG index
- `indexed_files`: Number of files in the RAG index

**Error Response:** `400 Bad Request`
```json
{
  "detail": "Storage metadata not set. Call /set-storage-metadata first."
}
```

**Error Response:** `404 Not Found`
```json
{
  "detail": "RAG database not found. Generate RAG first using /generate-rag endpoint."
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

**Important:** If embeddings have different dimensions (e.g., from different embedding models), the system will:
1. Detect all embedding dimensions
2. Use the majority dimension (most common)
3. Exclude files with mismatched dimensions from the RAG
4. Return the list of excluded files so you can re-embed them with the correct model

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

**Dimension Mismatch Warning (if applicable):**
```json
{
  "type": "status",
  "message": "Warning: 3 file(s) had mismatched embedding dimensions and were excluded from RAG",
  "data": {
    "mismatched_files": ["image1.jpg", "image2.jpg", "video1.mp4"],
    "majority_dimension": 768
  }
}
```

```json
{
  "type": "status",
  "message": "Removing image1.jpg: dimension 1024 (expected 768)"
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
  "message": "RAG database created and loaded successfully",
  "data": {
    "total_indexed": 153,
    "removed_count": 3,
    "mismatched_files": ["image1.jpg", "image2.jpg", "video1.mp4"],
    "majority_dimension": 768,
    "conversation_count": 12
  }
}
```

**Result Data Fields:**
- `total_indexed`: Number of files successfully added to the RAG index
- `removed_count`: Number of files excluded due to dimension mismatch
- `mismatched_files`: List of filenames that were excluded (re-embed these with the correct model)
- `majority_dimension`: The embedding dimension used for the RAG index
- `conversation_count`: Number of compacted conversation embeddings merged into the RAG index (from `/api/compact-conversations`)

**4. Connection Closes**

**Error Case:**

No embeddings available:
```json
{
  "type": "error",
  "message": "No embeddings available. Generate embeddings first."
}
```

**Use Case - Handling Mismatched Embeddings:**

If files were excluded due to dimension mismatch:
1. Note the `mismatched_files` list from the result
2. Check which embedding model was used for those files
3. Re-generate embeddings for those files using the correct model (matching `majority_dimension`)
4. Rebuild the RAG database

**Note:** The RAG database is saved to `{metadata_directory}/rag_db/` as:
- `faiss_index.bin` - FAISS vector index
- `faiss_index_idmap.pkl` - File ID mapping

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

**Related Endpoint:** For using external cloud LLMs (OpenAI, Anthropic, etc.) instead of local models, see `/api/cloud-chat` which provides RAG context without running inference.

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

### WS /api/deep-chat

Agentic deep chat with an OpenAI-compatible tool-calling loop. The local LLM autonomously calls 4 MCP tools to explore and retrieve from the media library before producing a final answer. The server pre-loads a global library context (tags, date range, relevant past conversations) into the system prompt so the agent can often skip early exploration calls.

**Requires `llm_mode: server`** — tool calling is only supported with llama-server.

**Important:** Each WebSocket connection handles a **single request-response cycle**. The connection closes after the answer is sent. For follow-up questions, open a new connection and provide `history`.

**Connection:** `ws://localhost:8000/api/deep-chat`

**Key Features:**
- **Pre-loaded library context**: Global tags, overall date range, and relevant past conversations are injected into the system prompt before the first tool call — the agent can go directly to `scoped_rag_search` when the global context is sufficient
- **4 MCP tools**: `get_scoped_tags`, `get_scoped_dates`, `scoped_rag_search`, `get_conversation_rag`
- **Budget enforcement**: `chat_rounds` config = total tool calls allowed. At budget=1 only `scoped_rag_search` is offered; at budget=0 the model is forced to answer immediately
- **Tool call history truncation**: After each tool result is consumed it is compacted to save context window space (controlled by `tool_history_max_tags` and `tool_history_max_results` config)
- **Transparent intermediate messages**: Client receives `thinking`, `progress`, `conclusion`, and `files` messages

**How This Differs from Regular Chat:**

| Aspect | `/api/chat` | `/api/deep-chat` |
|--------|-------------|------------------|
| Initial context | Automatic RAG search | Pre-loaded library context (tags, dates, conversations) |
| LLM tool access | None | 4 MCP tools (`get_scoped_tags`, `get_scoped_dates`, `scoped_rag_search`, `get_conversation_rag`) |
| Budget | N/A | `chat_rounds` tool calls (default 10) |
| System prompt | Standard chat prompt | Deep chat prompt with tool definitions, budget rules, and pre-loaded context |
| Requires llm_mode | `server` or `cli` | `server` only |
| Final answer format | Plain text | `<conclusion>...</conclusion>` and `<files>...</files>` tags |

---

**1. Client Connects**

**2. Server Ready Message:**
```json
{"type": "status", "message": "Deep Chat ready. Send your message."}
```

**3. Client Sends Message:**
```json
{
  "message": "What beach photos do I have and when were they taken?",
  "history": [],
  "image_name": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | ✅ | The user's question |
| `history` | array | ❌ | Previous conversation turns (OpenAI format). Omit or send `[]` for a new conversation |
| `image_name` | string | ❌ | Optional filename for a visual conversation — the image's tags and description are injected into the user message |

**4. Server Loads Model and Gathers Context:**
```json
{"type": "status", "message": "Loading chat model Qwen3-8B-Q4_K_M.gguf..."}
```
```json
{"type": "status", "message": "Deep Chat: Starting tool-calling loop..."}
```
```json
{"type": "status", "message": "Gathering library context..."}
```

**5. Tool-Calling Loop:**

The server iterates until the LLM produces a response with no tool calls (or the budget runs out):

```json
{
  "type": "status",
  "message": "[Iteration 1] Thinking... (10 tool call(s) remaining)",
  "data": {"iteration": 1, "budget_remaining": 10, "tools_called": 0}
}
```

When the LLM calls a tool, the server sends two messages — an invocation status and a result:

```json
{
  "type": "status",
  "message": "[Tool call 1] scoped_rag_search\n{\n  \"query\": \"beach photos\",\n  \"start_date\": \"2024-01-01\",\n  \"end_date\": \"2025-12-31\",\n  \"min_tags\": [\"beach\"]\n}",
  "data": {"tool_name": "scoped_rag_search", "arguments": {"query": "beach photos", "start_date": "2024-01-01", "end_date": "2025-12-31", "min_tags": ["beach"]}, "tool_call_index": 1, "tool_call_number": 1, "budget_remaining": 9}
}
```

```json
{
  "type": "progress",
  "message": "Top 5 results for 'beach photos'...\n\n• IMG_2847.jpg\n  Date: 2025-06-15\n  Tags: beach, family, summer",
  "data": {"tool_name": "scoped_rag_search", "tool_call_number": 1, "result_length": 312, "budget_after": 9}
}
```

> **Note:** The budget annotation (`[Tool calls remaining: N]`) is appended to the tool result only inside the LLM conversation history — it is **not** included in the `message` field of the WebSocket `progress` message sent to the client.

If the LLM emits reasoning content before a tool call, it is sent as a `thinking` message:
```json
{
  "type": "thinking",
  "message": "<think>The pre-loaded context shows beach tags in 2024-2025...</think>",
  "data": {"iteration": 1, "budget_remaining": 10, "pending_tool_calls": 1}
}
```

**6. Final Answer:**

When the LLM stops calling tools, the conclusion is streamed as `progress` chunks, then sent in full as a `conclusion` message:

```json
{"type": "progress", "message": "You have 12 beach photos"}
```
```json
{"type": "progress", "message": " taken between June and August 2025..."}
```
```json
{"type": "conclusion", "message": "You have 12 beach photos taken between June and August 2025. The most recent is IMG_2847.jpg from June 15, a family outing at a sandy beach."}
```

Referenced files:
```json
{"type": "files", "message": "IMG_2847.jpg, IMG_2901.jpg", "data": {"files": ["IMG_2847.jpg", "IMG_2901.jpg"]}}
```

Session summary:
```json
{"type": "full_response", "message": "You have 12 beach photos...", "data": {"tools_called": 1, "files": ["IMG_2847.jpg", "IMG_2901.jpg"]}}
```

**7. Connection Closes Automatically**

---

**Message Type Summary:**

| Type | When sent | Notable `data` fields |
|------|-----------|----------------------|
| `status` | Ready, model loading, context gathering, each iteration start, each tool invocation | `iteration`, `budget_remaining`, `tools_called`, `tool_name`, `arguments` |
| `thinking` | LLM reasoning content emitted before tool calls | `iteration`, `budget_remaining`, `pending_tool_calls` |
| `progress` | Tool result (during loop) and final answer chunks (streaming) | `tool_name`, `tool_call_number`, `budget_after` |
| `conclusion` | Complete final answer text | — |
| `files` | Referenced file names | `files` (array of strings) |
| `full_response` | Session summary sent just before connection closes | `tools_called`, `files` |
| `error` | Any error | — |

---

**Available MCP Tools:**

| Tool | Required args | Optional args | Description |
|------|--------------|---------------|-------------|
| `get_scoped_tags` | — | `start_date`, `end_date`, `min_tags` | List tags present in media files within an optional date/tag scope |
| `get_scoped_dates` | — | `start_date`, `end_date`, `min_tags` | List date ranges within an optional scope |
| `scoped_rag_search` | `query`, `start_date`, `end_date`, `min_tags` | — | Semantic search within a date/tag scope. Always available, even at budget=1 |
| `get_conversation_rag` | `query` | — | Semantic search over compacted conversation memories |

---

**Configuration:**

The tool budget is controlled by the `chat_rounds` setting (default: 10, range: 1–50):
```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"chat_rounds": 10}'
```

See also: `tool_history_max_tags`, `tool_history_max_results`, `max_tags_per_scope`, `max_dates_per_scope` in the configuration reference.

**When to Use:**

- **Use `/api/deep-chat`** for complex, exploratory questions that benefit from multi-step library investigation (e.g., "What did we do at the beach last summer?")
- **Use `/api/chat`** for simple questions where a single-pass automatic RAG search is sufficient
- **Use `/api/mcp`** when you want to drive the same 4-tool loop yourself from a Cloud AI (GPT-4, Claude, etc.)

**Error Cases:**

```json
{"type": "error", "message": "Deep Chat requires server mode (llm_mode: server). Tool calling needs llama-server."}
```
```json
{"type": "error", "message": "No message provided"}
```
```json
{"type": "error", "message": "Deep chat error: <error details>"}
```

---

### WS /api/cloud-chat

Get RAG context and system prompt for use with external cloud LLMs (OpenAI, Anthropic, Google, etc.). This endpoint performs RAG search and returns structured context without running any local LLM inference. The client is responsible for calling their chosen cloud LLM provider.

**Use Case:** Ideal when you want to use powerful cloud models (GPT-4, Claude, Gemini) while leveraging the server's RAG capabilities for context-aware responses about your local files.

**Connection:** `ws://localhost:8000/api/cloud-chat`

**1. Client Connects (No Initial Configuration Required)**

**2. Server Loads Embedding Model:**
```json
{
  "type": "status",
  "message": "Loading RAG database..."
}
```

```json
{
  "type": "status",
  "message": "Loading embedding model embeddinggemma-300M-Q8_0.gguf..."
}
```

```json
{
  "type": "status",
  "message": "Ready to provide RAG context. Send your message."
}
```

**3. Client Sends Message:**

**Basic Request:**
```json
{
  "message": "What beach photos do I have?"
}
```

**With Conversation History:**
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

**With Image Context:**
```json
{
  "message": "Tell me about this photo",
  "image_name": "beach_sunset.jpg",
  "history": []
}
```

**Message Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | ✅ | The current user message to search for |
| `history` | array | ❌ | Optional chat history in OpenAI format. Used to build context-aware search query |
| `image_name` | string | ❌ | Optional image filename for visual context (loads tags, description, and base64 image) |

**4. Server Performs RAG Search:**
```json
{
  "type": "status",
  "message": "Searching knowledge base..."
}
```

**5. Server Returns Complete Context Package:**
```json
{
  "type": "result",
  "message": "RAG context retrieved successfully",
  "data": {
    "system_prompt": "You are Persona, a helpful AI assistant...",
    "rag_context": "File: vacation/beach_sunset.jpg\nTags: beach, sunset, ocean, vacation\nDescription: A beautiful sunset over the ocean...\nSimilarity: 0.892\n\nFile: summer/surfing_day.jpg\nTags: beach, surfing, sports, summer\nDescription: Action shot of surfer riding a wave...\nSimilarity: 0.845",
    "relevant_files": [
      "vacation/beach_sunset.jpg",
      "summer/surfing_day.jpg"
    ],
    "file_details": [
      {
        "fileName": "vacation/beach_sunset.jpg",
        "type": "photo",
        "tags": ["beach", "sunset", "ocean"],
        "description": "A beautiful sunset over the ocean...",
        "creationTime": "2024-07-15T18:30:00",
        "source": "file"
      },
      {
        "fileName": "conv:Hello2026-04-08T10:09:35",
        "type": "conversation_memory",
        "tags": [],
        "description": "User discussed their favorite beach vacation spots and planned a summer trip.",
        "creationTime": "2026-04-08T10:15:00",
        "source": "conversation"
      }
    ],
    "image_context": null,
    "user_message": "What beach photos do I have?",
    "history": []
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `system_prompt` | string | The configured chat system prompt from server config |
| `rag_context` | string | Formatted text with relevant file information, ready to be included in your LLM prompt |
| `relevant_files` | array[string] | List of relevant filenames |
| `file_details` | array[object] | Detailed metadata for each result. Each object includes a `source` field (`"file"` or `"conversation"`) to distinguish file results from compacted conversation memories |
| `image_context` | object/null | If `image_name` provided: contains `image_name`, `tags`, `description`, and `image_base64` |
| `user_message` | string | Echo of the user's message |
| `history` | array/null | Echo of the provided conversation history |

**6. Connection Closes**

**With Image Context Response:**
```json
{
  "type": "result",
  "message": "RAG context retrieved successfully",
  "data": {
    "system_prompt": "You are Persona...",
    "rag_context": "File: vacation/beach_sunset.jpg...",
    "relevant_files": ["vacation/beach_sunset.jpg"],
    "file_details": [...],
    "image_context": {
      "image_name": "beach_sunset.jpg",
      "tags": ["beach", "sunset", "ocean"],
      "description": "A beautiful sunset over the ocean with palm trees",
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    "user_message": "Tell me about this photo",
    "history": []
  }
}
```

**Error Cases:**

RAG not available:
```json
{
  "type": "error",
  "message": "RAG not available. Generate RAG first."
}
```

Invalid history format:
```json
{
  "type": "error",
  "message": "history parameter must be a list of message objects"
}
```

Image not found:
```json
{
  "type": "error",
  "message": "File not found: unknown.jpg"
}
```

**Usage Example with OpenAI:**

```python
import asyncio
import websockets
import json
import openai

async def cloud_chat_with_openai():
    # 1. Get RAG context from AI Server
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        # Wait for ready
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        # Send request
        await ws.send(json.dumps({
            "message": "What photos do I have from my vacation?",
            "history": []
        }))
        
        # Get context
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'result':
                context_data = data['data']
                break
    
    # 2. Call OpenAI with the context
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": context_data['system_prompt']
            },
            {
                "role": "system",
                "content": f"Relevant files from user's collection:\n\n{context_data['rag_context']}"
            },
            {
                "role": "user",
                "content": context_data['user_message']
            }
        ]
    )
    
    print(response.choices[0].message.content)

asyncio.run(cloud_chat_with_openai())
```

**Usage Example with Anthropic Claude:**

```python
import asyncio
import websockets
import json
import anthropic

async def cloud_chat_with_claude():
    # Get RAG context
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        await ws.send(json.dumps({
            "message": "Describe my beach photos",
            "history": []
        }))
        
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'result':
                context_data = data['data']
                break
    
    # Call Claude
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system=f"{context_data['system_prompt']}\n\nRelevant files:\n{context_data['rag_context']}",
        messages=[
            {
                "role": "user",
                "content": context_data['user_message']
            }
        ]
    )
    
    print(message.content[0].text)

asyncio.run(cloud_chat_with_claude())
```

**Multi-turn Conversation Example:**

```python
async def multi_turn_cloud_chat():
    conversation_history = []
    
    # First turn
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        # Wait for ready and send first message
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        await ws.send(json.dumps({
            "message": "What vacation photos do I have?",
            "history": conversation_history
        }))
        
        # Get context
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'result':
                first_context = data['data']
                break
    
    # Call your cloud LLM
    first_response = call_your_llm(first_context)  # Your LLM call
    
    # Add to history
    conversation_history.append({
        "role": "user",
        "content": "What vacation photos do I have?"
    })
    conversation_history.append({
        "role": "assistant",
        "content": first_response
    })
    
    # Second turn with history
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if "ready" in data['message'].lower():
                break
        
        await ws.send(json.dumps({
            "message": "Show me the beach ones",
            "history": conversation_history  # Include previous conversation
        }))
        
        async for message in ws:
            data = json.loads(message)
            if data['type'] == 'result':
                second_context = data['data']
                break
    
    # Call your cloud LLM with updated context
    second_response = call_your_llm(second_context)
```

**Features:**
- **No local LLM inference**: Server only performs RAG search, client calls cloud LLM
- **Conversation history support**: Include previous turns for context-aware RAG search
- **Image context**: Load image metadata and base64 data for multimodal cloud models
- **Flexible**: Use with any cloud LLM provider (OpenAI, Anthropic, Google, Cohere, etc.)
- **Cost-effective**: Leverage powerful cloud models while using local RAG for privacy-sensitive file indexing
- **Lower latency for RAG**: Local embedding search is faster than cloud-based retrieval

**When to Use:**
- You want to use GPT-4, Claude, or other cloud models for better response quality
- You need features only available in cloud models (longer context, better reasoning)
- You want to keep file indexing local but use cloud models for conversation
- You're building a production application with commercial cloud LLM APIs

**Related Endpoint:** For fully local chat using server's LLM, see `/api/chat`

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

# 5c. Cloud Chat (Get RAG context for external LLM)
async def cloud_chat():
    async with websockets.connect(f"{WS_BASE}/api/cloud-chat") as ws:
        # Wait for ready
        while True:
            message = await ws.recv()
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            if "ready" in data['message'].lower():
                break
        
        # Send message
        await ws.send(json.dumps({
            "message": "What photos do I have from the beach?",
            "history": []
        }))
        
        # Get RAG context
        async for message in ws:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")
            
            if data['type'] == 'result':
                # Use this context with your cloud LLM (OpenAI, Anthropic, etc.)
                system_prompt = data['data']['system_prompt']
                rag_context = data['data']['rag_context']
                relevant_files = data['data']['relevant_files']
                user_message = data['data']['user_message']
                
                print(f"\nSystem Prompt: {system_prompt[:100]}...")
                print(f"\nRAG Context:\n{rag_context}")
                print(f"\nRelevant Files: {', '.join(relevant_files)}")
                
                # Now call your cloud LLM with this context
                # Example: openai.ChatCompletion.create(...)
                break
        # Connection closes automatically

asyncio.run(cloud_chat())

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

### WS /api/compact-conversations

Compact (summarize) conversations into memory entries for RAG retrieval. This is the "dreaming mechanism" — it reads raw conversations from `conversation_map.json`, uses the chat LLM to produce a concise summary for each, then embeds those summaries. After compacting, call `/api/generate-rag` to merge conversation embeddings into the main RAG index.

**Use Case:** Automatically distill conversation history into searchable memories so the AI can recall past discussions in future chats. Similar to how dreaming consolidates memories.

**Connection:** `ws://localhost:8000/api/compact-conversations`

**Prerequisites:**
- Storage metadata must be set (`/api/set-storage-metadata`)
- A `conversation_map.json` file must exist in the RAG directory

**conversation_map.json Format:**
```json
{
  "Hello2026-04-08T10:09:35": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well! How can I help you?"}
  ],
  "VacationPlanning2026-04-09T14:22:10": [
    {"role": "user", "content": "Help me plan a beach vacation"},
    {"role": "assistant", "content": "I'd love to help! What dates are you considering?"},
    {"role": "user", "content": "July 2026, somewhere tropical"},
    {"role": "assistant", "content": "Great! Here are some options..."}
  ]
}
```

Each key is a conversation ID and each value is an array of message objects with `role` and `content`.

**1. Client Connects and Sends Request:**

```json
{
  "count": 5,
  "force_recompact": false,
  "chat_model": null,
  "embedding_model": null
}
```

**Request Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `count` | int | ❌ | 5 | Maximum number of conversations to compact in this session |
| `force_recompact` | bool | ❌ | false | If true, re-compact conversations from `conversation_map.json` even if they were already compacted. Only affects conversations present in `conversation_map.json` — any additional entries in `conversation_embeddings_map.json` (from previous compaction runs) are preserved unchanged |
| `chat_model` | string/null | ❌ | null | Chat model to use for summarization (null = use config default) |
| `embedding_model` | string/null | ❌ | null | Embedding model for summary embeddings (null = use config default) |

**2. Server Loads Conversations and Begins Compaction:**

```json
{
  "type": "status",
  "message": "Found 15 conversations in conversation_map.json"
}
```

```json
{
  "type": "status",
  "message": "Compacting 5 conversations (already compacted: 3)..."
}
```

```json
{
  "type": "status",
  "message": "Loading chat model Qwen3-0.6B-Q4_K_M.gguf..."
}
```

**3. Phase 1 — Summarization:**

For each conversation, the server sends status and progress messages:

```json
{
  "type": "status",
  "message": "Compacting conversation 1/5: Hello2026-04-08T10:09:35"
}
```

```json
{
  "type": "progress",
  "message": "Summarized 1/5",
  "data": {
    "conversation_id": "Hello2026-04-08T10:09:35",
    "summary": "User greeted the assistant; nothing notable discussed.",
    "current": 1,
    "total": 5
  }
}
```

The LLM summarization prompt instructs: *"Extract facts about the user worth remembering for future interactions — personal preferences, traits, life details, stated goals, diagnoses, decisions, or any information revealed or inferred about them. Write each fact as a short, direct statement (e.g. 'User prefers red and black shoes.'). List multiple facts one per line. Do not summarise what happened — only state facts about the user. If there are no facts worth remembering, respond with exactly: 'nothing to remember'."*

**4. Phase 2 — Embedding:**

```json
{
  "type": "status",
  "message": "Loading embedding model embeddinggemma-300M-Q8_0.gguf..."
}
```

```json
{
  "type": "status",
  "message": "Embedding summary 1/5: Hello2026-04-08T10:09:35"
}
```

**5. Server Saves and Returns Result:**

```json
{
  "type": "status",
  "message": "Unloading model..."
}
```

```json
{
  "type": "result",
  "message": "Successfully compacted 5 conversations",
  "data": {
    "compacted_ids": [
      "Hello2026-04-08T10:09:35",
      "VacationPlanning2026-04-09T14:22:10",
      "PhotoSearch2026-04-09T15:00:00",
      "WeatherChat2026-04-10T09:30:00",
      "RecipeLookup2026-04-10T12:15:00"
    ],
    "compacted_count": 5,
    "remaining_uncompacted": 7,
    "total_conversations": 15
  }
}
```

**Result Data Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `compacted_ids` | array[string] | List of conversation IDs that were compacted |
| `compacted_count` | int | Number of conversations compacted in this session |
| `remaining_uncompacted` | int | Number of conversations in `conversation_map.json` that have not yet been compacted |
| `total_conversations` | int | Total number of conversations in `conversation_map.json` |

**6. Connection Closes Automatically**

**After Compacting — Rebuild RAG:**

After compacting conversations, call `/api/generate-rag` to rebuild the RAG index. The new index will include both file embeddings and conversation memory embeddings. The `conversation_count` field in the generate-rag result shows how many conversation embeddings were merged.

**Error Cases:**

No RAG directory:
```json
{
  "type": "error",
  "message": "No RAG directory configured. Set storage metadata first."
}
```

Missing conversation map:
```json
{
  "type": "error",
  "message": "conversation_map.json not found in RAG directory."
}
```

Invalid count:
```json
{
  "type": "error",
  "message": "count must be >= 1"
}
```

No conversations to compact:
```json
{
  "type": "result",
  "message": "No conversations to compact",
  "data": {
    "compacted_ids": [],
    "compacted_count": 0,
    "remaining_uncompacted": 0,
    "total_conversations": 15
  }
}
```

General error:
```json
{
  "type": "error",
  "message": "Compaction error: <ErrorType>: <details>"
}
```

---

### WS /api/cloud-compact

**Connection:** `ws://localhost:8000/api/cloud-compact`

**Description:** Embeds client-provided conversation summaries into the conversation compaction service. Designed for the **Cloud AI workflow**: the client app summarizes conversations using a cloud LLM (e.g., GPT-4, Claude) and sends the summaries here; the local server is only responsible for creating embeddings. No chat model is used on the server side.

**Typical Workflow:**
1. Client app summarizes conversations using a cloud LLM
2. Client calls `/api/cloud-compact` with the summaries → server embeds them
3. Client calls `/api/generate-rag` → conversation embeddings are merged into the RAG index
4. Subsequent `/api/cloud-chat` or `/api/chat` calls will include conversation memory in RAG results

**Use Case vs `/api/compact-conversations`:**

| Feature | `/api/compact-conversations` | `/api/cloud-compact` |
|---------|------------------------------|----------------------|
| Summarization | Local chat model | Client-provided (cloud AI) |
| Embedding | Local embedding model | Local embedding model |
| Requires chat model | Yes | No |
| Use when | Fully local workflow | Cloud AI companion workflow |

---

**Request Format:**

```json
{
  "conversations": [
    {"id": "conv_abc123", "summary": "User asked about vacation spots in Italy."},
    {"id": "conv_def456", "summary": "Discussion about recipe modifications for gluten-free cooking."}
  ],
  "embedding_model": null,
  "force_reembed": false
}
```

**Request Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `conversations` | array | Yes | — | List of objects with `id` and `summary` fields |
| `conversations[].id` | string | Yes | — | Unique conversation identifier |
| `conversations[].summary` | string | Yes | — | Compacted summary text produced by the client |
| `embedding_model` | string/null | No | config value | Local embedding model to use for embedding summaries |
| `force_reembed` | boolean | No | `false` | If `true`, re-embed conversations that already have an entry |

---

**Message Sequence:**

**1. Status — embedding started:**
```json
{
  "type": "status",
  "message": "Embedding 2 conversation summaries (skipping 0 already embedded)..."
}
```

**2. Status — model loading:**
```json
{
  "type": "status",
  "message": "Loading embedding model nomic-embed-text-v1.5.Q4_K_M.gguf..."
}
```

**3. Status — per-conversation progress:**
```json
{
  "type": "status",
  "message": "Embedding summary 1/2: conv_abc123"
}
```

**4. Progress — after each embedding:**
```json
{
  "type": "progress",
  "message": "Embedded 1/2",
  "data": {
    "conversation_id": "conv_abc123",
    "current": 1,
    "total": 2
  }
}
```

**5. Status — model unloaded:**
```json
{"type": "status", "message": "Unloading embedding model..."}
```

**6. Result:**
```json
{
  "type": "result",
  "message": "Successfully embedded 2 conversation summaries",
  "data": {
    "embedded_ids": ["conv_abc123", "conv_def456"],
    "embedded_count": 2,
    "skipped_count": 0
  }
}
```

**7. Connection Closes Automatically**

**Result Data Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `embedded_ids` | array[string] | Conversation IDs that were embedded in this call |
| `embedded_count` | int | Number of summaries embedded |
| `skipped_count` | int | Number of entries skipped because they were already embedded (and `force_reembed` was false) |

**After Embedding — Rebuild RAG:**

Call `/api/generate-rag` to rebuild the RAG index so that the new conversation embeddings are included in future RAG searches. The `conversation_count` field in the result will reflect the total merged count.

**Error Cases:**

Invalid input:
```json
{"type": "error", "message": "conversations must be a non-empty list"}
```

```json
{"type": "error", "message": "Each entry in conversations must have 'id' and 'summary' fields"}
```

Nothing to embed (all already embedded, `force_reembed` = false):
```json
{
  "type": "result",
  "message": "No new conversations to embed",
  "data": {
    "embedded_ids": [],
    "embedded_count": 0,
    "skipped_count": 3
  }
}
```

General error:
```json
{
  "type": "error",
  "message": "Cloud-compact error: <ErrorType>: <details>"
}
```

---

### WS /api/mcp

**Cloud AI Deep Chat** — MCP tool endpoint. Exposes the same four MCP tools that drive the local `/deep-chat` agentic loop, so a Cloud AI (GPT-4, Claude, Gemini, etc.) can run its own multi-turn tool-calling session with identical retrieval capabilities.

Unlike other WebSocket endpoints, this is a **persistent, multi-message connection** — the client can send multiple requests without reconnecting, enabling a full multi-iteration tool-calling loop.

**Connection:** `ws://localhost:8000/api/mcp`

**Prerequisites:**
- Storage metadata must be set (`/api/set-storage-metadata`)
- RAG must be generated (`/api/generate-rag`)
- Conversations should be compacted for best results (`/api/compact-conversations` or `/api/cloud-compact`)

**Five Actions Available:**

| Action | Equivalent MCP Tool | Purpose |
|--------|---------------------|---------|
| `get_library_context` | _(pre-load helper)_ | Get the full library overview string + structured metadata that local deep-chat injects into its system prompt |
| `get_scoped_tags` | `get_scoped_tags` | List top tags from a date/tag-scoped subset of the library |
| `get_scoped_dates` | `get_scoped_dates` | List contiguous date ranges matching a scope |
| `scoped_rag_search` | `scoped_rag_search` | Semantic search within a date/tag-scoped subset |
| `get_conversation_rag` | `get_conversation_rag` | Semantic search across compacted conversation summaries |

---

**1. Client Connects**

**2. Server Sends Ready:**
```json
{
  "type": "status",
  "message": "Cloud Deep Chat MCP tools ready."
}
```

---

**Action: `get_library_context`**

Returns the pre-loaded library context string (identical to what the local agent receives in its system prompt) plus structured metadata for programmatic use by the Cloud AI.

**Request:**
```json
{
  "action": "get_library_context",
  "query": "beach vacation photos"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ❌ | User question — used to find relevant past conversations to include in context |

**Response:**
```json
{
  "type": "result",
  "message": "Library context retrieved",
  "data": {
    "system_prompt": "You are a research assistant...\n\nPRE-LOADED LIBRARY CONTEXT...",
    "library_context": "PRE-LOADED LIBRARY CONTEXT\n\nGlobal tags (top 50 from 1250 files):\nbeach, sunset, family, ...\n\nLibrary date range: 2023-01-15 → 2026-04-10\n\nRelevant past conversations:\n...",
    "tool_definitions": [{"type": "function", "function": {"name": "get_scoped_tags", ...}}, ...],
    "tool_budget": 10,
    "total_files": 1250,
    "date_range": {"min": "2023-01-15", "max": "2026-04-10"},
    "top_tags": ["beach", "sunset", "family", "vacation", "dogs"],
    "conversation_count": 8
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `system_prompt` | string | Full system prompt with library context already injected — ready to use directly in Cloud AI calls |
| `library_context` | string | Pre-loaded context block (same as injected into system prompt) — useful if Cloud AI wants to customize the prompt |
| `tool_definitions` | array | OpenAI function-calling schemas for all registered MCP tools — pass directly as the `tools` parameter in Cloud AI LLM calls, no hardcoding needed |
| `tool_budget` | int | Number of tool calls allowed before Cloud AI should produce a final answer |
| `total_files` | int | Total files in the library |
| `date_range` | object | `min` and `max` creation dates |
| `top_tags` | array | Top 200 most common tags, sorted by frequency |
| `conversation_count` | int | Number of compacted conversations with embeddings |

---

**Action: `get_scoped_tags`**

Returns the top-M tags from a date- and/or tag-scoped subset of the file library.

**Request:**
```json
{
  "action": "get_scoped_tags",
  "start_date": "2025-06-01",
  "end_date": "2025-06-30",
  "min_tags": ["beach"],
  "strict": false,
  "top_m": 50
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `start_date` | string\|null | ❌ | null | Inclusive start date (YYYY-MM-DD) |
| `end_date` | string\|null | ❌ | null | Inclusive end date (YYYY-MM-DD) |
| `min_tags` | array\|null | ❌ | null | Tag filter — files must match at least one (or all if `strict=true`) |
| `strict` | bool | ❌ | false | `true` = all tags must match (AND logic); `false` = any tag matches (OR logic) |
| `top_m` | int | ❌ | 50 | Number of top tags to return |
| `budget_remaining` | int\|null | ❌ | null | Cloud AI's remaining tool call count **before** this call. Server appends `[Tool calls remaining: N]` to the result — same annotation the local deep-chat handler uses. |

**Response:**
```json
{
  "type": "result",
  "message": "Top 15 tags in scope (2025-06-01 → 2025-06-30, tags: beach):\nbeach, family, summer, ...\n\n[Tool calls remaining: 7]",
  "data": {
    "tool": "get_scoped_tags",
    "raw": "Top 15 tags in scope (2025-06-01 → 2025-06-30, tags: beach):\nbeach, family, summer, ...",
    "budget_after": 7
  }
}
```

`data.raw` is always the unannotated result. Use `message` as the tool result string in the Cloud AI conversation.

---

**Action: `get_scoped_dates`**

Returns the date ranges (and file counts) of files matching the given scope.

**Request:**
```json
{
  "action": "get_scoped_dates",
  "start_date": "2025-01-01",
  "end_date": "2025-12-31",
  "min_tags": ["family"],
  "strict": false,
  "top_k": 10
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `start_date` | string\|null | ❌ | null | Inclusive start date |
| `end_date` | string\|null | ❌ | null | Inclusive end date |
| `min_tags` | array\|null | ❌ | null | Tag filter |
| `strict` | bool | ❌ | false | AND vs OR tag matching |
| `top_k` | int | ❌ | 10 | Number of date clusters to return |
| `budget_remaining` | int\|null | ❌ | null | Remaining tool calls before this call — server appends budget annotation to result. |

**Response:**
```json
{
  "type": "result",
  "message": "Date ranges in scope:\n2025-06 (42 files)\n2025-08 (18 files)\n...\n\n[Tool calls remaining: 6]",
  "data": {"tool": "get_scoped_dates", "raw": "...", "budget_after": 6}
}
```

---

**Action: `scoped_rag_search`**

Semantic search within a date/tag-scoped subset of files + conversations. `start_date`, `end_date`, and `min_tags` are **required** — this tool is designed for a targeted final retrieval step, not open-ended exploration.

**Request:**
```json
{
  "action": "scoped_rag_search",
  "query": "beach vacation photos with family",
  "start_date": "2025-06-01",
  "end_date": "2025-06-30",
  "min_tags": ["beach", "family"],
  "strict": false,
  "top_k": 5,
  "embedding_model": null
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Semantic search phrase for FAISS ranking |
| `start_date` | string | ✅ | — | Inclusive start date (YYYY-MM-DD) |
| `end_date` | string | ✅ | — | Inclusive end date (YYYY-MM-DD) |
| `min_tags` | array | ✅ | — | Tag filter (at least one tag required) |
| `strict` | bool | ❌ | false | AND vs OR tag matching |
| `top_k` | int | ❌ | 5 | Number of top results to return |
| `embedding_model` | string\|null | ❌ | config default | Override embedding model for this search |
| `budget_remaining` | int\|null | ❌ | null | Remaining tool calls before this call — server appends budget annotation to result. |

**Response:**
```json
{
  "type": "result",
  "message": "Top 5 results for 'beach vacation photos with family' (2025-06-01 → 2025-06-30, tags: beach, family):\n\n• IMG_2847.jpg\n  Date: 2025-06-15\n  Tags: beach, family, summer\n  Desc: Family playing on sandy beach...\n\n• [Conversation conv_20250610]\n  Compacted: 2025-06-20\n  Keywords: beach, vacation, malibu\n  Facts: Discussed planning a beach vacation to Malibu...\n\n[Tool calls remaining: 0 — generate your final answer now]",
  "data": {"tool": "scoped_rag_search", "raw": "...", "budget_after": 0}
}
```

The `message` string is pre-formatted and budget-annotated — use it directly as the tool result in the Cloud AI conversation. `data.raw` contains the unannotated text only.

---

**Action: `get_conversation_rag`**

Semantic search across compacted conversation summaries only (no files).

**Request:**
```json
{
  "action": "get_conversation_rag",
  "query": "trip planning for beach vacation",
  "top_n": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | — | Semantic search phrase |
| `top_n` | int | ❌ | 5 | Number of top conversation matches to return |
| `budget_remaining` | int\|null | ❌ | null | Remaining tool calls before this call — server appends budget annotation to result. |

**Response:**
```json
{
  "type": "result",
  "message": "Top 3 relevant past conversations:\n\n[Conversation conv_20250610]\nCompacted: 2025-06-20\nKeywords: beach, vacation, malibu\nFacts: Discussed planning a beach vacation to Malibu in June...\n\n[Tool calls remaining: 5]",
  "data": {"tool": "get_conversation_rag", "raw": "...", "budget_after": 5}
}
```

---

**Workflow for Cloud AI Deep Chat:**

A Cloud AI implementing Deep Chat should follow this pattern:

1. Connect to `/api/mcp`
2. Receive ready message
3. Send `get_library_context` (with the user's question)
4. Use the returned `system_prompt` directly in your LLM call, or customize it as needed
5. Run a multi-turn tool-calling loop using the 4 MCP tool actions (respecting `tool_budget`):
   - Pass `budget_remaining` (decrementing from `tool_budget`) in every tool request — the server appends the same `[Tool calls remaining: N]` annotation the local agent sees
   - Use `get_scoped_tags` / `get_scoped_dates` to explore the library scope
   - Use `scoped_rag_search` for targeted retrieval
   - Use `get_conversation_rag` to find relevant past conversations
6. After each tool call, use the `message` field (top-level in the WebSocket JSON) as the tool result in the Cloud AI conversation (already includes the budget annotation)
7. When `budget_after` reaches 0, or the LLM produces a final answer (no more tool calls), disconnect

**Status messages** are sent before each tool execution:
```json
{"type": "status", "message": "Executing MCP tool: scoped_rag_search..."}
```

---

**Error Cases:**

```json
{"type": "error", "message": "RAG database not available. Generate RAG first using /generate-rag."}
```

```json
{
  "type": "error",
  "message": "Unknown action: 'foo'. Use 'get_library_context', 'get_scoped_tags', 'get_scoped_dates', 'scoped_rag_search', or 'get_conversation_rag'."
}
```

```json
{
  "type": "error",
  "message": "Cloud Deep Chat error: <ErrorType>: <details>",
  "data": {"error_type": "...", "error_message": "...", "traceback": "..."}
}
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
11. **Conversation memory:** Use `/api/compact-conversations` followed by `/api/generate-rag` to build conversation memories into the RAG index
12. **Cloud AI workflow:** Use `/api/cloud-compact` when your client app performs summarization via a cloud LLM — the server embeds only, no local chat model required
13. **Cloud Deep Chat:** Use `/api/mcp` — it exposes the same 4 MCP tools (`get_scoped_tags`, `get_scoped_dates`, `scoped_rag_search`, `get_conversation_rag`) used by local deep-chat, so a Cloud AI can drive an identical tool-calling loop. See the [Deep Chat Cloud Build Guide](Documentation/DEEP_CHAT_CLOUD_GUIDE.md)

---

## Version

API Version: 3.1  
Documentation Last Updated: April 12, 2026
