# AI Server

A comprehensive backend AI server with RAG (Retrieval Augmented Generation), Vision, and Chat capabilities. Built with modular architecture for easy component replacement.

## Features

- **Configuration Management**: Flexible server configuration with REST API
- **Storage Metadata Management**: Link to file metadata JSON for RAG operations
- **Embedding Generation**: Generate embeddings from file metadata using LLM models
- **RAG Database**: FAISS-based vector database with PCA dimensionality reduction
- **Vision Processing**: Image/video tagging and description using Vision LLMs
- **Chat with RAG**: Conversational AI with context from file database
- **WebSocket Support**: Real-time progress updates and interactive operations
- **Modular Architecture**: Easy to swap Vector DB, LLM backends, or other components

## Project Structure

```
AI_Capability/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── config/                 # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── models/                 # Data models
│   │   ├── __init__.py
│   │   ├── metadata.py
│   │   ├── requests.py
│   │   └── responses.py
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── llm_service.py      # LLM abstraction (server/cli)
│   │   ├── embedding_service.py
│   │   ├── rag_service.py      # FAISS-based RAG
│   │   └── vision_service.py
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   └── routes.py
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── image_processor.py
│       └── process_manager.py
├── binary/                     # Llama binaries (llama-server, llama-cli, llama-mtmd-cli)
├── model/                      # Model files (.gguf, .mmproj)
├── venv/                       # Python virtual environment
├── requirements.txt            # Python dependencies
├── ai_server.spec             # PyInstaller spec file
├── run_server.py              # Server startup script
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8+
- llama binaries: `llama-server`, `llama-cli`, `llama-mtmd-cli` in `binary/` folder
- Model files (.gguf) in `model/` folder
- MMProj files for vision models in `model/` folder

### Setup

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare directories:**
   ```bash
   mkdir -p binary model
   # Place your llama binaries in binary/
   # Place your model files in model/
   ```

## Running the Server

### Development Mode

```bash
python run_server.py
```

Or directly:

```bash
python -m app.main
```

Server will start at `http://127.0.0.1:8000`

### Building Standalone Executable

```bash
pyinstaller ai_server.spec
```

The executable will be created in `dist/ai_server/` folder.

Run the executable:
```bash
./dist/ai_server/ai_server
```

## API Endpoints

### REST Endpoints

#### GET /api/config
Get current server configuration.

**Response:**
```json
{
  "reduced_embedding_size": 512,
  "chat_rounds": 3,
  "image_quality": "medium",
  "llm_mode": "server",
  "top_k": 5,
  "recency_bias": 1.5,
  "llm_params": {
    "ctx_size": 12192,
    "temp": 0.35,
    "top_p": 0.9,
    "top_k": 40,
    "presence_penalty": 0.2,
    "mirostat": 0,
    "batch_size": 8192,
    "ubatch_size": 1024
  },
  "rag_directory_name": "rag",
  "storage_metadata_path": "/path/to/storage-metadata.json"
}
```

#### POST /api/config
Update server configuration. Only editable fields can be updated.

**Request:**
```json
{
  "reduced_embedding_size": 512,
  "image_quality": "high",
  "top_k": 10,
  "recency_bias": 2.0,
  "chat_model": "Qwen3-8B-Q4_K_M.gguf",
  "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf",
  "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf"
}
```

**Response:** Same as GET /config

#### POST /api/set-storage-metadata
Set the storage metadata JSON file location. This must be called before using any other endpoints.

**Request:**
```json
{
  "path": "/path/to/storage-metadata.json"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Storage metadata set",
  "data": {
    "metadata_count": 100,
    "rag_directory": "/path/to/rag"
  }
}
```

#### POST /api/load-rag
Load existing RAG database from disk. The RAG directory is automatically determined from the storage metadata path.

**Response:**
```json
{
  "status": "success",
  "message": "RAG database loaded successfully"
}
```

### WebSocket Endpoints

All WebSocket endpoints use JSON messages with the following structure:

```json
{
  "type": "status|progress|result|error|confirmation_needed",
  "message": "Human-readable message",
  "data": {}  // Optional additional data
}
```

#### WS /generate-embeddings

Generate embeddings for all files in the storage metadata. The embeddings are stored in memory and used for RAG operations.

**Connection:** `ws://localhost:8000/generate-embeddings`

**Client → Server (Initial Message):**
```json
{
  "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf"
}
```

**Server → Client (Progress Updates):**
```json
{
  "type": "progress",
  "message": "Processing example.jpg",
  "data": {
    "current": 5,
    "total": 100,
    "filename": "example.jpg"
  }
}
```

**Server → Client (Completion):**
```json
{
  "type": "result",
  "message": "Embeddings generated successfully",
  "data": {
    "count": 100
  }
}
```

**Server → Client (Error):**
```json
{
  "type": "error",
  "message": "Model file not found: model.gguf"
}
```

#### WS /api/generate-rag

Build the RAG (Retrieval Augmented Generation) database from previously generated embeddings. This creates a FAISS index for fast similarity search.

**Connection:** `ws://localhost:8000/api/generate-rag`

**Client → Server:** No initial message required, connects and starts immediately

**Server → Client (Status Updates):**
```json
{
  "type": "status",
  "message": "Building RAG database..."
}
```

**Server → Client (Completion):**
```json
{
  "type": "result",
  "message": "RAG database created and loaded successfully"
}
```

**Server → Client (Error):**
```json
{
  "type": "error",
  "message": "No embeddings available. Generate embeddings first."
}
```

**Note:** RAG database is automatically saved to the RAG directory alongside the metadata file.

#### WS /api/tag

Generate AI tags for images and videos. Tags are automatically saved to the metadata file.

**Connection:** `ws://localhost:8000/api/tag`

**Client → Server (Initial Message):**
```json
{
  "file_paths": ["beach.jpg", "birthday.mp4"],
  "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
  "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
}
```

**Note:** `file_paths` contains filenames only (as they appear in metadata), not absolute paths.

**Server → Client (Confirmation Request):**
```json
{
  "type": "confirmation_needed",
  "message": "Ready to tag beach.jpg. Send 'continue' to proceed.",
  "data": {
    "current": 1,
    "total": 2
  }
}
```

**Client → Server (Confirmation):**
```json
{
  "action": "continue"
}
```

**Server → Client (Progress):**
```json
{
  "type": "status",
  "message": "Generating tags for vacation/beach.jpg..."
}
```

**Server → Client (Result):**
```json
{
  "type": "result",
  "message": "Tags generated for beach.jpg",
  "data": {
    "filename": "beach.jpg",
    "tags": ["beach", "ocean", "sunset", "vacation"]
  }
}
```

**Server → Client (Completion):**
```json
{
  "type": "status",
  "message": "Tagging complete"
}
```

#### WS /api/describe

Generate detailed AI descriptions for images and videos. Descriptions are automatically saved to the metadata file.

**Connection:** `ws://localhost:8000/api/describe`

**Client → Server (Initial Message):**
```json
{
  "file_paths": ["sunset.jpg", "concert.mp4"],
  "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
  "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
}
```

**Note:** `file_paths` contains filenames only (as they appear in metadata), not absolute paths.

**Server → Client (Confirmation Request):**
```json
{
  "type": "confirmation_needed",
  "message": "Ready to describe sunset.jpg. Send 'continue' to proceed.",
  "data": {
    "current": 2,
    "total": 2
  }
}
```

**Client → Server (Confirmation):**
```json
{
  "action": "continue"
}
```

**Server → Client (Status):**
```json
{
  "type": "status",
  "message": "Generating description for photos/sunset.jpg..."
}
```

**Server → Client (Result):**
```json
{
  "type": "result",
  "message": "Description generated for sunset.jpg",
  "data": {
    "filename": "sunset.jpg",
    "description": "A beautiful sunset over the ocean with orange and pink hues in the sky. The sun is setting on the horizon, reflecting on the calm water."
  }
}
```

**Server → Client (Completion):**
```json
{
  "type": "status",
  "message": "Description generation complete"
}
```

#### WS /api/chat

Interactive chat with RAG context. The chat uses the RAG database to provide context-aware responses based on your file collection.

**Connection:** `ws://localhost:8000/api/chat`

**Client → Server (Initial Message):**
```json
{
  "chat_model": "Qwen3-8B-Q4_K_M.gguf"
}
```

**Server → Client (Status Updates):**
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

**Client → Server (User Message):**
```json
{
  "message": "What photos do I have from the beach?"
}
```

**Server → Client (Search Status):**
```json
{
  "type": "status",
  "message": "Searching knowledge base..."
}
```

**Server → Client (Generating Response):**
```json
{
  "type": "status",
  "message": "Generating response..."
}
```

**Server → Client (Response):**
```json
{
  "type": "result",
  "message": "Response generated",
  "data": {
    "response": "Based on your collection, you have 12 beach photos including sunset scenes, family gatherings, and surfing activities from your summer vacation.",
    "context": [
      {
        "filename": "vacation/beach_sunset.jpg",
        "score": 0.89,
        "tags": ["beach", "sunset", "vacation"]
      },
      {
        "filename": "summer/surfing.jpg",
        "score": 0.85,
        "tags": ["beach", "surfing", "ocean"]
      }
    ]
  }
}
```

**Client → Server (End Chat):**
```json
{
  "action": "end"
}
```

**Server → Client (Farewell):**
```json
{
  "type": "status",
  "message": "Chat session ended"
}
```

**Note:** Chat maintains conversation history for context-aware multi-turn conversations. The number of rounds is controlled by the `chat_rounds` configuration parameter.

## Configuration Options

### Editable Configuration

- **reduced_embedding_size**: Target embedding dimension after PCA (null = no reduction)
- **chat_rounds**: Number of conversation rounds (1-10)
- **image_quality**: Image scale multiplier (0.0-1.0)
  - `1.0`: Original dimensions (no scaling)
  - `0.75`: 75% of original dimensions
  - `0.5`: 50% of original dimensions (half size)
  - `0.25`: 25% of original dimensions
- **llm_mode**: LLM backend selection
  - `server`: Use llama-server (persistent process, faster)
  - `cli`: Use llama-cli (run per request)
- **top_k**: Number of RAG retrieval results (1-50)
- **recency_bias**: Favor recent files in search (1.0 = no bias, >1.0 = favor recent)
- **llm_params**: LLM execution parameters
  - ctx_size, temp, top_p, top_k, presence_penalty, mirostat, batch_size, ubatch_size

### Read-Only Configuration

- **rag_directory_name**: RAG directory name (default: "rag")
- **storage_metadata_path**: Current metadata file location

## Architecture & Modularity

The application is designed for easy component replacement:

### Replacing Vector Database

Edit `app/services/rag_service.py`:
1. Implement a new class inheriting from `VectorDB` abstract base class
2. Implement required methods: `add_vectors`, `search`, `save`, `load`
3. Update `RAGService` to use your new implementation

### Replacing LLM Backend

Edit `app/services/llm_service.py`:
1. Implement a new class inheriting from `LLMBackend` abstract base class
2. Implement required methods: `start`, `stop`, `generate`, `embed`, `is_running`
3. Add your backend to `LLMService`

### Adding New Models

1. Place model files in `model/` directory
2. Place binaries in `binary/` directory
3. Update configuration to use new models

## File Metadata Format

The server expects metadata in the following JSON format:

```json
[
  {
    "fileName": "example.jpg",
    "deviceId": "device-123",
    "deviceName": "Device Name",
    "uploadTime": "2025-10-04T18:15:26.445815",
    "tags": ["tag1", "tag2"],
    "aspectRatio": 1.5,
    "type": "image",
    "aiModel": "model-name",
    "creationTime": "2025-09-04T18:14:25.000",
    "description": "Description text",
    "descriptionState": "AI"
  }
]
```

## Examples

### Python Client Example

```python
import asyncio
import websockets
import json

async def generate_embeddings():
    uri = "ws://127.0.0.1:8000/api/generate-embeddings"
    
    async with websockets.connect(uri) as websocket:
        # Send configuration
        await websocket.send(json.dumps({
            "embedding_model": "embedding-model.gguf"
        }))
        
        # Receive progress updates
        async for message in websocket:
            data = json.loads(message)
            print(f"{data['type']}: {data['message']}")
            
            if data['type'] == 'result':
                break

asyncio.run(generate_embeddings())
```

### cURL Examples

```bash
# Get configuration
curl http://127.0.0.1:8000/api/config

# Update configuration
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"image_quality": "high", "top_k": 10}'

# Set storage metadata
curl -X POST http://127.0.0.1:8000/api/set-storage-metadata \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/metadata.json"}'

# Load RAG
curl -X POST http://127.0.0.1:8000/api/load-rag
```

## Troubleshooting

### Models Not Found
- Verify model files are in `model/` directory
- Check file permissions

### Binaries Not Executing
- Ensure binaries have execute permissions: `chmod +x binary/*`
- Verify binaries are compatible with your system architecture

### FAISS Import Error
- Install FAISS: `pip install faiss-cpu` (or `faiss-gpu` for GPU support)

### Memory Issues
- Reduce `ctx_size` in LLM parameters
- Use smaller models
- Reduce embedding dimensions with `reduced_embedding_size`

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black app/
flake8 app/
mypy app/
```

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

## Support

For issues and questions, please open an issue on the repository.
