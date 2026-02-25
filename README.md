# AI Capability - AI Server Backend

A comprehensive, modular backend AI server with RAG (Retrieval Augmented Generation), Vision processing, and Chat capabilities. Built for easy deployment with PyInstaller.

## 🎯 Quick Start

### macOS/Linux

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Make binaries executable
chmod +x binary/*

# 3. Validate setup
python validate.py

# 4. Run the server
python run_server.py
```

### Windows

```powershell
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Validate setup
python validate.py

# 3. Run the server
python run_server.py
```

Server will start at: **http://127.0.0.1:8000**  
API docs available at: **http://127.0.0.1:8000/docs**

## 🖥️ Platform Support

**Cross-Platform Compatible** - Works on Windows, macOS, and Linux

- **Windows**: Use `.exe` binaries in `binary/` directory
- **macOS/Linux**: Use standard binaries (no extension) and ensure they're executable

See **[WINDOWS_SUPPORT.md](WINDOWS_SUPPORT.md)** for detailed cross-platform information.

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide for immediate usage
- **[README_AI_SERVER.md](README_AI_SERVER.md)** - Comprehensive documentation
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Technical overview and architecture
- **[MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md)** - ⭐ Model download feature guide
- **[HOT_RELOAD_GUIDE.md](HOT_RELOAD_GUIDE.md)** - 🔥 Hot reload & dynamic configuration
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation

## ✨ Features

- ✅ **Configuration Management** - Flexible REST API configuration
- ✅ **Hot Reload** - 🔥 Configuration changes without server restart
- ✅ **Model Download** - Download GGUF models from Hugging Face repositories
- ✅ **Dynamic Model Paths** - Flexible model directory management
- ✅ **RAG Database** - FAISS-based vector search with PCA reduction
- ✅ **Embedding Generation** - LLM-powered embedding creation
- ✅ **Vision Processing** - Image/video tagging and description
- ✅ **Chat with Context** - Conversational AI with RAG integration
- ✅ **Deep Chat** - Multi-round thinking with RAG function access
- ✅ **WebSocket Support** - Real-time progress and streaming
- ✅ **Modular Architecture** - Easy component replacement

## 🏗️ Project Structure

```
AI_Capability/
├── app/                      # Main application
│   ├── config/              # Configuration management
│   ├── models/              # Data models
│   ├── services/            # Business logic (LLM, RAG, Vision)
│   ├── api/                 # REST & WebSocket endpoints
│   └── utils/               # Utilities
├── binary/                   # Llama binaries (user to add)
├── model/                    # Model files (user to add)
├── venv/                     # Python virtual environment
├── run_server.py            # Server startup
├── validate.py              # Setup validation
├── build.sh                 # Build script (macOS/Linux)
├── build.ps1                # Build script (Windows PowerShell)
├── build.bat                # Build script (Windows Batch)
├── example_client.py        # Example client
├── ai_capability.spec       # PyInstaller spec (macOS/Linux)
└── ai_capability_windows.spec  # PyInstaller spec (Windows)
```

## 🚀 Building Executable

### macOS/Linux
```bash
# Using build script (recommended)
./build.sh

# Or manually with Unix spec file
pyinstaller ai_capability.spec
```

### Windows
```powershell
# Using PowerShell script (recommended)
.\build.ps1

# Or using batch script
build.bat

# Or manually with Windows spec file
pyinstaller ai_capability_windows.spec
```

### Running the Executable

**macOS/Linux:**
```bash
./dist/ai_capability_server/ai_capability_server
```

**Windows:**
```powershell
.\dist\ai_capability_server\ai_capability_server.exe
```

See **[WINDOWS_SUPPORT.md](WINDOWS_SUPPORT.md)** for detailed build instructions.

## 📋 Prerequisites

### Required

**Python & Environment:**
- Python 3.8+

**Binaries (in `binary/` folder):**

*Windows:*
- `llama-server.exe`
- `llama-cli.exe`
- `llama-mtmd-cli.exe`
- `llama-embedding.exe`

*macOS/Linux:*
- `llama-server`
- `llama-cli`
- `llama-mtmd-cli`
- `llama-embedding`

**Models (in `model/` folder):**
- Model files (.gguf)
- MMProj files for vision models

### Already Installed
All Python dependencies are installed in the virtual environment:
- FastAPI & Uvicorn
- WebSockets & aiohttp
- Pillow & OpenCV
- NumPy & scikit-learn
- FAISS (CPU version)
- Pydantic

## 🎮 API Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `/api/config` | GET/POST | Get/update configuration |
| `/api/set-storage-metadata` | POST | Set metadata JSON path |
| `/api/load-rag` | POST | Load RAG database |
| `/api/generate-embeddings` | WebSocket | Generate embeddings |
| `/api/generate-rag` | WebSocket | Build RAG database |
| `/api/tag` | WebSocket | Tag images/videos |
| `/api/describe` | WebSocket | Describe images/videos |
| `/api/chat` | WebSocket | Chat with RAG context |
| `/api/deep-chat` | WebSocket | Multi-round thinking chat |

## 🔧 Configuration

Default configuration includes:
- **Image Quality**: medium (512/1024/2048px or original)
- **LLM Mode**: server (or cli)
- **Top K**: 5 RAG results
- **Recency Bias**: 1.0 (no bias)
- **Embedding Reduction**: None (can set to 512 or other)
- **LLM Parameters**: ctx_size, temp, top_p, etc.

Update via `/api/config` endpoint or config files.

## 📖 Usage Example

```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://127.0.0.1:8000/api/chat"
    async with websockets.connect(uri) as ws:
        # Initialize
        await ws.send(json.dumps({
            "chat_model": "your-model.gguf"
        }))
        
        # Wait for ready
        while True:
            msg = json.loads(await ws.receive())
            if "ready" in msg['message'].lower():
                break
        
        # Send message
        await ws.send(json.dumps({
            "message": "What photos do I have?"
        }))
        
        # Get response
        while True:
            msg = json.loads(await ws.receive())
            if msg['type'] == 'result':
                print(msg['data']['response'])
                break

asyncio.run(chat())
```

See `example_client.py` for more examples.

## 🧪 Validation

Run validation script to check setup:

```bash
python validate.py
```

This checks:
- ✅ Python dependencies
- ✅ Application structure
- ✅ Module imports
- ✅ Required directories
- ✅ Binaries and models
- ✅ Server creation

## 🏗️ Architecture Highlights

### Modular Design
- **Abstract Interfaces**: Easy to swap Vector DB or LLM backend
- **Service Layer**: Clean separation of concerns
- **Configuration Manager**: Centralized settings with validation
- **Process Manager**: Safe external process lifecycle

### Technology Stack
- **FastAPI**: Modern async web framework
- **FAISS**: Efficient vector similarity search
- **Pillow + OpenCV**: Image/video processing
- **scikit-learn**: PCA dimensionality reduction
- **WebSockets**: Real-time bidirectional communication

## 📊 Key Components

### LLM Service
- Supports llama-server (persistent) and llama-cli (on-demand)
- Abstract backend interface for easy replacement
- Automatic model loading/unloading
- Embedding and generation support

### RAG Service
- FAISS-based vector database
- Optional PCA dimensionality reduction
- Recency bias for time-aware ranking
- Hybrid search (embedding + keywords)
- Modular VectorDB interface

### Vision Service
- Image and video support
- Quality-based preprocessing (LANCZOS)
- Tag and description generation
- Frame extraction from videos

## 🔒 Resource Management

- Automatic model cleanup after requests
- Graceful process termination
- WebSocket connection handling
- Memory-efficient streaming

## 🤝 Contributing

The modular architecture makes it easy to:
- Replace FAISS with another Vector DB
- Add new LLM backends
- Extend image processing
- Add new API endpoints

## 📝 License

[Your License Here]

## 🆘 Support

1. Check validation: `python validate.py`
2. Read documentation: `README_AI_SERVER.md`
3. Try examples: `example_client.py`
4. View API docs: http://127.0.0.1:8000/docs

## 🎉 Status

**✅ Production Ready**

All features implemented and tested:
- Configuration management
- Storage metadata handling
- Embedding generation with PCA
- FAISS-based RAG
- Vision processing (tag/describe)
- Chat with RAG context
- WebSocket real-time updates
- PyInstaller build support