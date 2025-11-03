# Quick Start Guide - AI Server

## Prerequisites Setup

1. **Create required directories:**
   ```bash
   mkdir -p binary model
   ```

2. **Add your binaries to `binary/` folder:**
   - `llama-server`
   - `llama-cli`
   - `llama-mtmd-cli` (for vision tasks)
   
   Make them executable:
   ```bash
   chmod +x binary/*
   ```

3. **Add your models to `model/` folder:**
   - Embedding model (e.g., `embedding-model.gguf`)
   - Vision model (e.g., `vision-model.gguf`)
   - Chat/Instruct model (e.g., `chat-model.gguf`)
   - MMProj files for vision models (e.g., `mmproj-model.mmproj`)

## Running the Server

### Option 1: Run Directly (Development)

```bash
# Activate virtual environment
source venv/bin/activate

# Run server
python run_server.py
```

Server starts at: `http://127.0.0.1:8000`
API docs at: `http://127.0.0.1:8000/docs`

### Option 2: Build Executable (Production)

```bash
# Activate virtual environment
source venv/bin/activate

# Build with PyInstaller
pyinstaller ai_server.spec

# Run the executable
./dist/ai_server/ai_server
```

## Basic Workflow

### 1. Set Storage Metadata Location

```bash
curl -X POST http://127.0.0.1:8000/api/set-storage-metadata \
  -H "Content-Type: application/json" \
  -d '{"path": "/absolute/path/to/your/storage-metadata.json"}'
```

This tells the server where your file metadata JSON is located. All file paths will be resolved relative to this location.

### 2. Check Configuration

```bash
curl http://127.0.0.1:8000/api/config
```

### 3. Update Configuration (Optional)

```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "image_quality": "high",
    "top_k": 10,
    "recency_bias": 1.5,
    "reduced_embedding_size": 512
  }'
```

### 4. Generate Embeddings (WebSocket)

Use the example client or WebSocket tools:

```bash
python example_client.py
```

Or programmatically:
```python
import asyncio
import websockets
import json

async def generate_embeddings():
    uri = "ws://127.0.0.1:8000/api/generate-embeddings"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "embedding_model": "your-embedding-model.gguf"
        }))
        async for msg in ws:
            data = json.loads(msg)
            print(data['message'])
            if data['type'] in ['result', 'error']:
                break

asyncio.run(generate_embeddings())
```

### 5. Build RAG Database (WebSocket)

```python
async def build_rag():
    uri = "ws://127.0.0.1:8000/api/generate-rag"
    async with websockets.connect(uri) as ws:
        async for msg in ws:
            data = json.loads(msg)
            print(data['message'])
            if data['type'] in ['result', 'error']:
                break

asyncio.run(build_rag())
```

### 6. Load RAG (For Subsequent Runs)

Once RAG is generated, you can load it directly:

```bash
curl -X POST http://127.0.0.1:8000/api/load-rag
```

### 7. Chat with RAG

```python
async def chat():
    uri = "ws://127.0.0.1:8000/api/chat"
    async with websockets.connect(uri) as ws:
        # Initialize
        await ws.send(json.dumps({
            "chat_model": "your-chat-model.gguf"
        }))
        
        # Wait for ready
        while True:
            msg = await ws.receive()
            data = json.loads(msg)
            if "ready" in data['message'].lower():
                break
        
        # Send message
        await ws.send(json.dumps({
            "message": "What photos do I have from the beach?"
        }))
        
        # Receive response
        while True:
            msg = await ws.receive()
            data = json.loads(msg)
            if data['type'] == 'result':
                print(data['data']['response'])
                break
        
        # End chat
        await ws.send(json.dumps({"action": "end"}))

asyncio.run(chat())
```

## Configuration Options Explained

### Image Quality
Controls image preprocessing before vision tasks:
- `low`: 512px max dimension - Fast, uses less memory
- `medium`: 1024px max dimension - Balanced (default)
- `high`: 2048px max dimension - Best quality, slower
- `original`: No resizing - Maximum quality, slowest

### LLM Mode
- `server`: Uses llama-server (persistent process, faster for multiple requests)
- `cli`: Uses llama-cli (spawns per request, lower memory when idle)

### Top K
Number of files to retrieve from RAG database (default: 5)

### Recency Bias
- `1.0`: No bias, purely semantic search
- `>1.0`: Favor more recent files (e.g., 2.0 strongly favors recent)

### Reduced Embedding Size
- `null`: Use full embedding dimensions
- `512` or lower: Reduce dimensions with PCA for faster search

## Troubleshooting

### "Model not found"
- Verify model files are in `model/` directory
- Check exact filename in your request

### "Binary not found"
- Verify binaries are in `binary/` directory
- Check they have execute permissions: `chmod +x binary/*`

### "Storage metadata not set"
- Call `/api/set-storage-metadata` endpoint first
- Verify the JSON file path is absolute and exists

### WebSocket connection fails
- Ensure server is running
- Check firewall settings
- Verify WebSocket URL uses `ws://` not `http://`

### Out of memory
- Reduce `ctx_size` in llm_params
- Use smaller models
- Set `reduced_embedding_size` to reduce memory usage
- Use `llm_mode: "cli"` instead of `"server"`

## Testing Endpoints

Use the interactive API docs:
```
http://127.0.0.1:8000/docs
```

This provides a Swagger UI where you can test all REST endpoints interactively.

## Next Steps

1. Check out `README_AI_SERVER.md` for detailed documentation
2. Explore `example_client.py` for more usage examples
3. Customize configuration in `/api/config` endpoint
4. Integrate with your application using the REST/WebSocket APIs

## Support

For issues, check:
1. Server logs in the console
2. Configuration via `/api/config`
3. API documentation at `/docs`
4. Detailed README at `README_AI_SERVER.md`
