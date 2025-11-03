# Usage Examples with Your Models

This guide provides specific examples for using the AI server with your installed models.

## Available Models

- **Chat**: `Qwen3-8B-Q4_K_M.gguf` (4.79 GB)
- **Embeddings**: `qwen3-embedding-8b-q4_k_m.gguf` (4.46 GB)
- **Vision**: `Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf` (2.72 GB)
- **MMProj**: `mmproj-Qwen2-VL-7B-Instruct-f16.gguf` (1.29 GB)

## 1. Start the Server

```bash
python run_server.py
```

The server will start on `http://localhost:8000`

## 2. Configure the Server

```bash
# View current configuration
curl http://localhost:8000/api/config

# Update configuration (example)
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "chat_model": "Qwen3-8B-Q4_K_M.gguf",
    "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf",
    "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
    "mmproj_model": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf",
    "image_quality": 0.75,
    "backend": "server"
  }'
```

## 3. Set Storage Metadata Path

```bash
curl -X POST http://localhost:8000/api/set-storage-metadata \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/your/metadata.json"}'
```

## 4. Generate Embeddings (WebSocket)

```python
import asyncio
import websockets
import json

async def generate_embeddings():
    uri = "ws://localhost:8000/api/generate-embeddings"
    async with websockets.connect(uri) as websocket:
        request = {
            "files": [
                {"path": "/path/to/document1.txt", "description": "First document"},
                {"path": "/path/to/document2.txt", "description": "Second document"}
            ],
            "dimension": 512  # Optional: reduce from 4096 to 512
        }
        await websocket.send(json.dumps(request))
        
        async for message in websocket:
            response = json.loads(message)
            if response["type"] == "progress":
                print(f"Progress: {response['current']}/{response['total']} - {response['message']}")
            elif response["type"] == "complete":
                print(f"Done! {response['success']} successful, {response['failed']} failed")
                break
            elif response["type"] == "error":
                print(f"Error: {response['message']}")
                break

asyncio.run(generate_embeddings())
```

## 5. Load RAG System

```bash
curl -X POST http://localhost:8000/api/load-rag \
  -H "Content-Type: application/json" \
  -d '{
    "index_path": "/path/to/your/faiss.index",
    "metadata_path": "/path/to/your/metadata.json"
  }'
```

## 6. Generate RAG Response (WebSocket)

```python
import asyncio
import websockets
import json

async def generate_rag():
    uri = "ws://localhost:8000/api/generate-rag"
    async with websockets.connect(uri) as websocket:
        request = {
            "query": "What is machine learning?",
            "top_k": 5,
            "recency_bias": 0.1,
            "keywords": ["ML", "AI"]
        }
        await websocket.send(json.dumps(request))
        
        response = await websocket.recv()
        data = json.loads(response)
        
        if data["type"] == "complete":
            print("Relevant documents:")
            for doc in data["results"]:
                print(f"- {doc['path']} (score: {doc['score']:.3f})")
        elif data["type"] == "error":
            print(f"Error: {data['message']}")

asyncio.run(generate_rag())
```

## 7. Tag Images/Videos (WebSocket)

```python
import asyncio
import websockets
import json

async def tag_media():
    uri = "ws://localhost:8000/api/tag"
    async with websockets.connect(uri) as websocket:
        request = {
            "file_paths": ["image1.jpg", "video1.mp4"],
            "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
            "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
        }
        await websocket.send(json.dumps(request))
        
        async for message in websocket:
            response = json.loads(message)
            if response["type"] == "status":
                print(response["message"])
            elif response["type"] == "result":
                print(f"Tags: {response['data']['tags']}")
            elif response["type"] == "confirmation_needed":
                # Send continue for next file
                await websocket.send(json.dumps({"action": "continue"}))
            elif response["type"] == "error":
                print(f"Error: {response['message']}")
                if "data" in response:
                    print(f"Error type: {response['data']['error_type']}")
                    print(f"Traceback: {response['data']['traceback']}")
                break

asyncio.run(tag_media())
```

**Note:** Use filenames only (as they appear in metadata), not absolute paths.

## 8. Describe Images/Videos (WebSocket)

```python
import asyncio
import websockets
import json

async def describe_media():
    uri = "ws://localhost:8000/api/describe"
    async with websockets.connect(uri) as websocket:
        request = {
            "file_paths": ["image1.jpg", "video1.mp4"],
            "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
            "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
        }
        await websocket.send(json.dumps(request))
        
        async for message in websocket:
            response = json.loads(message)
            if response["type"] == "status":
                print(response["message"])
            elif response["type"] == "result":
                print(f"\nDescription: {response['data']['description']}")
            elif response["type"] == "confirmation_needed":
                # Send continue for next file
                await websocket.send(json.dumps({"action": "continue"}))
            elif response["type"] == "error":
                print(f"Error: {response['message']}")
                if "data" in response:
                    print(f"Error type: {response['data']['error_type']}")
                    print(f"Traceback: {response['data']['traceback']}")
                break

asyncio.run(describe_media())
```

**Note:** Use filenames only (as they appear in metadata), not absolute paths.
        }
        await websocket.send(json.dumps(request))
        
        async for message in websocket:
            response = json.loads(message)
            if response["type"] == "progress":
                print(f"\n{response['file']}:")
                print(response['description'])
            elif response["type"] == "complete":
                print(f"\nAll done! {response['success']} files described")
                break
            elif response["type"] == "error":
                print(f"Error: {response['message']}")
                break

asyncio.run(describe_media())
```

## 9. Chat with RAG Context (WebSocket)

```python
import asyncio
import websockets
import json

async def chat_with_rag():
    uri = "ws://localhost:8000/api/chat"
    async with websockets.connect(uri) as websocket:
        request = {
            "query": "Explain neural networks based on my documents",
            "top_k": 3,
            "recency_bias": 0.1,
            "keywords": ["neural", "network"]
        }
        await websocket.send(json.dumps(request))
        
        response = await websocket.recv()
        data = json.loads(response)
        
        if data["type"] == "complete":
            print("Response:", data["response"])
            print("\nSources:")
            for doc in data["context"]:
                print(f"- {doc['path']}")
        elif data["type"] == "error":
            print(f"Error: {data['message']}")

asyncio.run(chat_with_rag())
```

## Binary Selection

The server automatically selects the correct binary for vision tasks:
- **Qwen2.5-VL** models → Uses `llama-qwen2vl-cli`
- **Other vision models** → Uses `llama-mtmd-cli`

## Tips

1. **First Run**: The first API call will load the model into memory (may take 10-30 seconds)
2. **Dimension Reduction**: Use `dimension: 512` for embeddings to reduce memory by 87%
3. **Image Quality**: Set `image_quality` to scale images:
   - `1.0` = Original dimensions (best quality, most memory)
   - `0.5` = Half size (good balance)
   - `0.25` = Quarter size (fastest, least memory)
4. **Recency Bias**: Values between 0.0-1.0; higher = prefer newer documents
5. **Keywords**: Boost specific documents containing these terms
6. **MMProj**: Required for vision models to process images

## Performance Notes

- **Chat Model**: ~4.8GB RAM when loaded
- **Embedding Model**: ~4.5GB RAM when loaded  
- **Vision Model**: ~2.7GB RAM + 1.3GB MMProj when loaded
- Models auto-unload after 300 seconds of inactivity (configurable via `model_timeout`)

## Troubleshooting

**Model not loading**: Check `model/` directory contains the .gguf files
**Vision failing**: Ensure mmproj file matches your vision model
**Out of memory**: Reduce batch sizes or use dimension reduction for embeddings
**Slow responses**: First call loads model; subsequent calls are faster

For more details, see `MODEL_CONFIG.md` and `README_AI_SERVER.md`.
