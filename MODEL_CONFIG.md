# Model Configuration Guide

This file documents the models available in your setup and recommended configurations.

## Available Models

### Embedding Model
- **File**: `qwen3-embedding-8b-q4_k_m.gguf`
- **Purpose**: Generate embeddings for RAG
- **Usage**: Use in `/api/generate-embeddings` endpoint
- **Binary**: llama-server or llama-cli

### Chat/Instruct Model
- **File**: `Qwen3-8B-Q4_K_M.gguf`
- **Purpose**: Conversational AI and chat
- **Usage**: Use in `/api/chat` endpoint
- **Binary**: llama-server or llama-cli

### Vision Model
- **File**: `Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf`
- **Purpose**: Image/video tagging and description
- **Usage**: Use in `/api/tag` and `/api/describe` endpoints
- **Binary**: llama-mtmd-cli (multimodal support)
- **MMProj Files**:
  - `mmproj-Qwen2-VL-7B-Instruct-f16.gguf` (recommended for Qwen2.5-VL)
  - `mmproj-F16.gguf` (alternative)

## Recommended API Configurations

### For Embedding Generation
```json
{
  "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf"
}
```

### For Vision Tasks (Tagging)
```json
{
  "file_paths": ["your-image.jpg"],
  "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
  "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
}
```

### For Vision Tasks (Description)
```json
{
  "file_paths": ["your-image.jpg"],
  "vision_model": "Qwen2.5-VL-7B-Instruct-UD-IQ2_M.gguf",
  "mmproj_file": "mmproj-Qwen2-VL-7B-Instruct-f16.gguf"
}
```

### For Chat
```json
{
  "chat_model": "Qwen3-8B-Q4_K_M.gguf"
}
```

## Binary Selection

The server automatically selects the correct vision binary:
- **llama-mtmd-cli**: Used for multimodal vision models (including Qwen2.5-VL)
- **llama-cli**: Fallback for other vision models

You can override this in configuration:
```json
{
  "vision_binary": "auto"  // or "llama-mtmd-cli" or "llama-cli"
}
```

## LLM Backend Modes

The server supports two LLM backend modes via the `llm_mode` (or `backend`) configuration:

### Server Mode (`llm_mode: "server"`)
- **Uses**: `llama-server` binary (persistent process on port 8080)
- **Applied to**: **All LLM tasks** - chat, embeddings, and vision
- **Behavior**: 
  - Starts a persistent llama-server process
  - Models stay loaded between requests
  - Better performance for repeated use
  - Higher memory usage
  - Vision: Sends base64-encoded images with multimodal message format
- **LLM Parameters**: Applied at server startup (ctx_size, batch_size, ubatch_size) and per-request (temp, top_p, top_k, presence_penalty, mirostat)

### CLI Mode (`llm_mode: "cli"`)
- **Uses**: Specialized binaries per task (per-request processes)
- **Applied to**: **All LLM tasks** - chat, embeddings, and vision
- **Behavior**:
  - Spawns new process for each request
  - Models unloaded after each request
  - Lower memory footprint
  - Slower response time
  - Chat/embeddings: Uses `llama-cli`
  - Vision: Uses `llama-mtmd-cli` (multimodal binary with image file input)
- **LLM Parameters**: Applied per invocation

**Summary**: 
- `llm_mode: "server"` → **All tasks** (chat, embeddings, vision) use llama-server
- `llm_mode: "cli"` → Chat/embeddings use llama-cli, vision uses llama-mtmd-cli
- Vision in server mode: Base64-encoded images sent to llama-server API
- Vision in CLI mode: Image files passed to llama-mtmd-cli binary

## Server Configuration Recommendations

### For Best Performance
```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "image_quality": "medium",
    "llm_mode": "server",
    "top_k": 5,
    "recency_bias": 1.5,
    "reduced_embedding_size": null
  }'
```

### For Memory-Constrained Systems
```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "image_quality": "low",
    "llm_mode": "cli",
    "top_k": 3,
    "reduced_embedding_size": 512,
    "llm_params": {
      "ctx_size": 8192,
      "batch_size": 4096
    }
  }'
```

### For Maximum Quality
```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "image_quality": "high",
    "llm_mode": "server",
    "top_k": 10,
    "recency_bias": 1.0,
    "llm_params": {
      "ctx_size": 16384,
      "temp": 0.3
    }
  }'
```

## Model-Specific Notes

### Qwen3 Embedding Model
- Optimized for semantic search
- Produces high-quality embeddings
- Recommended embedding size: 768 (original) or 512 (reduced)

### Qwen3 Chat Model
- 8B parameter model, Q4_K_M quantization
- Good balance of speed and quality
- Context window: Up to 32K tokens (configured via ctx_size)

### Qwen2.5 Vision Model
- 7B parameter vision-language model
- Supports both images and videos
- Requires mmproj file for vision encoding
- IQ2_M quantization for smaller size

## Troubleshooting

### Model Not Found
```bash
# Check available models
ls -lh model/

# Verify paths in validation
python validate.py
```

### Binary Not Executable
```bash
# Make binaries executable
chmod +x binary/*
```

### Vision Model Issues
- Ensure the correct mmproj file is specified
- The server auto-detects llama-qwen2vl-cli for Qwen models
- Check image quality setting if processing fails

### Memory Issues
- Reduce ctx_size in llm_params
- Use reduced_embedding_size (512 or 384)
- Switch to llm_mode: "cli" instead of "server"
- Lower image_quality to "low" or "medium"

## Testing Your Setup

```python
# Test embedding generation
import asyncio
import websockets
import json

async def test_embeddings():
    uri = "ws://127.0.0.1:8000/api/generate-embeddings"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "embedding_model": "qwen3-embedding-8b-q4_k_m.gguf"
        }))
        async for msg in ws:
            data = json.loads(msg)
            print(data['message'])
            if data['type'] in ['result', 'error']:
                break

asyncio.run(test_embeddings())
```

## Performance Tips

1. **Use llama-server mode** for chat and frequent requests
2. **Pre-generate embeddings** before building RAG
3. **Adjust image quality** based on your use case
4. **Use reduced embeddings** if search speed is more important than precision
5. **Set appropriate ctx_size** - larger isn't always better for all tasks
