# LLM Backend Implementation Summary

## Overview

This document summarizes how the AI Server handles LLM operations across different backends and ensures all default LLM parameters are consistently applied.

## Backend Architecture

### 1. Backend Modes

The server supports two LLM backend modes controlled by `llm_mode` (or `backend`) configuration:

#### Server Mode (`llm_mode: "server"`)
- **Binary**: `llama-server` (persistent process on port 8080)
- **Applied to**: Chat and embedding tasks
- **Advantages**:
  - Models stay loaded between requests
  - Better performance for repeated use
  - Single persistent process
- **Disadvantages**:
  - Higher memory usage (models stay loaded)

#### CLI Mode (`llm_mode: "cli"`)
- **Binary**: `llama-cli` (per-request processes)
- **Applied to**: Chat and embedding tasks  
- **Advantages**:
  - Lower memory footprint
  - Models unloaded after each request
- **Disadvantages**:
  - Slower response time (model loading overhead)
  - New process spawned per request

### 2. Vision Tasks (Now Fully Integrated)

**Vision tasks (tag/describe) now respect the `llm_mode` setting:**

#### Server Mode (`llm_mode: "server"`)
- Uses `llama-server` with base64-encoded images
- Images sent via multimodal message format:
  ```json
  {
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "prompt"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }],
    "slot_id": -1
  }
  ```
- Ephemeral slot (`slot_id: -1`) avoids cache storage

#### CLI Mode (`llm_mode: "cli"`)
- Uses specialized vision binaries: `llama-mtmd-cli` (multimodal) or `llama-cli` (fallback)
- Images passed as file paths to binary
- Temporary file created for image processing

**Advantage**: Vision tasks now benefit from persistent llama-server when in server mode, improving performance for repeated vision operations.

## LLM Parameters Applied

All LLM operations use the following default parameters (configurable via `/api/config`):

| Parameter | Default Value | Applied To | Description |
|-----------|---------------|------------|-------------|
| `ctx_size` | 12192 | All | Context window size |
| `temp` | 0.35 | Chat, Vision | Sampling temperature |
| `top_p` | 0.9 | Chat, Vision | Nucleus sampling |
| `top_k` | 40 | Chat, Vision | Top-K sampling |
| `presence_penalty` | 0.2 | Chat, Vision | Presence penalty |
| `mirostat` | 0 | Chat, Vision | Mirostat sampling mode |
| `batch_size` | 8192 | All | Batch processing size |
| `ubatch_size` | 1024 | All | Micro-batch size |

### Parameter Application by Backend

#### LlamaServerBackend (server mode)
- **Startup**: Applies `ctx_size`, `batch_size`, `ubatch_size` when starting llama-server
- **Per-request**: Applies `temp`, `top_p`, `top_k`, `presence_penalty`, `mirostat` via API payload
- **Embeddings**: No sampling parameters (deterministic)
- **Vision**: Applies sampling parameters via API payload with base64-encoded images

#### LlamaCLIBackend (cli mode)
- **Per-invocation**: Applies ALL parameters (ctx_size, temp, top_p, top_k, presence_penalty, mirostat, batch_size, ubatch_size) as command-line arguments
- **Embeddings**: Only essential parameters (no sampling)
- **Vision**: Uses specialized binaries (llama-mtmd-cli) with all LLM parameters applied

## Task Routing Summary

| Task | Backend="server" | Backend="cli" | Notes |
|------|-----------------|---------------|-------|
| **Chat** | llama-server | llama-cli | Routes through LLMService |
| **Generate Embeddings** | llama-server | llama-cli | Routes through LLMService |
| **Tag (Vision)** | llama-server | llama-mtmd-cli | **Now respects backend setting!** |
| **Describe (Vision)** | llama-server | llama-mtmd-cli | **Now respects backend setting!** |

### Vision Task Details by Backend

**Server Mode:**
- Vision model loaded into llama-server
- Images converted to base64 and sent via multimodal API
- Persistent process improves performance
- Uses ephemeral slots to avoid cache bloat

**CLI Mode:**
- Uses `llama-mtmd-cli` binary for multimodal models
- Images saved as temporary files
- Binary invoked with image file path
- Process terminates after response

## Implementation Details

### Services Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      API Endpoints                           │
│  /api/chat, /api/generate-embeddings, /api/tag, /api/describe│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
         ┌──────────────────┴──────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐
│   LLMService    │                   │  VisionService  │
│  (abstraction)  │                   │   (direct)      │
└─────────────────┘                   └─────────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐                   ┌─────────────────┐
│ Backend Factory │                   │ Vision Binaries │
│  (server/cli)   │                   │  qwen2vl/mtmd   │
└─────────────────┘                   └─────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Server │ │  CLI   │
│Backend │ │Backend │
└────────┘ └────────┘
```

### Code Locations

- **LLM Service**: `app/services/llm_service.py`
  - `LLMBackend` (abstract base)
  - `LlamaServerBackend` (server mode implementation)
  - `LlamaCLIBackend` (cli mode implementation)

- **Vision Service**: `app/services/vision_service.py`
  - `_call_vision_model()` - Directly calls vision binaries with LLM params

- **Embedding Service**: `app/services/embedding_service.py`
  - Routes through `get_llm_service()` for backend selection

- **Chat Endpoint**: `app/api/routes.py`
  - Routes through `get_llm_service()` for backend selection

## Configuration

### View Current Settings

```bash
curl http://localhost:8000/api/config
```

### Change Backend Mode

```bash
# Switch to server mode (persistent llama-server)
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"backend": "server"}'

# Switch to CLI mode (per-request llama-cli)
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"backend": "cli"}'
```

### Modify LLM Parameters

```bash
curl -X POST http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "llm_params": {
      "ctx_size": 16384,
      "temp": 0.7,
      "top_p": 0.95,
      "top_k": 50,
      "presence_penalty": 0.1,
      "mirostat": 2,
      "batch_size": 10240,
      "ubatch_size": 2048
    }
  }'
```

## Verification

All LLM parameters are verified to be applied:

- ✅ **ctx_size** (12192): Applied to server startup, CLI invocations, vision binaries
- ✅ **temp** (0.35): Applied to server requests, CLI invocations, vision binaries
- ✅ **top_p** (0.9): Applied to server requests, CLI invocations, vision binaries
- ✅ **top_k** (40): Applied to server requests, CLI invocations, vision binaries
- ✅ **presence_penalty** (0.2): Applied to server requests, CLI invocations, vision binaries
- ✅ **mirostat** (0): Applied to server requests, CLI invocations, vision binaries
- ✅ **batch_size** (8192): Applied to server startup, CLI invocations, vision binaries
- ✅ **ubatch_size** (1024): Applied to server startup, CLI invocations, vision binaries

## Summary

✅ **Backend Consistency**: **ALL** LLM tasks (chat, embeddings, and vision) now respect `llm_mode` setting

✅ **Vision Integration**: Vision tasks can now use llama-server (with base64 images) or specialized binaries depending on backend mode

✅ **Parameter Coverage**: ALL default LLM parameters are applied across all operations (chat, embeddings, vision)

✅ **Configuration Flexibility**: Backend and parameters can be changed via `/api/config` endpoint

✅ **Performance Options**: 
- Server mode: Better for repeated operations (persistent process)
- CLI mode: Better for memory-constrained systems (lower footprint)
