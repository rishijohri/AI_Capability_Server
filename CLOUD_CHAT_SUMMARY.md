# Cloud Chat API Endpoint - Summary

## What Was Added

A new WebSocket API endpoint `/api/cloud-chat` that enables integration with external cloud LLM providers (OpenAI, Anthropic, Google Gemini, etc.) while leveraging the AI Server's local RAG capabilities.

## Key Features

### 1. **RAG Search Without Local Inference**
   - Server performs RAG search using local embeddings
   - Returns structured context without running any LLM
   - Client handles all LLM inference via cloud APIs

### 2. **Complete Context Package**
   The endpoint returns:
   - `system_prompt`: Configured chat system prompt
   - `rag_context`: Formatted text with relevant file information
   - `relevant_files`: List of matching filenames
   - `file_details`: Similarity scores for each file
   - `image_context`: Image metadata and base64 data (if image provided)
   - `user_message`: Echo of the user's message
   - `history`: Echo of conversation history

### 3. **Conversation History Support**
   - Include previous conversation turns for context-aware RAG search
   - History is formatted into embedding query for better retrieval
   - Client manages conversation state across connections

### 4. **Image Context Support**
   - Optionally include an image filename
   - Server loads image tags, description, and base64-encoded data
   - Useful for multimodal cloud models (GPT-4 Vision, Claude 3, Gemini Pro Vision)

## Implementation Details

### Files Modified

1. **`app/api/routes.py`**
   - Added `/cloud-chat` WebSocket endpoint (line 1507)
   - Performs RAG search using embedding model
   - Returns structured context package
   - Handles connection lifecycle and error cases

2. **`API_REFERENCE.md`**
   - Added endpoint to overview table
   - Added complete documentation section with examples
   - Added cross-reference from `/api/chat` endpoint
   - Added cloud-chat example to complete workflow section

### Files Created

1. **`test_cloud_chat.py`**
   - Test script to verify endpoint functionality
   - Tests connection, RAG search, and response format
   - Provides helpful error messages for setup requirements

2. **`CLOUD_CHAT_EXAMPLES.md`**
   - Comprehensive integration examples
   - Examples for OpenAI, Anthropic, Google Gemini
   - Multi-turn conversation patterns
   - Error handling and best practices
   - Security and performance considerations

## API Endpoint Summary

### Endpoint: `WS /api/cloud-chat`

**Request Format:**
```json
{
  "message": "What photos do I have?",
  "history": [],  // Optional conversation history
  "image_name": "photo.jpg"  // Optional image context
}
```

**Response Format:**
```json
{
  "type": "result",
  "message": "RAG context retrieved successfully",
  "data": {
    "system_prompt": "...",
    "rag_context": "...",
    "relevant_files": ["file1.jpg", "file2.jpg"],
    "file_details": [...],
    "image_context": {...},
    "user_message": "...",
    "history": [...]
  }
}
```

## Use Cases

### When to Use `/api/cloud-chat`
- ✅ Need GPT-4, Claude, or other cloud model capabilities
- ✅ Want better response quality than local models
- ✅ Building production apps with commercial cloud APIs
- ✅ Need features specific to cloud models (longer context, better reasoning)
- ✅ Keep file indexing local but use cloud for inference

### When to Use `/api/chat` (Regular Chat)
- ✅ Need fully local, private operation
- ✅ No internet connection required
- ✅ No API costs
- ✅ Lower latency (no cloud API roundtrip)

## Testing

To test the new endpoint:

```bash
# Make sure server is running with RAG database configured
python test_cloud_chat.py
```

Prerequisites:
1. Storage metadata must be set
2. Embeddings must be generated
3. RAG database must be built

## Benefits

1. **Best of Both Worlds**
   - Local RAG for privacy-sensitive file indexing
   - Cloud LLM for high-quality responses

2. **Flexibility**
   - Use any cloud LLM provider
   - Switch providers without changing RAG setup
   - Mix and match based on use case

3. **Cost-Effective**
   - Only pay for cloud LLM inference
   - Local RAG search is free
   - Optimize costs by choosing appropriate models

4. **Performance**
   - Fast local embedding search
   - Leverage cloud provider's optimized inference
   - No need to load large models locally

## Example Integration

```python
import asyncio
import websockets
import json
from openai import OpenAI

async def chat_with_cloud_llm(question: str):
    # Get RAG context from AI Server
    async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if "ready" in data['message'].lower():
                break
        
        await ws.send(json.dumps({"message": question}))
        
        async for msg in ws:
            data = json.loads(msg)
            if data['type'] == 'result':
                context = data['data']
                break
    
    # Call cloud LLM with context
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": context['system_prompt']},
            {"role": "system", "content": f"Context:\n{context['rag_context']}"},
            {"role": "user", "content": question}
        ]
    )
    
    return response.choices[0].message.content

# Usage
response = asyncio.run(chat_with_cloud_llm("What photos do I have?"))
print(response)
```

## Total API Endpoints

The AI Server now has **15 API endpoints**:

### REST API (5)
1. GET `/api/config` - Get configuration
2. POST `/api/config` - Update configuration
3. GET `/api/available-models` - Get available models
4. POST `/api/set-storage-metadata` - Set metadata path
5. POST `/api/load-rag` - Load RAG database

### REST API - Operations (4)
6. POST `/api/kill` - Shutdown server
7. POST `/api/detect-faces` - Detect faces
8. POST `/api/get-face-crop` - Get face crop
9. POST `/api/rename-face-id` - Rename face ID

### WebSocket API (6)
10. WS `/api/vector-embeddings` - Generate embeddings
11. WS `/api/generate-rag` - Build RAG database
12. WS `/api/tag` - Generate tags
13. WS `/api/describe` - Generate descriptions
14. WS `/api/chat` - Local LLM chat
15. WS `/api/cloud-chat` - Cloud LLM chat (NEW)

## Documentation

Complete documentation available in:
- **API_REFERENCE.md** - Full API documentation including cloud-chat endpoint
- **CLOUD_CHAT_EXAMPLES.md** - Practical integration examples with major cloud providers
- **test_cloud_chat.py** - Test script with verification

## Verification

All code has been verified to compile without errors:
- ✅ `app/api/routes.py` compiles successfully
- ✅ `test_cloud_chat.py` compiles successfully
- ✅ All 15 endpoints accounted for in documentation
- ✅ Endpoints overview table updated
- ✅ Complete workflow examples updated
