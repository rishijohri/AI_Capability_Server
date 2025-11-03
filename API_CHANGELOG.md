# API Changelog

## Version 2.0 - October 25, 2025

### Breaking Changes

#### Chat WebSocket (`/api/chat`)
- **REMOVED**: Initial configuration message requirement
- **NEW**: Server automatically selects models from configuration
- Models now selected based on:
  - `chat_model` for text conversations
  - `vision_model` + `mmproj_model` for visual conversations (if `enable_visual_chat` is true)
  - `embedding_model` for RAG search

**Before (v1.0):**
```javascript
// Connect and send config
ws.send(JSON.stringify({
  chat_model: "Qwen3-8B-Q4_K_M.gguf"
}));
```

**After (v2.0):**
```javascript
// Connect and wait for ready message
// No initial config needed
```

#### Tag and Describe WebSockets (`/api/tag`, `/api/describe`)
- **NEW**: Structured XML output with thinking and conclusion sections
- **NEW**: `thinking` message type contains model's analysis process
- Responses now include:
  1. `thinking` message - Model's reasoning (optional, only if available)
  2. `result` message - Final tags/description

**Message Flow Before (v1.0):**
```
status → result
```

**Message Flow After (v2.0):**
```
status → thinking (optional) → result
```

### New Features

#### Face Recognition APIs

**POST /api/detect-faces**
- Detect and identify faces in images using InsightFace
- Automatically matches against stored face embeddings
- Creates new face IDs for unmatched faces
- Configurable similarity threshold (default: 0.5)
- Returns bounding boxes, similarity scores, and face IDs

**POST /api/get-face-crop**
- Extract face crops by face ID
- Configurable padding around face bbox
- Returns base64-encoded PNG image
- Includes similarity score and bbox information

#### Structured Response Format

All vision tasks (chat, tag, describe) now use XML-tagged responses:

**Chat Response Structure:**
```xml
<think>
Analysis and reasoning process
</think>

<conclusion>
Final answer to user's question
</conclusion>

<files>
- relevant_file1.txt
- relevant_file2.jpg
</files>
```

**Tag/Describe Response Structure:**
```xml
<think>
Image analysis process
</think>

<conclusion>
comma-separated, tags, for, image
</conclusion>
```

### Configuration Changes

#### New Configuration Fields

**`enable_visual_chat`** (boolean, default: false)
- Enable visual conversation mode
- When true, chat uses vision model for image-containing messages
- Automatically switches between chat and vision models based on message content

#### Updated Prompt Templates

All prompt templates now enforce XML format:

**`chat_system_prompt`**
- Requires `<think>`, `<conclusion>`, and `<files>` tags
- Provides structured output for better parsing

**`tag_prompt`**
- Requires `<think>` and `<conclusion>` tags
- Conclusion must contain only comma-separated keywords
- Strict rules against explanatory text

**`describe_prompt`**
- Requires `<think>` and `<conclusion>` tags
- Conclusion contains detailed description
- Structured format for consistent output

### WebSocket Message Types

#### New Message Type: `thinking`

Used by: `/api/tag`, `/api/describe`, `/api/chat`

Contains the model's analysis and reasoning process before providing the final result.

**Example:**
```json
{
  "type": "thinking",
  "message": "Analysis for image.jpg",
  "data": {
    "filename": "image.jpg",
    "thinking": "I can see several elements in this image..."
  }
}
```

### Migration Guide

#### Updating Chat Clients

**Old Code:**
```python
async with websockets.connect('ws://localhost:8000/api/chat') as ws:
    # Send initial config
    await ws.send(json.dumps({
        "chat_model": "model.gguf"
    }))
    
    # Wait for ready
    # ... rest of code
```

**New Code:**
```python
async with websockets.connect('ws://localhost:8000/api/chat') as ws:
    # No initial config needed - just wait for ready
    while True:
        message = await ws.recv()
        data = json.loads(message)
        if "ready" in data['message'].lower():
            break
    
    # ... rest of code
```

#### Parsing Structured Responses

**Old Code (Tag/Describe):**
```python
if data['type'] == 'result':
    tags = data['data']['tags']
    print(f"Tags: {', '.join(tags)}")
```

**New Code (Tag/Describe):**
```python
if data['type'] == 'thinking':
    print(f"Analysis: {data['data']['thinking']}")
elif data['type'] == 'result':
    tags = data['data']['tags']
    print(f"Tags: {', '.join(tags)}")
```

**Old Code (Chat):**
```python
if data['type'] == 'result':
    response = data['data']['response']
    print(f"Response: {response}")
```

**New Code (Chat):**
```python
thinking = ""
conclusion = ""
files = []

async for message in ws:
    data = json.loads(message)
    
    if data['type'] == 'thinking':
        thinking = data['message']
    elif data['type'] == 'conclusion':
        conclusion = data['message']
    elif data['type'] == 'files':
        files = data['data'].get('relevant_files', [])
    elif data['type'] == 'result':
        print(f"Thinking: {thinking}")
        print(f"Conclusion: {conclusion}")
        print(f"Files: {', '.join(files)}")
        break
```

### Face Recognition Integration

**Example Usage:**
```python
import requests

# 1. Detect faces in images
response = requests.post("http://localhost:8000/api/detect-faces", json={
    "file_paths": ["family_photo.jpg", "party.jpg"],
    "similarity_threshold": 0.5
})

faces = response.json()
for result in faces['results']:
    print(f"\n{result['filename']}:")
    for face in result['faces']:
        print(f"  Face ID: {face['face_id']}")
        print(f"  Similarity: {face['similarity']}")
        print(f"  New face: {face['is_new']}")

# 2. Extract face crop
response = requests.post("http://localhost:8000/api/get-face-crop", json={
    "file_path": "family_photo.jpg",
    "face_id": "face_001",
    "padding": 20
})

if response.status_code == 200:
    import base64
    from PIL import Image
    import io
    
    img_data = base64.b64decode(response.json()['image_base64'])
    img = Image.open(io.BytesIO(img_data))
    img.save("face_crop.png")
```

### Best Practices (Updated)

1. **Face Recognition Thresholds:**
   - Detection: 0.5 (balanced accuracy)
   - Cropping: 0.4 (looser to include more matches)
   - Adjust based on your use case

2. **Parsing Structured Output:**
   - Always check for `thinking` type messages
   - Parse XML tags from conclusion if needed
   - Handle fallback when XML tags are missing

3. **Model Selection:**
   - Configure models in `/api/config` before connecting to chat
   - Use `enable_visual_chat` for image-aware conversations
   - Models are automatically selected based on message content

4. **Backward Compatibility:**
   - Old chat clients that send initial config will ignore it
   - Old tag/describe clients will still receive `result` messages
   - XML parsing is optional but recommended

### Performance Improvements

- Face embeddings cached in memory after first load
- L2-normalized cosine similarity for faster face matching
- Process cleanup prevents multiple llama instances
- Auto-model selection reduces WebSocket handshake overhead

### Bug Fixes

- Fixed variable naming issue in `generate_description` method
- RAG search now properly isolates embedding model
- Process manager kills existing binaries before starting new ones
- Tag prompt enforces strict comma-separated output format

---

## Version 1.0 - October 23, 2025

Initial release with core functionality:
- Configuration management
- Embedding generation
- RAG database creation
- Tag and describe for media
- Chat with RAG context
- Server shutdown endpoint
