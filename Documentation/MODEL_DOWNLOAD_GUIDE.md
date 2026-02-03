# Model Download Feature

## Overview

The AI Capability server now includes a model download endpoint that allows you to download GGUF models from Hugging Face repositories directly to your model directory. This works seamlessly with PyInstaller-packaged servers on both macOS and Windows.

## Features

- **WebSocket-based**: Real-time progress updates during download
- **Batch downloads**: Download multiple models in a single request
- **Resume support**: Automatically resumes interrupted downloads
- **Skip existing**: Optionally skip models that are already downloaded
- **PyInstaller compatible**: Works correctly when packaged as a standalone executable
- **Vision model support**: Downloads both model files and mmproj files for vision models

## Setup

### 1. Install Dependencies

```bash
pip install huggingface_hub>=0.20.0
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Model Repository IDs

Edit `app/config/settings.py` and add `repo_id` fields to the models you want to download:

```python
model_options = {
    "qwen_3_0.6B": {
        "model_file": "Qwen3-0.6B-Q4_K_M.gguf",
        "name": "qwen_3_0.6B",
        "type": "chat",
        "repo_id": "Qwen/Qwen3-0.6B-GGUF"  # Add Hugging Face repo ID
    },
    "gemma3_4b_q4_k_m": {
        "name": "gemma3_4b_q4_k_m",
        "type": "vision",
        "model_file": "gemma-3-4b-it-Q4_K_M.gguf",
        "mmproj_file": "gemma_3_mmproj-F16.gguf",
        "repo_id": "google/gemma-3-4b-gguf"  # Add repo ID
    }
}
```

### 3. Finding Repository IDs

To find the correct `repo_id` for a model:

1. Go to [Hugging Face](https://huggingface.co/)
2. Search for the model you want (e.g., "Qwen3 GGUF")
3. The repo ID is in the format: `username/repository-name`
   - Example: `Qwen/Qwen3-0.6B-GGUF`
   - Example: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`

Make sure the `model_file` name matches the exact filename in the repository.

## API Endpoint

### WebSocket: `/api/download-models`

Downloads models from Hugging Face based on configured `repo_id`.

#### Request Format

```javascript
{
    "model_ids": ["qwen_3_0.6B", "gemma3_4b_q4_k_m"],
    "force_redownload": false  // Optional, default: false
}
```

#### Parameters

- `model_ids` (required): Array of model IDs from `model_options`
- `force_redownload` (optional): If `true`, re-downloads existing files

#### Response Messages

The endpoint sends WebSocket messages with the following types:

##### Status Message
```json
{
    "type": "status",
    "message": "Starting download for 2 model(s)...",
    "data": null
}
```

##### Progress Message
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

##### Result Message
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

##### Error Message
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

#### Status Values

- `pending`: Download not yet started
- `downloading`: Currently downloading
- `completed`: Download completed successfully
- `failed`: Download failed with error
- `skipped`: File already exists (not re-downloaded)

#### Overall Status Values

- `completed`: All files downloaded successfully (or skipped)
- `partial`: Some files succeeded, some failed
- `failed`: All files failed to download

## Usage Examples

### Python Example

```python
import asyncio
import json
import websockets

async def download_models():
    uri = "ws://localhost:8000/api/download-models"
    
    async with websockets.connect(uri) as websocket:
        # Send request
        await websocket.send(json.dumps({
            "model_ids": ["qwen_3_0.6B"],
            "force_redownload": False
        }))
        
        # Receive progress updates
        async for message in websocket:
            data = json.loads(message)
            print(f"[{data['type']}] {data['message']}")

asyncio.run(download_models())
```

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/api/download-models');

ws.onopen = () => {
    ws.send(JSON.stringify({
        model_ids: ["qwen_3_0.6B", "gemma3_4b_q4_k_m"],
        force_redownload: false
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`[${data.type}] ${data.message}`);
    
    if (data.type === 'progress' && data.data) {
        const { filename, bytes_downloaded, total_bytes } = data.data;
        const percent = (bytes_downloaded / total_bytes) * 100;
        console.log(`  ${filename}: ${percent.toFixed(1)}%`);
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Download complete');
};
```

### cURL Example (for testing)

```bash
# Install websocat for WebSocket testing
brew install websocat  # macOS
# or
apt-get install websocat  # Ubuntu/Debian

# Test download
echo '{"model_ids": ["qwen_3_0.6B"], "force_redownload": false}' | \
  websocat ws://localhost:8000/api/download-models
```

## PyInstaller Integration

The download endpoint is fully compatible with PyInstaller-packaged executables:

### How It Works

1. **Model Directory Resolution**: Uses `config.get_model_path()` which works correctly in both development and frozen (PyInstaller) modes
2. **Path Handling**: Automatically detects if running from PyInstaller bundle via `sys.frozen` check
3. **Download Location**:
   - **Development**: Downloads to `project_root/model/`
   - **PyInstaller (macOS)**: Downloads to bundle's model directory
   - **PyInstaller (Windows)**: Downloads to bundle's model directory

### Building with PyInstaller

The existing PyInstaller spec files already handle the necessary configurations:

```bash
# macOS
python -m PyInstaller ai_capability.spec

# Windows
python -m PyInstaller ai_capability_windows.spec
```

### Model Directory in Packaged App

When packaged:
- **macOS**: `YourApp.app/Contents/MacOS/model/`
- **Windows**: `YourApp/model/`

Downloaded models will be placed in the correct location automatically.

## Error Handling

### Common Errors

#### 1. Missing repo_id
```json
{
    "type": "error",
    "message": "Model qwen_3 does not have a repo_id configured. Please add repo_id to model_options in settings.py"
}
```
**Solution**: Add `repo_id` field to the model in `settings.py`

#### 2. Invalid model_id
```json
{
    "type": "error",
    "message": "Invalid model_id: unknown_model. Not found in model_options."
}
```
**Solution**: Use a valid model ID from `model_options`

#### 3. Hugging Face library not installed
```json
{
    "type": "error",
    "message": "huggingface_hub library not installed. Please install it with: pip install huggingface_hub"
}
```
**Solution**: `pip install huggingface_hub`

#### 4. File not found in repository
```json
{
    "type": "error",
    "message": "Failed to download model_file.gguf: File not found",
    "data": {
        "filename": "model_file.gguf",
        "error": "File not found in repository"
    }
}
```
**Solution**: Verify the `model_file` name matches the exact filename in the Hugging Face repository

#### 5. Network/Connection Issues
```json
{
    "type": "error",
    "message": "Failed to download: Connection timeout"
}
```
**Solution**: Check internet connection and try again. Downloads automatically resume if interrupted.

## Best Practices

### 1. Start with Small Models
Test the download functionality with small models first:
- `qwen_3_0.6B` (~600MB) - Good for testing
- `granite4_350m` (~350MB) - Very small

### 2. Download During Setup
Download all required models during initial setup:

```python
# Download all required models at once
model_ids = ["qwen_3_0.6B", "qwen_3_4B", "gemma3_4b_q4_k_m"]
```

### 3. Use force_redownload Sparingly
Only use `force_redownload: true` when:
- You suspect file corruption
- The model was updated in the repository
- Testing download functionality

### 4. Check Available Models First
Before downloading, check which models are already present:

```bash
curl http://localhost:8000/api/available-models
```

### 5. Monitor Disk Space
Large models can be several GB. Ensure sufficient disk space:
- Vision models: 2-8 GB each
- Chat models: 0.5-8 GB each
- Embedding models: 0.5-4 GB each

## Testing

Use the provided test script:

```bash
cd /path/to/AI_Capability
python tests/test_download_models.py
```

This script:
1. Connects to the download endpoint
2. Requests a small test model
3. Shows real-time progress
4. Reports success/failure

## Troubleshooting

### Downloads Are Slow
- Normal for large models over slow connections
- Downloads resume automatically if interrupted
- Consider downloading overnight for large models

### Permission Errors
- Ensure write permissions for the model directory
- On macOS/Linux: `chmod -R 755 model/`
- On Windows: Run as administrator if needed

### Out of Disk Space
- Check available space: `df -h` (macOS/Linux) or `dir` (Windows)
- Remove old models from `model/` directory
- Consider using smaller quantized models (Q4, Q5 instead of Q8)

### Model Files Corrupt
- Re-download with `force_redownload: true`
- Verify file integrity by comparing size with repository

## Security Notes

1. **Repository Trust**: Only download models from trusted Hugging Face repositories
2. **File Verification**: The endpoint does not verify file integrity (checksums)
3. **Network Security**: Downloads occur over HTTPS through Hugging Face's CDN
4. **No Authentication**: This endpoint does not require authentication (add if needed)

## Performance

- **Concurrent Downloads**: Currently downloads files sequentially within a model
- **Resume Support**: Yes, automatically resumes interrupted downloads
- **Bandwidth**: Limited by your internet connection and Hugging Face's CDN
- **Typical Speeds**: 
  - Fast connection: 50-100 MB/s
  - Medium connection: 10-20 MB/s
  - Slow connection: 1-5 MB/s

## Future Enhancements

Potential improvements for future versions:

1. **Parallel downloads**: Download multiple files simultaneously
2. **Checksum verification**: Verify file integrity after download
3. **Bandwidth limiting**: Limit download speed to prevent network saturation
4. **Progress percentage**: Add percentage completion for individual files
5. **Model catalog**: Browse available models from Hugging Face
6. **Authentication**: Support for private repositories
7. **Selective file download**: Download only specific quantization levels

## Additional Resources

- [Hugging Face Models](https://huggingface.co/models)
- [GGUF Format Documentation](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Model Quantization Guide](https://github.com/ggerganov/llama.cpp/discussions/2094)
