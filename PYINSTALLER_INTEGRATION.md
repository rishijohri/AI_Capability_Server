# PyInstaller Integration Summary

## Overview

The AI Capability Server is now fully compatible with PyInstaller packaging. All resource paths (binaries and models) are resolved correctly in both development (source) and production (frozen/packaged) environments.

## Key Components

### 1. Resource Path Utilities (`app/utils/resource_paths.py`)

Central module for PyInstaller-compatible path resolution:

```python
from app.utils.resource_paths import (
    get_base_path,           # Get project root or PyInstaller temp directory
    get_binary_path,         # Get path to binary files
    get_model_path,          # Get path to model files
    get_face_models_path,    # Get path to face models directory
    get_data_directory,      # Get user data directory
)
```

**How it works:**
- **Frozen (PyInstaller)**: Uses `sys._MEIPASS` (temp extraction directory)
- **Source**: Uses project root directory

### 2. Configuration Integration (`app/config/settings.py`)

The `ServerConfig` class now uses resource path utilities:

```python
config = get_config()

# Get paths (works in both source and frozen)
binary_path = config.get_binary_path("llama-server")
model_path = config.get_model_path("Qwen3-8B-Q4_K_M.gguf")
```

### 3. PyInstaller Spec File (`ai_capability.spec`)

Defines the build configuration:
- Automatically includes all binaries from `binary/`
- Automatically includes all GGUF models from `model/`
- Recursively includes InsightFace models from `model/models/` subdirectories
- Includes hidden imports for FastAPI, uvicorn, aiohttp, insightface, onnxruntime

### 4. Build Script (`build.sh`)

Automated build process:
```bash
./build.sh
```

## Directory Structure

### Source (Development)
```
AI_Capability/
├── app/
│   ├── utils/
│   │   └── resource_paths.py  # Resource path utilities
│   └── config/
│       └── settings.py         # Uses resource_paths
├── binary/                     # Binaries (llama-server, llama-cli, etc.)
├── model/                      # Models (.gguf files)
├── ai_capability.spec          # PyInstaller spec
├── build.sh                    # Build script
└── run_server.py               # Entry point
```

### Frozen (Packaged)
```
dist/ai_capability_server/
├── ai_capability_server        # Main executable
├── binary/                     # Bundled binaries
├── model/                      # Bundled models
└── ... (dependencies)
```

## Path Resolution Examples

### Running from Source
```python
# Resource paths resolve to project directories
get_base_path()
# → /Users/user/Projects/AI_Capability

get_binary_path("llama-server")
# → /Users/user/Projects/AI_Capability/binary/llama-server

get_model_path("Qwen3-8B-Q4_K_M.gguf")
# → /Users/user/Projects/AI_Capability/model/Qwen3-8B-Q4_K_M.gguf

get_data_directory()
# → /Users/user/Projects/AI_Capability
```

### Running from Packaged Executable
```python
# Resource paths resolve to PyInstaller temp directory
get_base_path()
# → /var/folders/.../T/_MEI12345

get_binary_path("llama-server")
# → /var/folders/.../T/_MEI12345/binary/llama-server

get_model_path("Qwen3-8B-Q4_K_M.gguf")
# → /var/folders/.../T/_MEI12345/model/Qwen3-8B-Q4_K_M.gguf

get_data_directory()
# → /Users/user/.ai_capability/data
```

## Building the Executable

### Quick Build
```bash
./build.sh
```

### Manual Build
```bash
pyinstaller --clean ai_capability.spec
```

### Output Structure

The build creates a **folder-based distribution**:
```
dist/ai_capability_server/
├── ai_capability_server           # Main executable
├── binary/                        # Bundled binaries
│   ├── llama-server
│   ├── llama-cli
│   └── ...
├── model/                         # Bundled models
│   ├── Qwen3-8B-Q4_K_M.gguf
│   ├── qwen3-embedding-8b-q4_k_m.gguf
│   └── ...
├── _internal/                     # Python dependencies
│   ├── (Python libraries)
│   └── (PyInstaller runtime)
└── (Other PyInstaller files)
```

**Important:** The entire directory must be distributed together. Do not distribute only the executable file.

## Running the Packaged Application

```bash
cd dist/ai_capability_server
./ai_capability_server
```

Or with custom port:
```bash
./ai_capability_server --port 8080
```

## Distribution

### Create distributable package:
```bash
cd dist
zip -r ai_capability_server.zip ai_capability_server/
```

**Critical:** Distribute the entire `ai_capability_server/` folder, not just the executable. The folder contains:
- Main executable
- Binary files (llama-server, llama-cli, etc.)
- Model files (.gguf files)
- Python dependencies (_internal/)
- PyInstaller runtime files

### User extraction and run:
```bash
unzip ai_capability_server.zip
cd ai_capability_server
./ai_capability_server
```

### Why Folder-Based?

The application uses a folder-based distribution (not single-file) because:
- ✅ Handles large models (10+ GB) efficiently
- ✅ Faster startup (no extraction overhead)
- ✅ Binary files maintain proper execute permissions
- ✅ Easier to update individual components
- ✅ Better for debugging and inspection

## Data Storage

### Development (Source)
- User data stored in project root
- Storage metadata: `storage-metadata.json`
- RAG directory: `rag/`

### Production (Frozen)
- User data stored in `~/.ai_capability/data/`
- Persists across updates
- Separate from bundled resources

## Benefits

1. **No Python Required**: Users don't need Python installed
2. **Portable**: Single directory contains everything
3. **Consistent Paths**: Same code works in both environments
4. **Easy Distribution**: Zip and share
5. **Version Control**: Resources bundled with version

## Testing

Verify integration:
```bash
python -c "
from app.utils.resource_paths import get_binary_path, get_model_path
from app.config.settings import get_config

config = get_config()
print('Binary:', config.get_binary_path('llama-server'))
print('Model:', config.get_model_path(config.chat_model))
"
```

## Troubleshooting

### "Binary not found"
- Ensure binaries are in `binary/` before building
- Check execute permissions: `chmod +x binary/*`

### "Model not found"
- Ensure models are in `model/` before building
- Verify filenames match config

### Import errors
- Add missing modules to `hiddenimports` in `ai_capability.spec`
- Rebuild: `pyinstaller --clean ai_capability.spec`

## Size Considerations

- **Models**: 10-15 GB (largest component)
- **Binaries**: ~100 MB
- **Dependencies**: ~200 MB
- **Total**: ~10-15 GB packaged

## Platform Notes

### macOS
- Builds are macOS-only
- May need codesigning for distribution

### Linux
- Builds are Linux-only
- Ensure binary execute permissions

### Windows
- Builds are Windows-only
- Use `.exe` extension

## Migration Path

No code changes needed! The implementation:
1. Works identically in source and frozen modes
2. Automatically detects environment
3. Resolves paths appropriately
4. Maintains backward compatibility

## Future Enhancements

Potential improvements:
- Auto-updater integration
- Compressed model storage
- Separate model downloads
- Platform-specific optimizations
