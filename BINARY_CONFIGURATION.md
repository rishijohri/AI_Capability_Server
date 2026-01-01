# Binary Configuration System

## Overview

The AI Server now automatically detects your system properties (OS, CPU architecture, GPU) and selects the appropriate binary configuration from the `binary/llama_binaries/` folder. This ensures optimal performance by using the right llama binaries for your hardware.

## Features

### 1. Automatic Detection on Startup

When the server starts, it automatically:
- Detects your operating system (macOS or Windows)
- Identifies CPU architecture (ARM64, x64)
- Detects GPU capabilities (AMD Radeon, Intel Arc, NVIDIA, or CPU-only)
- Selects the best matching binary configuration

### 2. Manual Configuration

You can manually override the automatic selection using the API:
- View current configuration: `GET /api/config`
- Change configuration: `POST /api/config` with `binary_config` parameter

### 3. Available Configurations

The system supports the following binary configurations:

#### macOS:
- `llama-mac-arm64` - Apple Silicon (M1, M2, M3, etc.)
- `llama-mac-x64` - Intel-based Macs

#### Windows:
- `llama-win-cpu-x64` - CPU-only (universal fallback)
- `llama-win-vulkan-x64` - NVIDIA GPUs (using Vulkan)
- `llama-win-hip-radeon-x64` - AMD Radeon GPUs
- `llama-win-sycl-x64` - Intel Arc GPUs

## Directory Structure

```
binary/
├── llama_binaries/
│   ├── llama-mac-arm64/
│   │   ├── llama-server
│   │   ├── llama-cli
│   │   └── llama-embedding
│   ├── llama-mac-x64/
│   │   ├── llama-server
│   │   ├── llama-cli
│   │   └── llama-embedding
│   ├── llama-win-cpu-x64/
│   │   ├── llama-server.exe
│   │   ├── llama-cli.exe
│   │   └── llama-embedding.exe
│   ├── llama-win-vulkan-x64/
│   │   └── ...
│   ├── llama-win-hip-radeon-x64/
│   │   └── ...
│   └── llama-win-sycl-x64/
│       └── ...
└── (legacy binaries for backward compatibility)
```

## API Usage

### Get Current Configuration

```bash
GET /api/config
```

**Response includes:**
```json
{
  "binary_config": "llama-mac-arm64",
  "system_info": {
    "os": "mac",
    "architecture": "arm64",
    "gpu": "cpu",
    "platform": "macOS-14.0-arm64",
    "machine": "arm64",
    "processor": "arm"
  },
  "available_binary_configs": [
    "llama-mac-arm64",
    "llama-mac-x64",
    "llama-win-cpu-x64",
    "llama-win-hip-radeon-x64",
    "llama-win-sycl-x64",
    "llama-win-vulkan-x64"
  ],
  ...
}
```

### Change Binary Configuration

```bash
POST /api/config
Content-Type: application/json

{
  "binary_config": "llama-mac-x64"
}
```

**Response:**
```json
{
  "status": "success",
  "binary_config": "llama-mac-x64",
  ...
}
```

**Error if configuration doesn't exist:**
```json
{
  "detail": "Invalid binary configuration: non-existent-config"
}
```

## Python API

### Using SystemDetector

```python
from app.utils.system_detector import SystemDetector
from pathlib import Path

# Get system information
system_info = SystemDetector.get_system_info()
print(f"OS: {system_info['os']}")
print(f"Architecture: {system_info['architecture']}")
print(f"GPU: {system_info['gpu']}")

# Get available configurations
binary_dir = Path("binary")
available = SystemDetector.get_available_configs(binary_dir)
print(f"Available: {available}")

# Auto-detect best configuration
config = SystemDetector.auto_detect_config(binary_dir)
print(f"Recommended: {config}")

# Validate a configuration
is_valid = SystemDetector.validate_config("llama-mac-arm64", binary_dir)
print(f"Valid: {is_valid}")
```

### Using Server Configuration

```python
from app.config import initialize_config, get_config, update_config

# Initialize with auto-detection (called automatically on startup)
config = initialize_config()
print(f"Selected: {config.binary_config}")
print(f"System: {config.system_info}")

# Get binary path (automatically uses selected config)
binary_path = config.get_binary_path("llama-server")
print(f"Binary: {binary_path}")

# List available configurations
available = config.get_available_binary_configs()
print(f"Available: {available}")

# Manually change configuration
success = config.set_binary_config("llama-mac-x64")
if success:
    print("Configuration changed successfully")
else:
    print("Invalid configuration")

# Or use update_config
updated_config = update_config(binary_config="llama-mac-arm64")
```

## Startup Messages

When the server starts, you'll see output like:

```
AI Server starting...
System detected: mac (arm64, cpu)
Selected binary configuration: llama-mac-arm64
Available binary configurations: llama-mac-arm64, llama-mac-x64, llama-win-cpu-x64, ...
```

## Detection Logic

### Operating System
- **Darwin** → `mac`
- **Windows** → `win`
- **Linux/Other** → `mac` (default, since llama binaries are Unix-compatible)

### CPU Architecture
- **arm64/aarch64** → `arm64`
- **x86_64/amd64** → `x64`

### GPU Detection (Windows only)
Uses `wmic` to query video controller information:
- Contains "amd" or "radeon" → `hip-radeon`
- Contains "intel" and "arc" → `sycl`
- Contains "nvidia", "geforce", or "rtx" → `vulkan`
- Otherwise → `cpu` (CPU-only)

### Configuration Priority
1. Preferred: OS + GPU + Architecture (e.g., `llama-win-vulkan-x64`)
2. Fallback: Any config matching OS + Architecture
3. Fallback: Any config matching OS
4. Last resort: First available config

## Validation

The system validates configurations to ensure:
- The configuration folder exists in `binary/llama_binaries/`
- The folder is a valid directory
- The configuration name follows the naming convention

Invalid configuration names will be rejected with an error message.

## Testing

Run the test script to verify system detection:

```bash
python3 test_basic_detection.py
```

This will show:
- Detected OS and architecture
- GPU type (if applicable)
- Recommended configuration
- List of available configurations
- Whether the recommended config exists

## Troubleshooting

### Binary Configuration Not Found

If you see warnings about missing binaries:
1. Check that `binary/llama_binaries/[config-name]/` exists
2. Verify the folder contains the required binaries (llama-server, llama-cli, etc.)
3. Ensure binaries have execute permissions on macOS/Linux: `chmod +x binary/llama_binaries/*/llama-*`

### Wrong Configuration Selected

You can manually override the selection:
```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"binary_config": "llama-mac-x64"}'
```

### GPU Not Detected (Windows)

If your GPU isn't detected correctly:
1. Verify GPU drivers are installed
2. Run `wmic path win32_VideoController get name` to see if the GPU is visible
3. Manually select the appropriate configuration using the API

## Migration from Legacy System

If you were using the old binary structure (flat `binary/` folder):
1. The system still supports the old structure for backward compatibility
2. To use the new system, organize binaries into configuration folders
3. The server will use the new system automatically when `llama_binaries/` exists

## Performance Considerations

- **Apple Silicon (M1/M2/M3)**: Use `llama-mac-arm64` for best performance with Metal acceleration
- **Intel Macs**: Use `llama-mac-x64`
- **NVIDIA GPUs**: Use `llama-win-vulkan-x64` for GPU acceleration
- **AMD Radeon**: Use `llama-win-hip-radeon-x64` for ROCm support
- **Intel Arc**: Use `llama-win-sycl-x64` for oneAPI support
- **CPU-only**: Use `llama-win-cpu-x64` or `llama-mac-x64`

## Related Files

- `app/utils/system_detector.py` - System detection logic
- `app/config/settings.py` - Configuration management
- `app/api/routes.py` - API endpoints
- `app/main.py` - Startup initialization
- `test_basic_detection.py` - Test script
