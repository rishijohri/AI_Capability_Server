# Windows Support Documentation

This document outlines the cross-platform compatibility features of the AI Capability Server, specifically focusing on Windows support.

## Overview

The AI Capability Server has been designed to work seamlessly on both **Windows** and **macOS/Linux** systems. The key difference between platforms is the binary executable format:

- **macOS/Linux**: Binaries have no extension (e.g., `llama-server`, `llama-cli`)
- **Windows**: Binaries have `.exe` extension (e.g., `llama-server.exe`, `llama-cli.exe`)

## Binary Structure

### macOS/Linux Directory Structure
```
binary/
├── llama-cli
├── llama-embedding
├── llama-mtmd-cli
└── llama-server
```

### Windows Directory Structure
```
binary/
├── llama-cli.exe
├── llama-embedding.exe
├── llama-mtmd-cli.exe
└── llama-server.exe
```

## Implementation Details

### 1. Automatic Binary Path Resolution

The system automatically detects the operating system and appends `.exe` extension on Windows.

**File**: `app/utils/resource_paths.py`

```python
def get_binary_path(binary_name: str) -> Path:
    """Get absolute path to a binary file.
    
    On Windows, automatically appends .exe extension if not already present.
    On macOS/Linux, uses the binary name as-is.
    """
    if platform.system() == "Windows" and not binary_name.endswith(".exe"):
        binary_name = f"{binary_name}.exe"
    
    return get_resource_path(f"binary/{binary_name}")
```

**Usage** (code is platform-agnostic):
```python
# This works on both platforms
binary_path = config.get_binary_path("llama-server")
# Returns: /path/to/binary/llama-server (macOS/Linux)
#      or: C:\path\to\binary\llama-server.exe (Windows)
```

### 2. Process Management

The process manager handles process names correctly on both platforms.

**File**: `app/utils/process_manager.py`

```python
async def kill_existing_binary_processes(self, binary_name: str) -> int:
    """Kill all system processes with the given binary name.
    
    On Windows, automatically checks for .exe extension.
    On macOS/Linux, checks exact name.
    """
    binary_names = [binary_name]
    if platform.system() == "Windows":
        if not binary_name.endswith(".exe"):
            binary_names.append(f"{binary_name}.exe")
    
    # Process killing logic...
```

### 3. Path Handling

All path handling uses `pathlib.Path`, which is cross-platform compatible:

```python
from pathlib import Path

# Works on both Windows and Unix-like systems
model_path = Path("model/Qwen3-8B-Q4_K_M.gguf")
binary_path = Path("binary") / "llama-server"  # Will be corrected automatically
```

## Platform-Specific Considerations

### Windows-Specific Notes

1. **Binary Extensions**: Always place `.exe` binaries in the `binary/` directory
2. **Path Separators**: Use `pathlib.Path` - it handles backslashes automatically
3. **Process Management**: Windows uses different process handling, but `psutil` abstracts this
4. **Line Endings**: Git should be configured to handle CRLF/LF conversion automatically

### macOS/Linux-Specific Notes

1. **Binary Permissions**: Binaries must be executable:
   ```bash
   chmod +x binary/*
   ```
2. **Path Separators**: Forward slashes (handled automatically by `pathlib`)

## Testing Cross-Platform Compatibility

### Verify Binary Detection

```python
from app.config import get_config

config = get_config()

# Should work on both platforms
llama_server = config.get_binary_path("llama-server")
llama_cli = config.get_binary_path("llama-cli")
llama_mtmd = config.get_binary_path("llama-mtmd-cli")
llama_embed = config.get_binary_path("llama-embedding")

print(f"Server binary: {llama_server}")
print(f"CLI binary: {llama_cli}")
print(f"MTMD binary: {llama_mtmd}")
print(f"Embedding binary: {llama_embed}")
```

**Expected Output on Windows**:
```
Server binary: C:\...\binary\llama-server.exe
CLI binary: C:\...\binary\llama-cli.exe
MTMD binary: C:\...\binary\llama-mtmd-cli.exe
Embedding binary: C:\...\binary\llama-embedding.exe
```

**Expected Output on macOS/Linux**:
```
Server binary: /Users/.../binary/llama-server
CLI binary: /Users/.../binary/llama-cli
MTMD binary: /Users/.../binary/llama-mtmd-cli
Embedding binary: /Users/.../binary/llama-embedding
```

## PyInstaller Cross-Platform Building

### Building on Windows

**Option 1: PowerShell (Recommended)**
```powershell
# Windows PowerShell
.\build.ps1
```

**Option 2: Command Prompt**
```cmd
# Windows Command Prompt
build.bat
```

The Windows build scripts will:
- Check for virtual environment and activate if needed
- Verify PyInstaller is installed
- Check for `.exe` binaries in `binary/` directory
- Validate model files
- Build the executable
- Report success/failure with helpful messages

### Building on macOS/Linux

```bash
# macOS/Linux Terminal
./build.sh
```

The Unix build script will:
- Detect if accidentally run on Windows (Git Bash/WSL) and warn
- Check for virtual environment and activate if needed
- Verify PyInstaller is installed
- Check for binaries (no extension) in `binary/` directory
- Validate model files
- Build the executable
- Report success/failure with helpful messages

### Spec File Configuration

The `ai_capability.spec` file automatically handles binary collection:

```python
# In ai_capability.spec
a = Analysis(
    # ...
    datas=[
        ('binary/*', 'binary'),  # Includes all files in binary/
        ('model/', 'model'),
        # ...
    ]
)
```

**Note**: The wildcard `binary/*` will include:
- `llama-*.exe` files on Windows
- `llama-*` files (no extension) on macOS/Linux

### Build Scripts Available

| Platform | Script | Description |
|----------|--------|-------------|
| **Windows** | `build.ps1` | PowerShell script (recommended) |
| **Windows** | `build.bat` | Batch script (alternative) |
| **macOS/Linux** | `build.sh` | Bash script |

All scripts provide:
- ✅ Virtual environment activation
- ✅ Dependency checking
- ✅ Binary and model validation
- ✅ Platform-specific checks
- ✅ Helpful error messages
- ✅ Build success/failure reporting

## Deployment

### Windows Deployment

1. Place Windows binaries (`.exe` files) in `binary/` directory
2. Ensure all DLL dependencies are available (usually bundled with `.exe`)
3. Run or build the application

### macOS/Linux Deployment

1. Place Unix binaries (no extension) in `binary/` directory
2. Make binaries executable: `chmod +x binary/*`
3. Ensure system libraries are available (OpenMP, CUDA if using GPU)
4. Run or build the application

## Common Issues and Solutions

### Issue: Binary Not Found on Windows

**Error**: `FileNotFoundError: llama-server binary not found`

**Solution**: 
- Ensure `.exe` files are in the `binary/` directory
- Check that the binary names match exactly (e.g., `llama-server.exe`)

### Issue: Permission Denied on macOS/Linux

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
chmod +x binary/*
```

### Issue: Process Won't Start on Windows

**Problem**: Process fails to start with no error

**Solution**:
- Check Windows Defender/Antivirus settings
- Run as Administrator if needed
- Verify all DLL dependencies are present

### Issue: Process Won't Terminate on Windows

**Problem**: Processes remain running after stop command

**Solution**:
The system automatically handles this by checking for both process names:
- `llama-server` (basename)
- `llama-server.exe` (Windows full name)

## Code Examples

### Starting a Server (Cross-Platform)

```python
from app.services.llm_service import get_llm_service

llm_service = get_llm_service()

# This works on both Windows and macOS/Linux
await llm_service.load_model(
    "Qwen3-8B-Q4_K_M.gguf",
    use_server=True
)
```

### Running CLI Mode (Cross-Platform)

```python
from app.services.llm_service import get_llm_service

llm_service = get_llm_service()

# This works on both Windows and macOS/Linux
await llm_service.load_model(
    "Qwen3-8B-Q4_K_M.gguf",
    use_server=False  # Uses llama-cli
)
```

## Summary

The AI Capability Server achieves cross-platform compatibility through:

1. **Automatic Binary Extension Detection**: The system detects Windows and adds `.exe` automatically
2. **Platform-Agnostic Path Handling**: Using `pathlib.Path` for all file operations
3. **Smart Process Management**: Checking for both base names and `.exe` variants on Windows
4. **Unified API**: All code uses the same interface regardless of platform

No code changes are required when switching between Windows and macOS/Linux. Simply place the appropriate binaries in the `binary/` directory for your platform.
