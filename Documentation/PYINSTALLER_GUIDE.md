# PyInstaller Packaging Guide

This guide explains how to package the AI Capability Server as a standalone executable using PyInstaller.

## Prerequisites

```bash
pip install pyinstaller
```

## Building the Executable

### Step 1: Prepare Resources

Ensure the following directories contain the necessary files:

- `binary/` - LLaMA binaries (llama-server, llama-cli, llama-mtmd-cli, llama-qwen2vl-cli)
- `model/` - Model files (.gguf files)

### Step 2: Build

```bash
pyinstaller ai_capability.spec
```

The spec file automatically:
- Includes all binaries from the `binary/` directory
- Includes all model files (.gguf, .bin) from the `model/` directory
- Adds hidden imports for FastAPI, uvicorn, aiohttp, etc.
- **Creates a FOLDER-BASED distribution** (not a single executable)

### Step 3: Output

The build creates a **folder-based distribution** in:
```
dist/ai_capability_server/
├── ai_capability_server     # Main executable
├── binary/                  # LLaMA binaries
│   ├── llama-server
│   ├── llama-cli
│   └── ... (other binaries)
├── model/                   # Model files
│   ├── Qwen3-8B-Q4_K_M.gguf
│   ├── qwen3-embedding-8b-q4_k_m.gguf
│   └── ... (other models)
├── _internal/               # Python dependencies and libraries
└── ... (other PyInstaller files)
```

**Note:** The entire `ai_capability_server/` folder must be distributed together. The executable requires all the accompanying files and folders.

## Running the Packaged Application

```bash
cd dist/ai_capability_server
./ai_capability_server
```

Or specify a port:
```bash
./ai_capability_server --port 8080
```

## Resource Path Handling

The application uses PyInstaller-compatible resource paths via `app/utils/resource_paths.py`:

- **When frozen (packaged)**: Resources are accessed from `sys._MEIPASS` (PyInstaller's temp extraction directory)
- **When running from source**: Resources are accessed relative to project root

### Data Storage

- **Frozen**: User data stored in `~/.ai_capability/data/`
- **Source**: User data stored in project root

## Customizing the Build

Edit `ai_capability.spec` to:

### Add Additional Data Files
```python
datas = [
    ('path/to/file', 'destination/in/bundle'),
]
```

### Add Hidden Imports
```python
hiddenimports = [
    'your_module',
]
```

### Change Output Name
```python
exe = EXE(
    ...
    name='your_custom_name',
    ...
)
```

## Platform-Specific Notes

### macOS
- Built executable is macOS-only
- May need to sign the binary for distribution: `codesign -s "Your Identity" ai_capability_server`
- May need to notarize for Gatekeeper

### Linux
- Built executable is Linux-only
- Ensure binary files have execute permissions: `chmod +x binary/*`

### Windows
- Built executable is Windows-only
- Change `console=False` in spec file for windowed app
- May need to exclude `.so` files and include `.dll` files

## Distribution

To distribute the packaged application:

1. **Zip the entire directory:**
   ```bash
   cd dist
   zip -r ai_capability_server.zip ai_capability_server/
   ```
   
   **Important:** You must distribute the entire `ai_capability_server/` folder, not just the executable. The executable depends on:
   - Binary files in `binary/`
   - Model files in `model/`
   - Python dependencies in `_internal/`
   - Other PyInstaller runtime files

2. **Upload to releases or distribution platform**

3. **Users can extract and run:**
   ```bash
   unzip ai_capability_server.zip
   cd ai_capability_server
   ./ai_capability_server
   ```

## Why Folder-Based Distribution?

The spec file uses `exclude_binaries=True` which creates a folder-based distribution instead of a single executable file. This approach:

✅ **Supports large files**: Model files (10+ GB) work without issues
✅ **Faster startup**: No need to extract everything to temp on each run
✅ **Easier updates**: Replace individual files without rebuilding entire bundle
✅ **Better for resources**: Binary files maintain proper permissions
✅ **Debuggable**: Easier to inspect and troubleshoot issues

### Single File Alternative (NOT RECOMMENDED)

If you need a single executable (not recommended for this project due to large models):
1. Change `exclude_binaries=False` in spec file
2. Remove the `COLLECT()` section
3. Add `a.binaries` and `a.datas` to `EXE()`

However, this will:
- ❌ Create extraction overhead on every run
- ❌ Require large temp space (20+ GB)
- ❌ Slow down startup significantly
- ❌ May fail with very large models

## Troubleshooting

### "Binary not found" error
- Check that binaries are in the `binary/` directory before building
- Verify binaries have execute permissions
- Check `sys._MEIPASS` path during runtime

### "Model not found" error
- Ensure model files are in `model/` directory before building
- Verify model filenames match configuration
- Check available disk space for model extraction

### Import errors
- Add missing modules to `hiddenimports` in spec file
- Rebuild with `pyinstaller --clean ai_capability.spec`

### Size issues
- Models and binaries can be large (10+ GB)
- Consider excluding large unused models
- Use UPX compression (already enabled in spec)

## Development vs Production

### Development (from source)
```bash
python run_server.py
```
- Resources loaded from project directories
- Easy to modify and test
- Requires Python environment

### Production (packaged)
```bash
./dist/ai_capability_server/ai_capability_server
```
- Resources bundled in executable
- No Python installation required
- Portable across machines (same OS)

## Build Optimization

### Reduce Size
```python
# In spec file
exe = EXE(
    ...
    strip=True,      # Strip symbols
    upx=True,        # Compress with UPX
    console=False,   # No console window (if GUI)
)
```

### Debug Build
```python
exe = EXE(
    ...
    debug=True,      # Enable debug output
    console=True,    # Show console
)
```

## Version Information

To add version info (Windows only):
```python
exe = EXE(
    ...
    version='version_info.txt',
)
```

Create `version_info.txt` with version metadata.
