# PyInstaller Spec Files - Platform Guide

## Overview

The AI Capability Server uses **two separate PyInstaller spec files** to ensure optimal binary packaging on different platforms:

- **`ai_capability.spec`** - For macOS/Linux (binaries without extensions)
- **`ai_capability_windows.spec`** - For Windows (binaries with `.exe` extensions)

## Why Two Spec Files?

### The Problem

Binary executables have different naming conventions across platforms:

| Platform | Binary Name | Example |
|----------|-------------|---------|
| **macOS/Linux** | No extension | `llama-server`, `llama-cli` |
| **Windows** | `.exe` extension | `llama-server.exe`, `llama-cli.exe` |

PyInstaller needs to know exactly which files to include, and a one-size-fits-all approach would either:
- Include wrong files (e.g., looking for `.exe` on macOS)
- Miss required files (e.g., not finding binaries without extensions on Windows)

### The Solution

Platform-specific spec files with tailored binary collection logic:

```python
# Unix: Collect all files (no extension filtering)
for binary_file in binary_dir.iterdir():
    if binary_file.is_file():
        binaries.append((str(binary_file), 'binary'))

# Windows: Specifically collect .exe and .dll files
for binary_file in binary_dir.iterdir():
    if binary_file.is_file():
        if binary_file.suffix.lower() in ['.exe', '.dll']:
            binaries.append((str(binary_file), 'binary'))
```

## Spec File Comparison

### `ai_capability.spec` (macOS/Linux)

**Location**: Project root  
**Used by**: `build.sh`  
**Platform**: macOS, Linux, Unix-like systems

**Key Features**:
```python
# Binary Collection (No extension filtering)
for binary_file in binary_dir.iterdir():
    if binary_file.is_file():
        binaries.append((str(binary_file), 'binary'))

# Executable Name (No .exe extension)
exe = EXE(
    # ...
    name='ai_capability_server',  # Creates: ai_capability_server
    # ...
)
```

**Output**: `dist/ai_capability_server/ai_capability_server`

---

### `ai_capability_windows.spec` (Windows)

**Location**: Project root  
**Used by**: `build.ps1`, `build.bat`  
**Platform**: Windows

**Key Features**:
```python
# Binary Collection (Filters for .exe and .dll)
for binary_file in binary_dir.iterdir():
    if binary_file.is_file():
        if binary_file.suffix.lower() in ['.exe', '.dll']:
            binaries.append((str(binary_file), 'binary'))
        elif binary_file.suffix == '':
            # Fallback for files without extension
            binaries.append((str(binary_file), 'binary'))

# Executable Name (Automatically gets .exe)
exe = EXE(
    # ...
    name='ai_capability_server',  # Creates: ai_capability_server.exe
    # ...
)
```

**Output**: `dist\ai_capability_server\ai_capability_server.exe`

## Common Features

Both spec files share these identical features:

### 1. Model File Collection
```python
# Collect GGUF and BIN model files
for model_file in model_dir.iterdir():
    if model_file.is_file() and model_file.suffix in ['.gguf', '.bin']:
        datas.append((str(model_file), 'model'))
```

### 2. InsightFace Models (Recursive)
```python
# Collect InsightFace models with directory structure
models_subdir = model_dir / 'models'
if models_subdir.exists():
    for root, dirs, files in models_subdir.walk():
        for file in files:
            src_path = root / file
            rel_path = src_path.relative_to(model_dir)
            dest_dir = 'model' / rel_path.parent
            datas.append((str(src_path), str(dest_dir)))
```

### 3. Hidden Imports
```python
hiddenimports = [
    'uvicorn.logging',
    'uvicorn.loops.auto',
    'aiohttp',
    'fastapi',
    'insightface',
    'onnxruntime',
    # ... and more
]
```

### 4. Folder-Based Distribution
```python
exe = EXE(
    # ...
    exclude_binaries=True,  # CRITICAL: Creates folder, not single file
    # ...
)

# Creates distributable folder structure
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    # ...
    name='ai_capability_server',
)
```

## Usage

### macOS/Linux

**Using Build Script** (Recommended):
```bash
./build.sh
```

**Manual Build**:
```bash
pyinstaller --clean ai_capability.spec
```

**Output**:
```
dist/ai_capability_server/
├── ai_capability_server          # Executable (no extension)
├── binary/
│   ├── llama-server             # No extension
│   ├── llama-cli
│   └── llama-mtmd-cli
├── model/
│   ├── Qwen3-8B-Q4_K_M.gguf
│   └── models/
│       └── buffalo_l/
└── [Python dependencies]
```

---

### Windows

**Using Build Script** (Recommended):
```powershell
# PowerShell
.\build.ps1

# Command Prompt
build.bat
```

**Manual Build**:
```powershell
pyinstaller --clean ai_capability_windows.spec
```

**Output**:
```
dist\ai_capability_server\
├── ai_capability_server.exe      # Windows executable
├── binary\
│   ├── llama-server.exe          # With .exe
│   ├── llama-cli.exe
│   └── llama-mtmd-cli.exe
├── model\
│   ├── Qwen3-8B-Q4_K_M.gguf
│   └── models\
│       └── buffalo_l\
└── [Python dependencies & DLLs]
```

## Customization

### Adding Custom Binaries

**For macOS/Linux** (ai_capability.spec):
```python
# Add any additional binaries
binaries.append(('/path/to/custom-binary', 'binary'))
```

**For Windows** (ai_capability_windows.spec):
```python
# Add .exe or .dll files
binaries.append(('C:\\path\\to\\custom-binary.exe', 'binary'))
```

### Adding Data Files

Both spec files use the same syntax:
```python
# Add single file
datas.append(('config.json', '.'))

# Add entire directory
datas.append(('resources/', 'resources'))
```

### Adding Hidden Imports

Both spec files use the same syntax:
```python
hiddenimports = [
    # ... existing imports
    'your.custom.module',
    'another.module',
]
```

### Windows-Specific: Adding an Icon

Only in `ai_capability_windows.spec`:
```python
exe = EXE(
    # ...
    icon='path/to/icon.ico',  # Windows icon file
    # ...
)
```

## Distribution

### macOS/Linux

**Create Archive**:
```bash
cd dist
tar -czf ai_capability_server.tar.gz ai_capability_server/
```

**Or ZIP**:
```bash
cd dist
zip -r ai_capability_server.zip ai_capability_server/
```

---

### Windows

**Create ZIP** (PowerShell):
```powershell
cd dist
Compress-Archive -Path ai_capability_server -DestinationPath ai_capability_server.zip
```

**Or using Command Prompt**:
```cmd
cd dist
tar -a -c -f ai_capability_server.zip ai_capability_server
```

## Troubleshooting

### Issue: Binaries Not Included (Windows)

**Symptom**: Build succeeds but binaries missing from dist folder

**Cause**: Binaries don't have `.exe` extension

**Solution**: 
1. Verify files in `binary/` have `.exe` extension
2. Check build output for warnings about skipped files

---

### Issue: Binaries Not Included (macOS/Linux)

**Symptom**: Build succeeds but binaries missing from dist folder

**Cause**: Binaries are symbolic links or have unusual permissions

**Solution**:
1. Ensure binaries are actual files (not symlinks)
2. Make sure they're readable: `chmod +r binary/*`

---

### Issue: Model Files Too Large

**Symptom**: Build takes forever or runs out of disk space

**Solution**:
1. Models are necessary for the application
2. Ensure sufficient disk space (models can be several GB)
3. Consider distributing without models and downloading separately

---

### Issue: Hidden Import Missing

**Symptom**: Runtime error about missing module

**Solution**: Add to `hiddenimports` in spec file:
```python
hiddenimports = [
    # ... existing imports
    'missing.module.name',
]
```

## Best Practices

### 1. Always Use Build Scripts

✅ **Do**:
```bash
./build.sh          # macOS/Linux
.\build.ps1         # Windows
```

❌ **Don't**:
```bash
pyinstaller ai_capability.spec  # On Windows (wrong spec file!)
```

### 2. Clean Builds

Both build scripts automatically clean previous builds:
```bash
# Automatic in build scripts
rm -rf build/ dist/
```

### 3. Verify Binary Collection

After building, check that all binaries are present:

**macOS/Linux**:
```bash
ls -la dist/ai_capability_server/binary/
```

**Windows**:
```powershell
dir dist\ai_capability_server\binary\
```

### 4. Test Before Distribution

Always test the built executable before distributing:

**macOS/Linux**:
```bash
cd dist/ai_capability_server
./ai_capability_server
```

**Windows**:
```powershell
cd dist\ai_capability_server
.\ai_capability_server.exe
```

## Summary

| Aspect | macOS/Linux | Windows |
|--------|-------------|---------|
| **Spec File** | `ai_capability.spec` | `ai_capability_windows.spec` |
| **Build Script** | `build.sh` | `build.ps1` or `build.bat` |
| **Binary Extension** | None | `.exe` |
| **Executable Name** | `ai_capability_server` | `ai_capability_server.exe` |
| **Binary Collection** | All files | `.exe` and `.dll` only |
| **Output Format** | Same folder structure | Same folder structure |

Both spec files produce **functionally identical** applications, just with platform-appropriate executable formats.
