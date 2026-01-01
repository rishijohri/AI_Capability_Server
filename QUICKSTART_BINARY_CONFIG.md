# Quick Start Guide - Binary Configuration

## What's New?

The AI Server now automatically detects your system and selects the best binaries for your hardware!

## Quick Test

Run this to see what was detected:

```bash
python3 test_basic_detection.py
```

You'll see:
- Your operating system
- CPU architecture
- GPU type (Windows only)
- Recommended binary configuration
- All available configurations

## Starting the Server

Just start normally:

```bash
python3 run_server.py
```

You'll see output like:

```
AI Server starting...
System detected: mac (arm64, cpu)
Selected binary configuration: llama-mac-arm64
Available binary configurations: llama-mac-arm64, llama-mac-x64, ...
```

## API Usage

### Check Current Configuration

```bash
curl http://127.0.0.1:8000/api/config
```

Look for these new fields in the response:
```json
{
  "binary_config": "llama-mac-arm64",
  "system_info": {
    "os": "mac",
    "architecture": "arm64",
    "gpu": "cpu"
  },
  "available_binary_configs": ["llama-mac-arm64", "llama-mac-x64", ...]
}
```

### Change Configuration

```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"binary_config": "llama-mac-x64"}'
```

## Configuration Options

### macOS:
- `llama-mac-arm64` - Apple Silicon (M1/M2/M3) [Auto-detected for ARM Macs]
- `llama-mac-x64` - Intel Macs [Auto-detected for Intel Macs]

### Windows:
- `llama-win-vulkan-x64` - NVIDIA GPUs [Auto-detected for NVIDIA]
- `llama-win-hip-radeon-x64` - AMD Radeon GPUs [Auto-detected for AMD]
- `llama-win-sycl-x64` - Intel Arc GPUs [Auto-detected for Intel Arc]
- `llama-win-cpu-x64` - CPU-only, works on all systems [Fallback]

## Troubleshooting

### "Binary not found" error

Make sure binaries exist in the correct location:
```
binary/
└── llama_binaries/
    └── llama-mac-arm64/          (or your configuration)
        ├── llama-server
        ├── llama-cli
        └── llama-embedding
```

### Wrong configuration auto-selected

Manually override via API:
```bash
curl -X POST http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"binary_config": "YOUR-PREFERRED-CONFIG"}'
```

## Examples

See the example script:
```bash
python3 example_binary_config.py
```

## More Information

- Full documentation: `BINARY_CONFIGURATION.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
