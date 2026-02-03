# Logging System Quick Reference

## View Logs in Console.app

### While App is Running
```bash
# Terminal 1: Launch app
open dist/visarc_ai_server.app

# Terminal 2: View logs in real-time
log stream --predicate 'process == "visarc_ai_server"' --level debug
```

### After a Crash
```bash
# View last 30 minutes of logs
log show --predicate 'process == "visarc_ai_server"' --last 30m

# View only errors and critical
log show --predicate 'process == "visarc_ai_server" AND messageType >= "error"' --last 1h

# Save to file for analysis
log show --predicate 'process == "visarc_ai_server"' --last 1h > crash_logs.txt
```

## What Gets Logged

### ✅ Startup
- Python version and environment
- Bundle location (if PyInstaller)
- SSL certificate path
- System detection
- Binary configuration
- Module loading status

### ✅ Runtime
- API requests
- Model loading/unloading
- Process spawning (llama-server)
- Configuration changes
- Errors and warnings

### ✅ Shutdown
- Graceful shutdown initiation
- Resource cleanup
- Process termination

### ✅ Crashes
- **CRITICAL**: Uncaught exceptions with full stack traces
- Signal handlers (SIGTERM, SIGINT)
- Exit handlers

## Log Format
```
[2026-02-01 14:49:11.320] [INFO] [module.name] Log message here
```

## Enable Debug Mode
```bash
DEBUG=1 open dist/visarc_ai_server.app
```

## Test Logging
```bash
# Run test script
python tests/test_logging.py

# Watch in Console.app while test runs
log stream --predicate 'process == "python"' --level debug
```

## Common Filters

```bash
# All logs from app
process == "visarc_ai_server"

# Errors only
process == "visarc_ai_server" AND messageType >= "error"

# Warnings and above
process == "visarc_ai_server" AND messageType >= "default"

# Specific module
process == "visarc_ai_server" AND category == "app.main"

# Contains text
process == "visarc_ai_server" AND eventMessage CONTAINS "startup"

# Sandbox violations
eventMessage CONTAINS "Sandbox" AND eventMessage CONTAINS "visarc_ai_server"
```

## Crash Report Location
```bash
~/Library/Logs/DiagnosticReports/visarc_ai_server_*.crash
```
