# macOS Sandbox SSL Certificate Fix

## Problem
The visarc_ai_server.app was crashing with sandbox violations when trying to access SSL certificates:
```
Sandbox: visarc_ai_server(26219) deny(1) file-read-data /opt/homebrew/etc/ca-certificates/cert.pem
```

This occurred because HTTPS libraries (aiohttp, huggingface-hub, requests) need SSL certificates, which were outside the sandbox's allowed paths.

## Solution Applied

### 1. Created New Entitlements File
**File:** `entitlements-sandbox-ssl.plist`

Added read-only access to SSL certificate paths:
- `/opt/homebrew/etc/ca-certificates/` (Apple Silicon Homebrew)
- `/usr/local/etc/ca-certificates/` (Intel Homebrew)
- `/opt/homebrew/etc/openssl@3/`
- `/etc/ssl/` (system certificates)
- Additional backup paths

### 2. Bundle Certifi Certificates (Primary Solution)
**Files:** `hook-ssl-certifi.py`, `run_server.py`, `ai_capability.spec`

**This is the key fix** - ensures Python uses bundled certificates, not system paths:

**Runtime Hook (`hook-ssl-certifi.py`):**
- Runs BEFORE any Python imports
- Detects bundled certifi certificate location
- Sets `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` environment variables
- Forces all HTTPS libraries to use bundled certs

**Startup Configuration (`run_server.py`):**
- Configures SSL certificates at startup (backup to runtime hook)
- Runs before importing app modules

**Spec File (`ai_capability.spec`):**
- Bundles certifi's `cacert.pem` file in `certifi/` directory
- Adds `hook-ssl-certifi.py` as runtime hook
- Includes certifi in hiddenimports

### 3. Updated PyInstaller Spec
**File:** `ai_capability.spec`

Changes:
- ✅ Set `entitlements_file='entitlements-sandbox-ssl.plist'` in EXE section
- ✅ Added `certifi` and `ssl` to hiddenimports
- ✅ Bundle certifi's CA certificate file with the app
- ✅ Added `runtime_hooks=['hook-ssl-certifi.py']`

### 4. Updated Dependencies
**File:** `requirements.txt`

- ✅ Added `certifi>=2023.0.0` for bundled SSL certificates

## How to Rebuild

```bash
# 1. Install updated dependencies
pip install -r requirements.txt

# 2. Rebuild the app
./build.sh
# OR
pyinstaller ai_capability.spec

# 3. The app will now be at:
dist/visarc_ai_server.app
```

## How It Works

**Two-Layer Protection:**

1. **Primary: Bundled Certificates (Preferred)**
   - Certifi's `cacert.pem` is bundled inside the app
   - Runtime hook sets environment variables before any imports
   - All HTTPS libraries (aiohttp, requests, huggingface-hub) use bundled certs
   - **No system paths accessed** - works in strict sandbox

2. **Fallback: Entitlements (Backup)**
   - If bundled certs fail, entitlements allow read-only access to system paths
   - Homebrew and system certificate paths accessible
   - Prevents app crash if bundled cert is missing

**SSL Configuration Flow:**
```
App Launch
    ↓
Runtime Hook (hook-ssl-certifi.py)
    ↓
Find bundled certifi/cacert.pem
    ↓
Set SSL_CERT_FILE environment variable
    ↓
Import aiohttp, requests, etc.
    ↓
Libraries use bundled certificates ✓
```

## Testing

After rebuilding, test SSL configuration:
```bash
# Launch the bundle
open dist/visarc_ai_server.app

# Test SSL configuration
python tests/test_ssl_config.py

# Check console logs for any sandbox violations
log show --predicate 'process == "visarc_ai_server"' --last 5m
```

## Entitlements Explained

| Entitlement | Purpose |
|-------------|---------|
| `app-sandbox` | Enable macOS sandbox |
| `network.client/server` | Allow FastAPI and llama-server |
| `files.absolute-path.read-only` | Read SSL certificates from Homebrew/system |
| `files.absolute-path.read-write` | Write to /tmp for subprocess pipes |
| `cs.allow-jit` | Python JIT compilation |
| `cs.allow-unsigned-executable-memory` | NumPy, OpenCV, ML libraries |
| `inherit` | Inherit parent app's entitlements |

## Troubleshooting

### Still Getting Sandbox Violations?
1. Check console logs for the exact denied path
2. Add the path to `entitlements-sandbox-ssl.plist` under `files.absolute-path.read-only`
3. Rebuild with `./build.sh`

### SSL Certificate Errors?
- Verify certifi is installed: `pip show certifi`
- Check if certifi bundle is in dist: `ls -la dist/visarc_ai_server/certifi/`

### App Won't Launch?
- Check code signature: `codesign -dv --entitlements - dist/visarc_ai_server.app`
- Verify entitlements are applied: Look for `app-sandbox` in output
