# SSL Certificate Resolution in Sandboxed App

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    visarc_ai_server.app                         │
│                         (Sandboxed)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. App Launch                                                  │
│     ↓                                                           │
│  2. PyInstaller Runtime Hook (hook-ssl-certifi.py)             │
│     • Runs BEFORE any imports                                   │
│     • Searches for: _MEIPASS/certifi/cacert.pem                │
│     • Sets environment variables:                               │
│       - SSL_CERT_FILE                                           │
│       - REQUESTS_CA_BUNDLE                                      │
│       - CURL_CA_BUNDLE                                          │
│     ↓                                                           │
│  3. run_server.py (Backup SSL Config)                          │
│     • Double-checks SSL configuration                           │
│     • Fallback if runtime hook fails                           │
│     ↓                                                           │
│  4. Import Python Libraries                                     │
│     • aiohttp                                                   │
│     • huggingface-hub                                           │
│     • requests                                                  │
│     • urllib3                                                   │
│     ↓                                                           │
│  5. Libraries Check SSL_CERT_FILE                              │
│     • Find: /path/to/bundle/certifi/cacert.pem                 │
│     • USE BUNDLED CERTIFICATES ✓                                │
│     • NO system path access needed!                             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Bundled Files:                                                 │
│  • certifi/cacert.pem (Mozilla CA bundle)                      │
│  • All Python dependencies                                      │
│  • model/ directory                                             │
│  • binary/ directory                                            │
└─────────────────────────────────────────────────────────────────┘

          ↓ (If bundled cert missing - FALLBACK)

┌─────────────────────────────────────────────────────────────────┐
│              macOS Sandbox (Entitlements)                       │
├─────────────────────────────────────────────────────────────────┤
│  Allowed Read-Only Paths:                                       │
│  • /opt/homebrew/etc/ca-certificates/                          │
│  • /usr/local/etc/ca-certificates/                             │
│  • /etc/ssl/                                                    │
│                                                                 │
│  ⚠️ Only used if bundled cert fails                            │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Works

### Problem
Python HTTPS libraries look for SSL certificates in this order:
1. `SSL_CERT_FILE` environment variable
2. `certifi.where()` (usually points to system certifi)
3. System default paths (`/etc/ssl`, `/opt/homebrew/etc/...`)

In a sandbox, paths 2 and 3 are blocked → **App crashes**

### Solution
Set `SSL_CERT_FILE` BEFORE any library imports:
- Libraries check environment variable first
- Find bundled certificate inside app
- Never try to access system paths
- **Sandbox happy!** ✓

## File Roles

| File | Purpose |
|------|---------|
| `hook-ssl-certifi.py` | **Primary fix** - Runtime hook that configures SSL before any imports |
| `run_server.py` | Backup SSL configuration + app initialization |
| `ai_capability.spec` | Bundles certifi, adds runtime hook |
| `entitlements-sandbox-ssl.plist` | Fallback permissions for system cert paths |
| `requirements.txt` | Ensures certifi is installed |

## Verify It Works

```bash
# After building, check bundled cert exists
ls -la dist/visarc_ai_server/certifi/cacert.pem

# Should show: ~400KB file (Mozilla CA bundle)
```

## Testing Checklist

- [ ] Build app: `./build.sh`
- [ ] Verify bundled cert: `ls dist/visarc_ai_server/certifi/`
- [ ] Run test: `python tests/test_ssl_config.py`
- [ ] Launch app: `open dist/visarc_ai_server.app`
- [ ] Check console: No "deny file-read-data /opt/homebrew" errors
- [ ] Test HTTPS: Verify API requests work

## Common Issues

### "SSL: CERTIFICATE_VERIFY_FAILED"
- Bundled cert is missing
- Check: `ls dist/visarc_ai_server/certifi/cacert.pem`
- Fix: Ensure certifi is installed before building

### "Sandbox: deny(1) file-read-data /opt/homebrew/..."
- Runtime hook didn't run
- Check: `runtime_hooks=['hook-ssl-certifi.py']` in spec file
- Fix: Rebuild with updated spec

### App still crashes on launch
- Check entitlements were applied
- Run: `codesign -dv --entitlements - dist/visarc_ai_server.app`
- Should see `app-sandbox` and SSL paths

## Key Insight

**Environment variables are set BEFORE Python imports**, so all HTTPS libraries automatically use the bundled certificate. No code changes needed in app logic!
