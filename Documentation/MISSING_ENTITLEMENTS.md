# ⚠️ CRITICAL MISSING ENTITLEMENTS

## The Problem: 2.5ms Crash

Your current entitlements are **missing 3 critical ones** that cause instant crashes:

## ❌ What's Missing vs ✅ What You Need

### Your Current Entitlements:
```xml
<key>com.apple.security.app-sandbox</key>
<true/>
<key>com.apple.security.inherit</key>
<true/>
<key>com.apple.security.cs.allow-jit</key>
<true/>
<key>com.apple.security.cs.allow-unsigned-executable-memory</key>
<true/>
<key>com.apple.security.network.client</key>
<true/>
<key>com.apple.security.network.server</key>
<true/>
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
```

### ❌ MISSING (Causes Instant Crash):

#### 1. Disable Library Validation ⚠️ **CRITICAL - #1 CRASH CAUSE**
```xml
<key>com.apple.security.cs.disable-library-validation</key>
<true/>
```

**Why needed:**
- PyInstaller bundles Python extension modules: NumPy, OpenCV, ONNXRuntime, PIL, etc.
- Each is a separate `.dylib` file
- macOS requires **all dylibs to have matching code signatures**
- PyInstaller bundles may not have matching signatures

**Without this:**
```
Library load error: /path/to/numpy.dylib (code signature invalid)
App terminates immediately
```

**This is your 2.5ms crash!**

---

#### 2. Disable Executable Page Protection ⚠️ **CRITICAL**
```xml
<key>com.apple.security.cs.disable-executable-page-protection</key>
<true/>
```

**Why needed:**
- Required to spawn subprocesses (llama-server, llama-cli)
- Allows executing code from spawned processes
- Required for PyInstaller bootloader itself

**Without this:**
```
Cannot execute subprocess: Operation not permitted
```

---

#### 3. `/tmp/` Access (OPTIONAL but recommended)
```xml
<key>com.apple.security.temporary-exception.files.absolute-path.read-write</key>
<array>
    <string>/tmp/</string>
</array>
```

**Why needed:**
- `tempfile.NamedTemporaryFile` in vision_service.py for image processing
- NOT needed for subprocess.Popen (I was wrong - that uses anonymous pipes)

**Without this:**
- Image processing features may fail
- But app will start and run

**Note:** This can be removed if you refactor image processing to use in-memory buffers.

---

## ✅ Complete Fixed Entitlements

Use this file: `entitlements-minimal-sandbox.plist` (just updated)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    
    <key>com.apple.security.inherit</key>
    <true/>
    
    <key>com.apple.security.network.client</key>
    <true/>
    <key>com.apple.security.network.server</key>
    <true/>
    
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
    
    <!-- FIX #1: /tmp/ access -->
    <key>com.apple.security.temporary-exception.files.absolute-path.read-write</key>
    <array>
        <string>/tmp/</string>
        <string>/var/tmp/</string>
        <string>/private/tmp/</string>
        <string>/private/var/tmp/</string>
    </array>
    
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    
    <!-- FIX #2: Library validation (MOST LIKELY CRASH CAUSE) -->
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
    
    <!-- FIX #3: Subprocess execution -->
    <key>com.apple.security.cs.disable-executable-page-protection</key>
    <true/>
</dict>
</plist>
```

---

## 🔧 How to Fix

### 1. Update Your Main App Entitlements

In your VisArc PC app, update the entitlements file used for signing `visarc_ai_server.app` to include the 3 missing ones above.

### 2. Rebuild

```bash
./build.sh
```

### 3. Test

```bash
./test_direct.sh
```

You should now see the app start successfully instead of crashing at 2.5ms!

---

## 🎯 Expected Behavior After Fix

### Before (Current):
```
[16:19:27.558] runningboardd: Checking PreventLaunch
[16:19:27.743] secinitd: AppSandbox request successful
[16:19:28.055] loginwindow: appDeath (312ms later)
```
**No stderr output = Crashed before Python started**

### After (With Fixes):
```
[16:19:27.558] [BOOTSTRAP] VISARC AI SERVER BOOTSTRAP START
[16:19:27.559] [BOOTSTRAP] Python version: 3.11.x
[16:19:27.560] [BOOTSTRAP] Running as PyInstaller bundle
[16:19:27.561] [BOOTSTRAP] SSL: Using bundled certificates
[16:19:27.640] [BOOTSTRAP] app.main imported successfully
[16:19:27.750] [INFO] AI Server starting...
```

---

## 📊 Why Library Validation is the Most Likely Culprit

PyInstaller bundles contain:
- `libpython3.11.dylib`
- `numpy.cpython-311-darwin.so` (dylib)
- `cv2.cpython-311-darwin.so` (dylib)
- `onnxruntime_pybind11_state.cpython-311-darwin.so` (dylib)
- And many more...

Without `disable-library-validation`, macOS checks EVERY dylib signature on load. If ANY don't match the main executable's team ID and signature, **instant crash** before Python even runs.

This matches your symptoms:
- ✅ Clean exit (not a crash exception)
- ✅ Extremely fast (~300ms)
- ✅ No stderr output (crashed during library loading)
- ✅ "AppSandbox request successful" but immediate death

---

## 🚀 Next Steps

1. Add the 3 missing entitlements to your main app's code signing
2. Rebuild: `./build.sh`
3. Test: `./test_direct.sh`
4. The app should now start and show bootstrap logs!
