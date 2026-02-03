#!/bin/bash
set -e

echo "=============================================="
echo "Signing AI Server .app Bundle for Developer ID"
echo "=============================================="
echo ""

# Define your identity
IDENTITY="Developer ID Application: Rishi Johri (A68CR27KXR)"

APP_BUNDLE="dist/visarc_ai_server.app"
ENTITLEMENTS="entitlements-minimal-sandbox.plist"

if [ ! -d "$APP_BUNDLE" ]; then
    echo "❌ Error: AI server .app bundle not found at $APP_BUNDLE"
    exit 1
fi

if [ ! -f "$ENTITLEMENTS" ]; then
    echo "❌ Error: Entitlements file not found at $ENTITLEMENTS"
    exit 1
fi

echo "Signing AI Server .app bundle with inside-out approach..."
echo ""

# Step 1: Sign all dylibs and shared libraries first
echo "Step 1: Signing shared libraries and dylibs..."
LIB_COUNT=0
if [ -d "$APP_BUNDLE/Contents/Frameworks" ]; then
    while IFS= read -r file; do
        echo "  Signing library: $(basename "$file")"
        codesign --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" --timestamp "$file"
        ((LIB_COUNT++))
    done < <(find "$APP_BUNDLE/Contents/Frameworks" -type f \( -name "*.dylib" -o -name "*.so" \))
    echo "✅ Signed $LIB_COUNT shared libraries"
else
    echo "⚠️  No Frameworks directory found"
fi

# Step 2: Sign all framework binaries (e.g., Python.framework/Versions/3.13/Python)
echo ""
echo "Step 2: Signing framework binaries..."
FRAMEWORK_COUNT=0
if [ -d "$APP_BUNDLE/Contents/Frameworks" ]; then
    # Find framework binaries (executable files inside .framework directories)
    while IFS= read -r framework_dir; do
        FRAMEWORK_NAME=$(basename "$framework_dir" .framework)
        # Look for the binary in Versions/*/FrameworkName or just FrameworkName
        FRAMEWORK_BINARY=""
        if [ -f "$framework_dir/Versions/Current/$FRAMEWORK_NAME" ]; then
            FRAMEWORK_BINARY="$framework_dir/Versions/Current/$FRAMEWORK_NAME"
        elif [ -f "$framework_dir/$FRAMEWORK_NAME" ]; then
            FRAMEWORK_BINARY="$framework_dir/$FRAMEWORK_NAME"
        fi
        
        if [ -n "$FRAMEWORK_BINARY" ] && [ -f "$FRAMEWORK_BINARY" ]; then
            echo "  Signing framework binary: $FRAMEWORK_NAME"
            codesign --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" --timestamp "$FRAMEWORK_BINARY"
            ((FRAMEWORK_COUNT++))
        fi
    done < <(find "$APP_BUNDLE/Contents/Frameworks" -type d -name "*.framework" -depth 1)
    echo "✅ Signed $FRAMEWORK_COUNT framework binaries"
fi

# Step 3: Sign all executables inside Resources (like llama binaries)
echo ""
echo "Step 3: Signing resource executables..."
RES_COUNT=0
if [ -d "$APP_BUNDLE/Contents/Resources" ]; then
    while IFS= read -r file; do
        # Skip the main executable
        if [[ "$(basename "$file")" == "visarc_ai_server" ]]; then
            continue
        fi
        echo "  Signing resource: $(basename "$file")"
        codesign --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" --timestamp "$file"
        ((RES_COUNT++))
    done < <(find "$APP_BUNDLE/Contents/Resources" -type f -perm +111)
    echo "✅ Signed $RES_COUNT resource executables"
fi

# Step 4: Sign the main executable
echo ""
echo "Step 4: Signing main executable..."
MAIN_EXEC="$APP_BUNDLE/Contents/MacOS/visarc_ai_server"
if [ -f "$MAIN_EXEC" ]; then
    codesign --force --options runtime \
        --entitlements "$ENTITLEMENTS" \
        --sign "$IDENTITY" \
        --timestamp "$MAIN_EXEC"
    echo "✅ Signed main executable"
else
    echo "❌ Error: Main executable not found at $MAIN_EXEC"
    exit 1
fi

# Step 5: Sign the entire .app bundle
echo ""
echo "Step 5: Signing .app bundle..."
codesign --force --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$IDENTITY" \
    --timestamp "$APP_BUNDLE"
echo "✅ Signed .app bundle"

echo ""
echo "Code signing completed"
echo ""
echo "Verifying .app bundle signature..."
echo ""

# Verify the .app bundle signature
if codesign --verify --deep --strict "$APP_BUNDLE" 2>/dev/null; then
    SIGN_INFO=$(codesign -dv --verbose=4 "$APP_BUNDLE" 2>&1)
    
    if echo "$SIGN_INFO" | grep -q "Authority=$IDENTITY"; then
        echo "✅ .app bundle properly signed and verified!"
        echo ""
        echo "Signature details:"
        codesign -dv --verbose=4 "$APP_BUNDLE" 2>&1 | grep -E "(Authority|Identifier|Format|TeamIdentifier)"
        echo ""
        echo "==============================================="
        echo "Next step: Build and sign main app with:"
        echo "  ./scripts/build_and_sign_developerid.sh"
        echo "==============================================="
        exit 0
    else
        SIGNER=$(echo "$SIGN_INFO" | grep "Authority=" | head -1 | sed 's/Authority=//')
        echo "❌ .app bundle signed with wrong identity"
        echo "  Expected: $IDENTITY"
        echo "  Found: ${SIGNER:-unsigned}"
        exit 1
    fi
else
    echo "❌ .app bundle signature verification failed!"
    codesign --verify --deep --strict --verbose=4 "$APP_BUNDLE" 2>&1
    exit 1
fi
