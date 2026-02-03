# Define your identity

IDENTITY="Developer ID Application: Rishi Johri (A68CR27KXR)"

PY_DIST_DIR="dist/visarc_ai_server"

# Entitlements file
ENTITLEMENTS="entitlements.plist"

# Sign the main executable first
echo "Signing main executable..."
codesign --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$PY_DIST_DIR/visarc_ai_server"

# Find and sign all dynamic libraries and other executables
# This handles the binaries and llama_binaries mentioned in your spec
echo ""
echo "Signing libraries and binaries..."

find "$PY_DIST_DIR" -type f \( -name "*.dylib" -o -name "*.so" \) -o \( -type f -perm +111 ! -name "visarc_ai_server" \) | while read -r file; do

    echo "Signing $file"

    codesign --force --options runtime --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$file"
done

echo "Code signing completed with entitlements"
echo ""
echo "Verifying signatures..."
echo ""

# Verification counters
VERIFIED=0
FAILED=0
FAILED_FILES=()

# Create a temporary file list
TEMP_FILE_LIST=$(mktemp)
find "$PY_DIST_DIR" -type f \( -name "*.dylib" -o -name "*.so" -o -perm +111 \) > "$TEMP_FILE_LIST"

# Verify all signed files
while IFS= read -r file; do
    # Verify the signature
    if codesign --verify --deep --strict "$file" 2>/dev/null; then
        # Get detailed signing info
        SIGN_INFO=$(codesign -dv --verbose=4 "$file" 2>&1)
        
        # Check if signed with correct identity (look for Authority line)
        if echo "$SIGN_INFO" | grep -q "Authority=$IDENTITY"; then
            echo "✓ Valid: $file"
            ((VERIFIED++))
        else
            # Extract what it was actually signed with
            SIGNER=$(echo "$SIGN_INFO" | grep "Authority=" | head -1 | sed 's/Authority=//')
            echo "✗ Wrong identity: $file"
            echo "  Expected: $IDENTITY"
            echo "  Found: ${SIGNER:-unsigned}"
            ((FAILED++))
            FAILED_FILES+=("$file")
        fi
    else
        echo "✗ Invalid signature: $file"
        ((FAILED++))
        FAILED_FILES+=("$file")
    fi
done < "$TEMP_FILE_LIST"

rm -f "$TEMP_FILE_LIST"

# Verify main executable
echo ""
echo "Verifying main executable..."
if codesign --verify --deep --strict "$PY_DIST_DIR/visarc_ai_server" 2>/dev/null; then
    SIGN_INFO=$(codesign -dv --verbose=4 "$PY_DIST_DIR/visarc_ai_server" 2>&1)
    
    if echo "$SIGN_INFO" | grep -q "Authority=$IDENTITY"; then
        echo "✓ Main executable properly signed"
        ((VERIFIED++))
    else
        SIGNER=$(echo "$SIGN_INFO" | grep "Authority=" | head -1 | sed 's/Authority=//')
        echo "✗ Main executable signed with wrong identity"
        echo "  Expected: $IDENTITY"
        echo "  Found: ${SIGNER:-unsigned}"
        ((FAILED++))
        FAILED_FILES+=("$PY_DIST_DIR/visarc_ai_server")
    fi
else
    echo "✗ Main executable signature invalid"
    ((FAILED++))
    FAILED_FILES+=("$PY_DIST_DIR/visarc_ai_server")
fi

echo ""
echo "==============================================="
echo "Verification Summary:"
echo "  Verified: $VERIFIED files"
echo "  Failed: $FAILED files"
echo "==============================================="

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  WARNING: $FAILED file(s) failed verification!"
    echo ""
    echo "Failed files:"
    printf '  - %s\n' "${FAILED_FILES[@]}"
    exit 1
else
    echo ""
    echo "✓ All files properly signed and verified!"
fi

echo ""
echo "==============================================="
echo "Notarizing bundle..."
echo "==============================================="
echo ""

# Create a zip for notarization
ZIP_FILE="visarc_ai_server.zip"
echo "Creating zip archive for notarization..."
cd dist
ditto -c -k --keepParent visarc_ai_server "$ZIP_FILE"
cd ..

# Submit for notarization
echo "Submitting to Apple for notarization..."
xcrun notarytool submit "dist/$ZIP_FILE" --keychain-profile "visarc-notarization" --wait

# Check if notarization succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Notarization successful!"
    echo "Stapling notarization ticket to bundle..."
    
    xcrun stapler staple "$PY_DIST_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ Notarization ticket stapled successfully!"
        
        # Verify with spctl
        echo ""
        echo "Verifying Gatekeeper acceptance..."
        spctl --assess --verbose=4 --type execute "$PY_DIST_DIR/visarc_ai_server"
        
        if [ $? -eq 0 ]; then
            echo "✓ Bundle accepted by Gatekeeper!"
        else
            echo "⚠️  Gatekeeper assessment failed, but notarization was successful"
        fi
    else
        echo "⚠️  Failed to staple notarization ticket"
    fi
    
    # Clean up zip
    rm -f "dist/$ZIP_FILE"
else
    echo ""
    echo "✗ Notarization failed!"
    echo "Check the notarization log with: xcrun notarytool log <submission-id> --keychain-profile visarc-notarization"
    rm -f "dist/$ZIP_FILE"
    exit 1
fi

echo ""
echo "==============================================="
echo "✓ Signing and notarization complete!"
echo "==============================================="