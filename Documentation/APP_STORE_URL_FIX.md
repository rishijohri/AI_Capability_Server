# App Store URL Fix - sobhe.ir/hazm

## Issue
App Store rejected the build due to the presence of the URL `http://www.sobhe.ir/hazm/` in the binary. This URL violates App Store guidelines (Guideline 2.5.1).

## Root Cause
The URL was found in **spaCy** (Natural Language Processing library) files that were being bundled with the application:

1. **spacy/tests/test_displacy.py** - line 319, comment referencing the URL
2. **spacy/lang/fa/** - Persian (Farsi) language support files that reference the Hazm library

The URL appears in test files and Persian language modules that are **not actually used** by the application.

## Solution
The fix involves three layers of protection to ensure these files never make it into the final binary:

### 1. PyInstaller Exclusions (ai_capability.spec & ai_capability_windows.spec)
Added exclusions to prevent PyInstaller from bundling these modules:

```python
excludes = [
    # ... existing excludes ...
    'spacy.tests',        # Test files (contains App Store-prohibited URLs)
    'spacy.lang.fa',      # Persian/Farsi language support (references hazm/sobhe.ir)
]
```

### 2. Data/Binary Filters (Both spec files)
Added pattern matching to filter out any files that slip through:

```python
_blocked_patterns = (
    'spacy/tests', 'spacy\\tests',        # spaCy test files
    'spacy/lang/fa', 'spacy\\lang\\fa',   # Persian language support
)
a.binaries = [(name, path, typecode) for name, path, typecode in a.binaries
              if not any(blocked in name or blocked in path for blocked in _blocked_patterns)]
a.datas = [(name, path, typecode) for name, path, typecode in a.datas
           if not any(blocked in name or blocked in path for blocked in _blocked_patterns)]
```

### 3. Post-Build Cleanup (build.sh)
Added a post-build script that:
- Removes any remaining test or Persian language directories
- Verifies the URL is completely gone from the binary
- Reports the cleanup results

This provides triple-redundancy to ensure App Store compliance.

## Files Modified
- `ai_capability.spec` - macOS/Linux build configuration
- `ai_capability_windows.spec` - Windows build configuration
- `build.sh` - macOS/Linux build script with post-build cleanup

## Verification
After building, the script now automatically verifies:
```bash
Verifying URL removal...
  ✅ Confirmed: sobhe.ir URL completely removed from binary
```

If any files still contain the URL, the build will report a warning with file paths.

## Impact Assessment
**No functionality loss** - The removed components are:
- Test files (never needed in production)
- Persian/Farsi NLP support (not used by this application)

The application only uses:
- English NLP (spaCy's en_core_web_sm model)
- Computer vision (InsightFace, OpenCV)
- Other non-Persian language processing

## Testing
To verify the fix works:

1. **Build the application:**
   ```bash
   ./build.sh
   ```

2. **Check the output** - Should see:
   ```
   Removing App Store prohibited content...
     ✓ Removed: dist/visarc_ai_server/_internal/spacy/tests
     ✓ Removed: dist/visarc_ai_server/_internal/spacy/lang/fa
     ✅ Removed 2 prohibited content directories
   
   Verifying URL removal...
     ✅ Confirmed: sobhe.ir URL completely removed from binary
   ```

3. **Manual verification:**
   ```bash
   # Search for the URL in the built binary (should return no results)
   find dist -type f \( -name "*.py" -o -name "*.pyc" \) -exec grep -l "sobhe.ir" {} \;
   
   # Check for test directories (should not exist)
   ls dist/visarc_ai_server/_internal/spacy/tests 2>/dev/null
   
   # Check for Persian language files (should not exist)
   ls dist/visarc_ai_server/_internal/spacy/lang/fa 2>/dev/null
   ```

## URL is NOT Used in Runtime
The `sobhe.ir/hazm` URL is:
- Only present in comments and documentation strings
- Part of test code that never executes
- In language support modules that are never imported

The URL serves **no functional purpose** in the application and its removal has **zero impact** on functionality.

## App Store Submission Readiness
With these changes, the binary is now compliant with App Store guidelines:
- ✅ No prohibited URLs present
- ✅ No unused test code included
- ✅ No unnecessary language support modules
- ✅ All required functionality intact
- ✅ Automatic verification built into build process

The application can now be submitted to the App Store without URL-related rejections.
