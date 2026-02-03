# Automatic Permission Handling

## Overview

The AI Capability server now includes **automatic permission handling** for macOS sandbox restrictions. When the app encounters a permission denied error, it automatically opens a native folder picker dialog, allowing users to grant access without manual intervention.

## How It Works

### 1. **Automatic Fallback on Permission Errors**

When any file operation encounters a permission error:
- ✅ Error is detected automatically
- 📂 Native folder picker dialog opens at the problematic location
- 👤 User selects the folder to grant permission
- 🔄 Operation retries automatically with the granted permission

### 2. **Where It Applies**

The automatic permission handling works for:

- **Storage Metadata Access** - `/api/set-storage-metadata`
- **Model Directory Access** - When loading LLM models
- **File Operations** - Reading images, videos, embeddings
- **RAG Database** - Creating/accessing vector databases
- **Any File System Access** - Throughout the application

### 3. **API Endpoints**

#### Select Storage Folder (Recommended)
```bash
POST /api/select-storage-metadata
```
Opens a folder picker dialog upfront. User selects the storage folder, granting access to:
- `storage_metadata.json`
- `files/` directory
- `saved_llm/` models
- `rag/` database
- Any future subdirectories

**Response:**
```json
{
  "status": "success",
  "message": "Storage folder set to /Volumes/WD_Rishi/Remote_1",
  "data": {
    "selected_folder": "/Volumes/WD_Rishi/Remote_1",
    "metadata_file": "/Volumes/WD_Rishi/Remote_1/storage_metadata.json",
    "metadata_count": 150,
    "rag_directory": "/Volumes/WD_Rishi/Remote_1/rag",
    "saved_llm_directory": "/Volumes/WD_Rishi/Remote_1/saved_llm"
  }
}
```

#### Set Storage Metadata (With Auto-Fallback)
```bash
POST /api/set-storage-metadata
Content-Type: application/json

{
  "path": "/Volumes/WD_Rishi/Remote_1/storage_metadata.json"
}
```

If permission is denied:
1. Dialog automatically opens at `/Volumes/WD_Rishi/Remote_1/`
2. User selects the folder
3. Operation retries automatically

**Response (when permission granted via picker):**
```json
{
  "status": "success",
  "message": "Permission granted. Storage metadata set to...",
  "data": {
    "permission_granted_via_picker": true,
    "selected_folder": "/Volumes/WD_Rishi/Remote_1",
    "metadata_count": 150
  }
}
```

## Benefits

### ✅ Security
- User explicitly grants permissions
- No hardcoded paths in entitlements
- Follows macOS security best practices

### ✅ User Experience
- No cryptic "Operation not permitted" errors
- Seamless permission granting
- Works on external drives (including your WD drive)

### ✅ Flexibility
- Works with any storage location
- Automatic retry on success
- Graceful error handling

## Implementation Details

### File Picker Utility
Located in `app/utils/file_picker.py`:

```python
from app.utils.file_picker import (
    open_directory_picker,           # Manual folder selection
    is_permission_error,              # Check if error is permission-related
    handle_permission_error_with_picker  # Auto-handle permission errors
)
```

### Permission Handler
Located in `app/utils/permission_handler.py`:

```python
from app.utils.permission_handler import (
    with_permission_fallback,  # Decorator for functions
    safe_file_operation        # Wrapper for file operations
)

# Example usage:
@with_permission_fallback("model directory")
def load_model(path):
    return open(path).read()
```

### MetadataStore Auto-Permission
The `MetadataStore` class automatically handles permission errors:

```python
# Automatically prompts for permission on access error
store = MetadataStore("/Volumes/WD_Rishi/Remote_1/storage_metadata.json")
```

## Testing

Run the test suite:
```bash
python tests/test_file_picker.py
```

**Test Options:**
1. **Directory Picker** - Opens dialog immediately
2. **Direct Path** - Demonstrates auto-fallback on permission error
3. **Permission Scenario** - Full permission denied recovery workflow

## macOS Entitlements

The app uses minimal entitlements (`entitlements-minimal.plist`):

```xml
<key>com.apple.security.files.user-selected.read-write</key>
<true/>
```

This allows access to **any location** the user explicitly selects via the file picker, without requiring:
- ❌ External volume paths in entitlements
- ❌ Temporary exception paths
- ❌ Full disk access

## Examples

### Example 1: First-Time Setup
```python
import requests

# User calls this endpoint
response = requests.post("http://localhost:8000/api/select-storage-metadata")

# Dialog opens -> user selects /Volumes/WD_Rishi/Remote_1/
# Full access granted to entire folder!
```

### Example 2: Permission Denied Recovery
```python
# Try to access restricted path
response = requests.post(
    "http://localhost:8000/api/set-storage-metadata",
    json={"path": "/Volumes/WD_Rishi/Remote_1/storage_metadata.json"}
)

# Automatic flow:
# 1. Permission denied
# 2. Dialog opens at /Volumes/WD_Rishi/Remote_1/
# 3. User selects folder
# 4. Operation retries
# 5. Success!
```

### Example 3: Flutter/Frontend Integration
```dart
// In your Flutter app
Future<void> selectStorageFolder() async {
  final response = await http.post(
    Uri.parse('http://localhost:8000/api/select-storage-metadata')
  );
  
  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    print('Storage set to: ${data['data']['selected_folder']}');
  }
}
```

## Troubleshooting

### Dialog doesn't appear
- Ensure tkinter is available (bundled with app)
- Check app is not running in background mode
- Try using `/api/select-storage-metadata` endpoint

### Permission still denied after selection
- Ensure you selected the correct parent folder
- Check the folder contains `storage_metadata.json`
- Restart the app after granting permission

### External drive not accessible
- The file picker dialog automatically grants access!
- No need for entitlement modifications
- Works with USB drives, network shares, etc.

## Related Files

- `app/utils/file_picker.py` - Core picker functionality
- `app/utils/permission_handler.py` - Permission wrapper utilities
- `app/api/routes.py` - API endpoints with permission handling
- `app/models/metadata.py` - MetadataStore with auto-permission
- `entitlements-minimal.plist` - Minimal sandbox entitlements
- `tests/test_file_picker.py` - Test suite
