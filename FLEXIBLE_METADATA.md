# Flexible Metadata Handling

## Overview

The AI Server now supports flexible metadata handling that gracefully accepts and processes unknown or new metadata properties without breaking. This ensures forward compatibility as your metadata schema evolves.

## Key Features

### 1. **Automatic Property Detection**
- When loading `storage_metadata.json`, the system automatically identifies all properties across all entries
- Property types are detected (string, number, integer, boolean, array, object, null)
- Property information is reported in the `set-storage-metadata` response

### 2. **Graceful Handling of Extra Fields**
- Metadata entries can have different properties
- New properties are automatically accepted without schema changes
- No errors when encountering previously unseen properties

### 3. **Complete Property Preservation**
- All metadata properties (standard and extra) are:
  - Preserved in memory
  - Included in embedding generation
  - Sent to LLM as context during RAG queries
  - Available for all metadata operations

## API Changes

### `POST /api/set-storage-metadata`

**New Response Field: `identified_properties`**

```json
{
  "status": "success",
  "message": "Storage metadata set to /path/to/storage_metadata.json",
  "data": {
    "metadata_count": 100,
    "rag_directory": "/path/to/rag",
    "embeddings_loaded": true,
    "embeddings_count": 100,
    "identified_properties": {
      "fileName": "string",
      "deviceId": "string",
      "type": "string",
      "tags": "array[string]",
      "creationTime": "string",
      "description": "string",
      "location": "string",
      "weather": "string",
      "temperature": "number",
      "people": "array[string]",
      "camera_settings": "object"
    }
  }
}
```

## Example Metadata with Extra Fields

```json
[
  {
    "fileName": "photo1.jpg",
    "deviceId": "device123",
    "deviceName": "iPhone 15",
    "uploadTime": "2026-01-01T10:00:00Z",
    "creationTime": "2026-01-01T09:30:00Z",
    "type": "image",
    "tags": ["vacation", "beach"],
    "aspectRatio": 1.5,
    "description": "Beautiful sunset at the beach",
    
    // Extra fields - automatically handled
    "location": "Malibu, CA",
    "weather": "sunny",
    "temperature": 75.5,
    "people": ["John", "Jane"],
    "gps_coordinates": {
      "latitude": 34.0259,
      "longitude": -118.7798
    },
    "camera_settings": {
      "iso": 100,
      "aperture": "f/2.8",
      "shutter_speed": "1/2000"
    }
  },
  {
    "fileName": "video1.mp4",
    "deviceId": "device456",
    "deviceName": "GoPro Hero 12",
    "uploadTime": "2026-01-01T11:00:00Z",
    "creationTime": "2026-01-01T10:30:00Z",
    "type": "video",
    "tags": ["action", "sports"],
    "aspectRatio": 1.777,
    
    // Different extra fields for this entry
    "duration": 120,
    "fps": 60,
    "resolution": "4K",
    "stabilization": true,
    "audio_codec": "aac"
  }
]
```

## How It Works

### 1. **FileMetadata Model**
```python
class FileMetadata(BaseModel):
    model_config = {"extra": "allow"}  # Allows extra fields
    
    # Standard fields
    fileName: str
    type: str
    # ... other standard fields
```

The `extra: "allow"` configuration tells Pydantic to accept and store additional fields not explicitly defined in the model.

### 2. **Embedding Generation**

Extra fields are automatically included in the text representation used for embeddings:

```python
def to_text_representation(self) -> str:
    """Includes all fields, including extra ones."""
    parts = [
        "File: photo1.jpg",
        "Type: image",
        "Created: 2026-01-01T09:30:00Z",
        "Tags: vacation, beach",
        "Description: Beautiful sunset at the beach",
        # Extra fields included:
        "location: Malibu, CA",
        "weather: sunny",
        "temperature: 75.5",
        "people: John, Jane",
        "camera_settings: {\"iso\": 100, \"aperture\": \"f/2.8\"}"
    ]
```

### 3. **RAG Context**

When sending relevant files to the LLM, all metadata (including extra fields) is included:

```
Here are relevant files from the knowledge base:

- photo1.jpg
  Type: image
  Created: 2026-01-01T09:30:00Z
  Tags: vacation, beach
  Description: Beautiful sunset at the beach
  location: Malibu, CA
  weather: sunny
  temperature: 75.5
  people: John, Jane
  camera_settings: {"iso": 100, "aperture": "f/2.8", "shutter_speed": "1/2000"}
```

## Property Type Detection

The system automatically detects property types:

| Type | Examples | Detection |
|------|----------|-----------|
| `string` | `"text"`, `"2026-01-01"` | String values |
| `integer` | `42`, `100` | Whole numbers |
| `number` | `3.14`, `75.5` | Decimal numbers |
| `boolean` | `true`, `false` | Boolean values |
| `array[string]` | `["a", "b"]` | Array of strings |
| `array` | `[1, 2, 3]` | Other arrays |
| `object` | `{"key": "value"}` | JSON objects |
| `null` | `null` | Null values |

## Benefits

### ✅ **Forward Compatibility**
Add new metadata properties without updating code or schemas.

### ✅ **Flexible Schema**
Different files can have different metadata properties as needed.

### ✅ **No Data Loss**
All metadata properties are preserved and utilized throughout the system.

### ✅ **Better Context for LLM**
More metadata means better, more informed LLM responses.

### ✅ **Comprehensive Embeddings**
Embeddings capture all available information about files.

## Testing

Run the test to verify flexible metadata handling:

```bash
python3 test_flexible_metadata.py
```

This test:
- Creates metadata with extra/unknown fields
- Loads it into the system
- Verifies all properties are preserved
- Checks that embeddings include all fields
- Confirms property type detection

## Backward Compatibility

✅ **Fully backward compatible**
- Existing metadata files work without changes
- Standard fields continue to work as before
- No breaking changes to existing functionality

## Migration Guide

**No migration needed!** 

Your existing `storage_metadata.json` files will work as-is. To take advantage of flexible metadata:

1. Simply add new properties to your metadata entries
2. The system will automatically detect and use them
3. Check the `identified_properties` in the response to see what was found

## Technical Details

### Pydantic v2 Extra Fields

In Pydantic v2, extra fields are stored in `__pydantic_extra__`:

```python
file_meta = FileMetadata(fileName="test.jpg", location="NYC")
print(file_meta.__pydantic_extra__)  # {'location': 'NYC'}
```

### Accessing Extra Fields

Extra fields can be accessed as normal attributes:

```python
if hasattr(file_meta, 'location'):
    print(file_meta.location)  # "NYC"
```

### Serialization

Extra fields are included in serialization:

```python
file_meta.model_dump()  # Includes all fields
```

## Related Files

- `app/models/metadata.py` - FileMetadata model with extra field support
- `app/api/routes.py` - Updated to include extra fields in context
- `test_flexible_metadata.py` - Comprehensive test suite
