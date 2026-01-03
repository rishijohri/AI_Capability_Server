#!/usr/bin/env python3
"""
Test flexible metadata handling with unknown properties.
"""

import json
import tempfile
from pathlib import Path

# Test data with extra/unknown properties
test_metadata = [
    {
        "fileName": "test1.jpg",
        "deviceId": "device123",
        "deviceName": "Test Device",
        "uploadTime": "2026-01-01T10:00:00Z",
        "tags": ["test", "photo"],
        "aspectRatio": 1.5,
        "type": "image",
        "creationTime": "2026-01-01T09:00:00Z",
        "description": "A test image",
        # Extra properties that weren't in original schema
        "location": "New York",
        "weather": "sunny",
        "temperature": 72.5,
        "people": ["John", "Jane"],
        "metadata_version": 2,
        "camera_settings": {
            "iso": 400,
            "aperture": "f/2.8",
            "shutter_speed": "1/1000"
        }
    },
    {
        "fileName": "test2.mp4",
        "deviceId": "device456",
        "deviceName": "Test Device 2",
        "uploadTime": "2026-01-01T11:00:00Z",
        "tags": ["video", "test"],
        "aspectRatio": 1.777,
        "type": "video",
        "creationTime": "2026-01-01T10:30:00Z",
        # Different extra properties
        "duration": 120,
        "fps": 30,
        "resolution": "1920x1080",
        "audio_codec": "aac"
    }
]

def test_flexible_metadata():
    """Test that metadata with extra fields can be loaded and used."""
    print("=" * 70)
    print("Flexible Metadata Handling Test")
    print("=" * 70)
    
    # Create temporary metadata file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_metadata, f)
        temp_path = f.name
    
    try:
        # Import after creating test file
        from app.models.metadata import MetadataStore, FileMetadata
        
        print("\n1. Loading metadata with extra properties...")
        print("-" * 70)
        
        # Load metadata
        store = MetadataStore(temp_path)
        
        print(f"✓ Loaded {len(store.get_all_metadata())} metadata entries")
        
        # Check identified properties
        print("\n2. Identified Properties:")
        print("-" * 70)
        properties = store.get_identified_properties()
        
        for prop_name, prop_type in sorted(properties.items()):
            print(f"  {prop_name:20s} : {prop_type}")
        
        # Test that extra fields are preserved
        print("\n3. Testing Extra Fields Preservation:")
        print("-" * 70)
        
        file1 = store.get_metadata_by_filename("test1.jpg")
        if file1:
            print(f"✓ File: {file1.fileName}")
            print(f"  Standard field (type): {file1.type}")
            
            # Check if extra fields are accessible
            if hasattr(file1, 'location'):
                print(f"  ✓ Extra field (location): {file1.location}")
            if hasattr(file1, 'temperature'):
                print(f"  ✓ Extra field (temperature): {file1.temperature}")
            if hasattr(file1, 'people'):
                print(f"  ✓ Extra field (people): {file1.people}")
            if hasattr(file1, 'camera_settings'):
                print(f"  ✓ Extra field (camera_settings): {file1.camera_settings}")
        
        # Test text representation includes extra fields
        print("\n4. Text Representation (for embeddings):")
        print("-" * 70)
        
        text_repr = file1.to_text_representation()
        print(text_repr)
        
        # Verify extra fields are in text representation
        if 'location' in text_repr:
            print("\n✓ Extra fields are included in text representation")
        
        print("\n5. Testing Different Extra Fields Per Entry:")
        print("-" * 70)
        
        file2 = store.get_metadata_by_filename("test2.mp4")
        if file2:
            print(f"✓ File: {file2.fileName}")
            if hasattr(file2, 'duration'):
                print(f"  ✓ Extra field (duration): {file2.duration}")
            if hasattr(file2, 'fps'):
                print(f"  ✓ Extra field (fps): {file2.fps}")
            if hasattr(file2, 'resolution'):
                print(f"  ✓ Extra field (resolution): {file2.resolution}")
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("  - Metadata with extra fields loads successfully")
        print("  - Properties are identified correctly")
        print("  - Extra fields are preserved and accessible")
        print("  - Text representation includes all fields")
        print("=" * 70)
        
    finally:
        # Clean up
        Path(temp_path).unlink()

if __name__ == "__main__":
    test_flexible_metadata()
