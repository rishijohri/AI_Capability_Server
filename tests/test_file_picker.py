"""Test the file picker endpoint and permission handling."""

import requests
import json


# Base URL
BASE_URL = "http://localhost:8000"


def test_file_picker():
    """Test the select-storage-metadata endpoint with directory picker dialog."""
    
    print("=" * 60)
    print("Testing Directory Picker Endpoint")
    print("=" * 60)
    
    # Call the endpoint that opens directory picker
    print("\n📂 Opening directory picker dialog...")
    print("Please select your storage folder (containing storage_metadata.json) in the dialog that appears.")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/select-storage-metadata",
            timeout=300  # 5 minute timeout for user to select file
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            print(f"Selected folder: {result['data']['selected_folder']}")
            print(f"Metadata file: {result['data']['metadata_file']}")
            print(f"Metadata count: {result['data']['metadata_count']}")
            print(f"RAG directory: {result['data']['rag_directory']}")
            print(f"Embeddings loaded: {result['data']['embeddings_loaded']}")
            if result['data']['embeddings_loaded']:
                print(f"Embeddings count: {result['data']['embeddings_count']}")
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(response.json())
            
    except requests.RequestException as e:
        print(f"\n❌ Request failed: {str(e)}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def test_direct_path():
    """Test the traditional set-storage-metadata endpoint with auto-permission fallback."""
    
    print("\n" + "=" * 60)
    print("Testing Direct Path with Auto-Permission Handling")
    print("=" * 60)
    
    # You need to provide the path directly
    metadata_path = input("\nEnter path to storage_metadata.json (or press Enter to skip): ").strip()
    
    if not metadata_path:
        print("Skipped.")
        return
    
    print("\n📝 Attempting to access the path directly...")
    print("If permission is denied, a dialog will appear automatically.")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/set-storage-metadata",
            json={"path": metadata_path}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS!")
            
            # Check if permission was granted via picker
            if result['data'].get('permission_granted_via_picker'):
                print("⚠️  Permission was granted via folder selection dialog")
                print(f"Selected folder: {result['data'].get('selected_folder')}")
            else:
                print("✓ Direct access granted (no permission issues)")
                
            print(f"Metadata count: {result['data']['metadata_count']}")
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(response.json())
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


def test_permission_scenario():
    """Test permission denied scenario with automatic fallback."""
    
    print("\n" + "=" * 60)
    print("Testing Permission Denied Auto-Recovery")
    print("=" * 60)
    print("\nThis test demonstrates automatic permission handling:")
    print("1. Try to access a path on external drive/restricted location")
    print("2. If denied, dialog automatically opens")
    print("3. User selects folder to grant permission")
    print("4. Operation retries automatically")
    
    test_direct_path()


if __name__ == "__main__":
    print("\nFile Picker & Permission Handling Test")
    print("=" * 60)
    print("\nChoose test method:")
    print("1. Directory Picker Dialog (recommended - grants access to entire folder)")
    print("2. Direct Path Entry (demonstrates auto-permission fallback)")
    print("3. Permission Denied Scenario (shows automatic recovery)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        test_file_picker()
    elif choice == "2":
        test_direct_path()
    elif choice == "3":
        test_permission_scenario()
    else:
        print("Invalid choice. Using directory picker...")
        test_file_picker()

