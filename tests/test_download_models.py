"""Test script for model download endpoint."""

import asyncio
import json
import websockets


async def test_download_models():
    """Test downloading models from Hugging Face."""
    
    # Example model_ids - you'll need to configure repo_id in settings.py first
    model_ids = ["qwen_3_0.6B"]  # Start with a small model for testing
    
    uri = "ws://localhost:8000/api/download-models"
    
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send download request
            request = {
                "model_ids": model_ids,
                "force_redownload": False  # Set to True to re-download existing models
            }
            
            print(f"Sending request: {json.dumps(request, indent=2)}")
            await websocket.send(json.dumps(request))
            
            # Receive messages
            print("\n--- Download Progress ---")
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                msg = data.get("message")
                msg_data = data.get("data")
                
                if msg_type == "status":
                    print(f"[STATUS] {msg}")
                    
                elif msg_type == "progress":
                    print(f"[PROGRESS] {msg}")
                    if msg_data:
                        filename = msg_data.get("filename")
                        bytes_dl = msg_data.get("bytes_downloaded", 0)
                        total_bytes = msg_data.get("total_bytes", 0)
                        if total_bytes > 0:
                            percent = (bytes_dl / total_bytes) * 100
                            print(f"  -> {filename}: {bytes_dl / (1024**2):.2f} MB / {total_bytes / (1024**2):.2f} MB ({percent:.1f}%)")
                
                elif msg_type == "result":
                    print(f"\n[RESULT] {msg}")
                    if msg_data:
                        print(f"  Model ID: {msg_data.get('model_id')}")
                        print(f"  Overall Status: {msg_data.get('overall_status')}")
                        files = msg_data.get('files', [])
                        for file_status in files:
                            print(f"  - {file_status['filename']}: {file_status['status']}")
                            if file_status.get('error'):
                                print(f"    Error: {file_status['error']}")
                
                elif msg_type == "error":
                    print(f"[ERROR] {msg}")
                    if msg_data:
                        print(f"  Details: {json.dumps(msg_data, indent=2)}")
            
            print("\n--- Download Complete ---")
            
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    print("""
    Model Download Test Script
    ==========================
    
    Before running this test:
    1. Start the AI Capability server (python run_server.py)
    2. Edit app/config/settings.py and add repo_id for models you want to download
       Example:
       "qwen_3_0.6B": {
           "model_file": "Qwen3-0.6B-Q4_K_M.gguf",
           "name": "qwen_3_0.6B",
           "type": "chat",
           "repo_id": "Qwen/Qwen3-0.6B-GGUF"  # Add this line
       }
    3. Install huggingface_hub: pip install huggingface_hub
    
    """)
    
    asyncio.run(test_download_models())
