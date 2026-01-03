"""Test script for cloud-chat WebSocket endpoint."""

import asyncio
import websockets
import json
import sys

async def test_cloud_chat():
    """Test the cloud-chat endpoint."""
    try:
        # Connect to cloud-chat endpoint
        async with websockets.connect('ws://localhost:8000/api/cloud-chat') as ws:
            print("✓ Connected to /api/cloud-chat")
            
            # Wait for ready message
            ready = False
            while not ready:
                message = await ws.recv()
                data = json.loads(message)
                print(f"[{data['type']}] {data['message']}")
                
                if "ready" in data['message'].lower():
                    ready = True
            
            print("\n✓ Server is ready")
            
            # Send a test message
            test_request = {
                "message": "What photos do I have?",
                "history": []
            }
            
            print(f"\nSending request: {test_request['message']}")
            await ws.send(json.dumps(test_request))
            
            # Receive response
            async for message in ws:
                data = json.loads(message)
                print(f"\n[{data['type']}] {data['message']}")
                
                if data['type'] == 'result':
                    print("\n✓ Received RAG context")
                    result_data = data['data']
                    
                    # Verify all expected fields are present
                    expected_fields = [
                        'system_prompt',
                        'rag_context',
                        'relevant_files',
                        'file_details',
                        'image_context',
                        'user_message',
                        'history'
                    ]
                    
                    for field in expected_fields:
                        if field in result_data:
                            print(f"  ✓ {field}: present")
                        else:
                            print(f"  ✗ {field}: MISSING")
                    
                    # Display sample data
                    print(f"\nSystem Prompt (first 100 chars):")
                    print(f"  {result_data.get('system_prompt', '')[:100]}...")
                    
                    print(f"\nRAG Context (first 200 chars):")
                    print(f"  {result_data.get('rag_context', '')[:200]}...")
                    
                    print(f"\nRelevant Files: {len(result_data.get('relevant_files', []))}")
                    for file in result_data.get('relevant_files', [])[:3]:
                        print(f"  - {file}")
                    
                    print("\n✅ Cloud-chat endpoint test PASSED")
                    break
                
                elif data['type'] == 'error':
                    print(f"\n❌ Error: {data['message']}")
                    print("\nNote: Make sure to:")
                    print("  1. Set storage metadata: POST /api/set-storage-metadata")
                    print("  2. Generate embeddings: WS /api/vector-embeddings")
                    print("  3. Build RAG database: WS /api/generate-rag")
                    sys.exit(1)
    
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the AI Server is running on localhost:8000")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Testing /api/cloud-chat endpoint...")
    print("=" * 60)
    asyncio.run(test_cloud_chat())
