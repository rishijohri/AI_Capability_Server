"""Test script for deep-chat WebSocket endpoint."""

import asyncio
import websockets
import json
import sys

async def test_deep_chat():
    """Test the deep-chat endpoint with multi-round thinking."""
    try:
        # Connect to deep-chat endpoint
        async with websockets.connect('ws://localhost:8000/api/deep-chat') as ws:
            print("✓ Connected to /api/deep-chat")
            
            # Wait for ready message
            ready = False
            while not ready:
                message = await ws.recv()
                data = json.loads(message)
                print(f"[{data['type']}] {data['message']}")
                
                if "ready" in data['message'].lower():
                    ready = True
            
            print("\n✓ Server is ready")
            
            # Send a test message that should benefit from RAG access
            test_request = {
                "message": "What photos do I have of animals? Can you describe them?",
                "history": []
            }
            
            print(f"\nSending request: {test_request['message']}")
            await ws.send(json.dumps(test_request))
            
            # Receive response
            response_text = ""
            thinking_rounds = 0
            
            async for message in ws:
                data = json.loads(message)
                msg_type = data['type']
                msg_content = data['message']
                
                if msg_type == 'status':
                    print(f"\n[{msg_type}] {msg_content}")
                    if 'thinking round' in msg_content.lower():
                        thinking_rounds += 1
                elif msg_type == 'progress':
                    # Stream response
                    chunk = msg_content
                    print(chunk, end='', flush=True)
                    response_text += chunk
                elif msg_type == 'result':
                    print(f"\n\n✓ Response complete")
                    result_data = data.get('data', {})
                    print(f"Thinking rounds used: {result_data.get('thinking_rounds', 0)}")
                    if 'response' in result_data:
                        print(f"\nFull response:\n{result_data['response']}")
                    break
                elif msg_type == 'error':
                    print(f"\n✗ Error: {msg_content}")
                    if 'data' in data:
                        print(f"Error details: {json.dumps(data['data'], indent=2)}")
                    break
            
            print(f"\n\n✓ Test completed successfully")
            print(f"Total thinking rounds: {thinking_rounds}")
            
    except websockets.exceptions.WebSocketException as e:
        print(f"\n✗ WebSocket error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def test_deep_chat_with_history():
    """Test the deep-chat endpoint with conversation history."""
    try:
        async with websockets.connect('ws://localhost:8000/api/deep-chat') as ws:
            print("\n\n=== Testing Deep Chat with History ===")
            print("✓ Connected to /api/deep-chat")
            
            # Wait for ready
            ready = False
            while not ready:
                message = await ws.recv()
                data = json.loads(message)
                if "ready" in data['message'].lower():
                    ready = True
                    break
            
            # Test with conversation history
            test_request = {
                "message": "What other photos do I have?",
                "history": [
                    {"role": "user", "content": "Show me photos of cats"},
                    {"role": "assistant", "content": "I found 3 photos of cats in your collection."}
                ]
            }
            
            print(f"\nSending request with history: {test_request['message']}")
            await ws.send(json.dumps(test_request))
            
            # Receive response
            async for message in ws:
                data = json.loads(message)
                msg_type = data['type']
                
                if msg_type == 'status':
                    print(f"[{msg_type}] {data['message']}")
                elif msg_type == 'progress':
                    print(data['message'], end='', flush=True)
                elif msg_type == 'result':
                    print(f"\n\n✓ Response complete")
                    break
                elif msg_type == 'error':
                    print(f"\n✗ Error: {data['message']}")
                    break
            
            print(f"\n✓ History test completed")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_deep_chat_with_image():
    """Test the deep-chat endpoint with image (visual chat)."""
    try:
        async with websockets.connect('ws://localhost:8000/api/deep-chat') as ws:
            print("\n\n=== Testing Deep Chat with Image ===")
            print("✓ Connected to /api/deep-chat")
            
            # Wait for ready
            ready = False
            while not ready:
                message = await ws.recv()
                data = json.loads(message)
                if "ready" in data['message'].lower():
                    ready = True
                    break
            
            # Test with image (replace with actual image filename from your metadata)
            test_request = {
                "message": "What can you tell me about this image?",
                "image_name": "sample_image.jpg",  # Replace with actual image filename
                "history": []
            }
            
            print(f"\nSending request with image: {test_request['message']}")
            await ws.send(json.dumps(test_request))
            
            # Receive response
            async for message in ws:
                data = json.loads(message)
                msg_type = data['type']
                
                if msg_type == 'status':
                    print(f"[{msg_type}] {data['message']}")
                elif msg_type == 'progress':
                    print(data['message'], end='', flush=True)
                elif msg_type == 'result':
                    print(f"\n\n✓ Response complete")
                    break
                elif msg_type == 'error':
                    print(f"\n✗ Error: {data['message']}")
                    break
            
            print(f"\n✓ Image test completed")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=== Deep Chat WebSocket Test Suite ===\n")
    
    # Run basic test
    asyncio.run(test_deep_chat())
    
    # Run test with history
    asyncio.run(test_deep_chat_with_history())
    
    # Uncomment to test with image (requires valid image in metadata)
    # asyncio.run(test_deep_chat_with_image())
    
    print("\n=== All tests completed ===")
