#!/usr/bin/env python3
"""
Example client for AI Server WebSocket endpoints.
Demonstrates basic usage of the API.
"""

import asyncio
import websockets
import json
from pathlib import Path


async def set_storage_metadata(base_url: str, metadata_path: str):
    """Set storage metadata path."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/set-storage-metadata",
            json={"path": metadata_path}
        ) as response:
            data = await response.json()
            print(f"✓ Storage metadata set: {data['message']}")
            return data


async def generate_embeddings(ws_url: str, embedding_model: str):
    """Generate embeddings via WebSocket."""
    uri = f"{ws_url}/api/generate-embeddings"
    
    print(f"\n📊 Generating embeddings using {embedding_model}...")
    
    async with websockets.connect(uri) as websocket:
        # Send configuration
        await websocket.send(json.dumps({
            "embedding_model": embedding_model
        }))
        
        # Receive progress updates
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'progress':
                current = data['data']['current']
                total = data['data']['total']
                filename = data['data']['filename']
                print(f"  [{current}/{total}] Processing: {filename}")
            elif data['type'] == 'status':
                print(f"  {data['message']}")
            elif data['type'] == 'result':
                print(f"✓ {data['message']}")
                print(f"  Generated embeddings for {data['data']['count']} files")
                break
            elif data['type'] == 'error':
                print(f"✗ Error: {data['message']}")
                break


async def generate_rag(ws_url: str):
    """Generate RAG database via WebSocket."""
    uri = f"{ws_url}/api/generate-rag"
    
    print(f"\n🗄️  Building RAG database...")
    
    async with websockets.connect(uri) as websocket:
        # Receive progress updates
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'status':
                print(f"  {data['message']}")
            elif data['type'] == 'result':
                print(f"✓ {data['message']}")
                break
            elif data['type'] == 'error':
                print(f"✗ Error: {data['message']}")
                break


async def tag_files(ws_url: str, file_paths: list, vision_model: str, mmproj_file: str = None):
    """Tag files via WebSocket."""
    uri = f"{ws_url}/api/tag"
    
    print(f"\n🏷️  Tagging files using {vision_model}...")
    
    async with websockets.connect(uri) as websocket:
        # Send initial request
        await websocket.send(json.dumps({
            "file_paths": file_paths,
            "vision_model": vision_model,
            "mmproj_file": mmproj_file
        }))
        
        # Receive updates
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'status':
                print(f"  {data['message']}")
            elif data['type'] == 'result':
                filename = data['data']['filename']
                tags = data['data']['tags']
                print(f"✓ Tagged {filename}: {', '.join(tags[:5])}...")
            elif data['type'] == 'confirmation_needed':
                print(f"  {data['message']}")
                # Auto-continue for demo
                await websocket.send(json.dumps({"action": "continue"}))
            elif data['type'] == 'error':
                print(f"✗ Error: {data['message']}")


async def chat(ws_url: str, chat_model: str, messages: list):
    """Chat with the AI via WebSocket."""
    uri = f"{ws_url}/api/chat"
    
    print(f"\n💬 Starting chat with {chat_model}...")
    
    async with websockets.connect(uri) as websocket:
        # Send initial configuration
        await websocket.send(json.dumps({
            "chat_model": chat_model
        }))
        
        # Wait for ready
        ready = False
        async for message in websocket:
            data = json.loads(message)
            if data['type'] == 'status':
                print(f"  {data['message']}")
                if "ready" in data['message'].lower():
                    ready = True
                    break
            elif data['type'] == 'error':
                print(f"✗ Error: {data['message']}")
                return
        
        if not ready:
            return
        
        # Send messages
        for msg in messages:
            print(f"\n👤 User: {msg}")
            await websocket.send(json.dumps({"message": msg}))
            
            # Receive response
            response_text = ""
            async for message in websocket:
                data = json.loads(message)
                
                if data['type'] == 'status':
                    print(f"  {data['message']}")
                elif data['type'] == 'progress':
                    # Stream response
                    chunk = data['message']
                    print(chunk, end='', flush=True)
                    response_text += chunk
                elif data['type'] == 'result':
                    print(f"\n\n🤖 Assistant: {data['data']['response']}")
                    print(f"  Relevant files: {', '.join(data['data']['relevant_files'][:3])}")
                    break
                elif data['type'] == 'error':
                    print(f"\n✗ Error: {data['message']}")
                    break
        
        # End chat
        await websocket.send(json.dumps({"action": "end"}))


async def main():
    """Main example workflow."""
    # Configuration
    base_url = "http://127.0.0.1:8000"
    ws_url = "ws://127.0.0.1:8000"
    
    # Update these paths for your setup
    metadata_path = "/path/to/storage-metadata.json"
    embedding_model = "embedding-model.gguf"
    vision_model = "vision-model.gguf"
    chat_model = "chat-model.gguf"
    mmproj_file = "mmproj-model.mmproj"
    
    print("=" * 60)
    print("AI Server Example Client")
    print("=" * 60)
    
    try:
        # 1. Set storage metadata
        await set_storage_metadata(base_url, metadata_path)
        
        # 2. Generate embeddings
        await generate_embeddings(ws_url, embedding_model)
        
        # 3. Build RAG
        await generate_rag(ws_url)
        
        # 4. Tag some files (example)
        # await tag_files(ws_url, ["file1.jpg", "file2.mp4"], vision_model, mmproj_file)
        
        # 5. Chat with AI
        chat_messages = [
            "What files do I have?",
            "Tell me about the most recent photos."
        ]
        await chat(ws_url, chat_model, chat_messages)
        
        print("\n" + "=" * 60)
        print("Example workflow completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
