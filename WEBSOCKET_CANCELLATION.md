# WebSocket Disconnection and Resource Management

## Overview

The AI Capability Server now properly handles WebSocket disconnections when using CLI mode (`llm_mode: "cli"`). When a client disconnects during an LLM generation task, the system immediately cancels the running subprocess to prevent wasting computational resources.

## Implementation Details

### 1. Subprocess Tracking in LlamaCLIBackend

The `LlamaCLIBackend` class now tracks the currently running subprocess:

```python
class LlamaCLIBackend(LLMBackend):
    def __init__(self):
        # ... other initialization ...
        self._current_process: Optional[asyncio.subprocess.Process] = None
        self._process_lock = asyncio.Lock()
```

### 2. Cancellation Method

A new `cancel_generation()` method allows immediate termination of running processes:

```python
async def cancel_generation(self) -> None:
    """Cancel any running generation process."""
    async with self._process_lock:
        if self._current_process is not None:
            try:
                # Try graceful termination first
                self._current_process.terminate()
                try:
                    await asyncio.wait_for(self._current_process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Force kill if graceful termination times out
                    self._current_process.kill()
                    await self._current_process.wait()
            except ProcessLookupError:
                # Process already terminated
                pass
            finally:
                self._current_process = None
```

### 3. WebSocket Disconnection Handling

The `/chat` WebSocket endpoint now properly detects disconnections and cancels tasks:

```python
@router.websocket("/chat")
async def chat_ws(websocket: WebSocket):
    generation_task = None
    llm_service = None
    
    try:
        # ... generation logic ...
        generation_task = asyncio.create_task(generate_response())
        await generation_task
        
    except WebSocketDisconnect:
        # Client disconnected - cancel any running generation
        if generation_task and not generation_task.done():
            generation_task.cancel()
            
        # If using CLI mode, explicitly cancel the subprocess
        if llm_service and config.llm_mode == "cli":
            if hasattr(llm_service.current_backend, 'cancel_generation'):
                await llm_service.current_backend.cancel_generation()
        
        # Clean up resources
        if llm_service:
            await llm_service.unload_model()
```

## How It Works

1. **Subprocess Tracking**: Every CLI subprocess created for generation, embedding, or vision tasks is tracked in `_current_process`

2. **Generation as Task**: The generation process runs as an `asyncio.Task`, which can be cancelled

3. **Disconnect Detection**: When a WebSocket disconnects (via `WebSocketDisconnect` exception), the system:
   - Cancels the generation task
   - Calls `cancel_generation()` on the CLI backend
   - Cleans up the subprocess

4. **Graceful Termination**: The subprocess is terminated gracefully with `SIGTERM`, with a 2-second timeout before forcing `SIGKILL`

5. **Thread Safety**: A lock (`_process_lock`) ensures thread-safe access to the subprocess reference

## Benefits

- **Resource Efficiency**: No wasted CPU/GPU cycles on abandoned requests
- **Memory Management**: Subprocess cleanup prevents memory leaks
- **Fast Response**: Immediate termination when client disconnects
- **Graceful Handling**: Proper cleanup of all resources

## Testing

Run the test script to verify cancellation works:

```bash
python test_cli_cancellation.py
```

Expected output:
```
Testing CLI subprocess cancellation...
Starting CLI backend...
Backend started successfully

Starting generation task...
Generation started, checking if process is running...
✓ Subprocess tracked with PID: <pid>

Simulating WebSocket disconnect...
✓ Generation task cancelled
Cancelling subprocess...
✓ Subprocess successfully terminated and cleaned up

✓ Test passed: CLI subprocess cancellation works correctly
```

## Configuration

### LLM Mode
This feature works automatically when `llm_mode` is set to `"cli"`:

```json
{
  "llm_mode": "cli"
}
```

### Timeout Configuration
Control how long the server waits for LLM operations with the `llm_timeout` setting (default: 300 seconds):

```json
{
  "llm_timeout": 300
}
```

- **Range**: 10-3600 seconds
- **Default**: 300 seconds (5 minutes)
- **Applies to**: All LLM operations (chat, tagging, description, embeddings)
- **Behavior**: When timeout is reached, the operation is cancelled and an error message is sent via WebSocket

## Impact on All WebSocket Endpoints

This implementation applies to all WebSocket endpoints that use LLM tasks:

### Chat Endpoint (`/chat`)
- Wraps text and vision generation in cancellable tasks
- Handles WebSocket disconnections during streaming responses
- Cancels CLI subprocess when client disconnects

### Tag Endpoint (`/tag`)
- Wraps vision tagging tasks in cancellable operations
- Properly handles batch processing interruptions
- Cancels CLI subprocess for each file being processed

### Describe Endpoint (`/describe`)
- Wraps vision description tasks in cancellable operations  
- Properly handles batch processing interruptions
- Cancels CLI subprocess for each file being processed

### Generate Embeddings Endpoint (`/generate-embeddings`)
- Wraps embedding generation in cancellable tasks
- Handles interruptions during batch embedding processing
- Cancels CLI subprocess when client disconnects

### Low-Level Methods

The subprocess tracking and cancellation mechanism is also applied to:

- `LlamaCLIBackend.generate()` - Text generation
- `LlamaCLIBackend.embed()` - Embedding generation  
- `LlamaCLIBackend.generate_vision()` - Vision model generation

All these methods now properly track and can cancel their subprocesses.
