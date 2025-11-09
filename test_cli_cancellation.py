#!/usr/bin/env python3
"""Test script to verify CLI subprocess cancellation on WebSocket disconnect."""

import asyncio
import sys
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.llm_service import LlamaCLIBackend
from app.config import get_config


async def test_cli_cancellation():
    """Test that CLI subprocess can be properly cancelled."""
    print("Testing CLI subprocess cancellation...")
    
    backend = LlamaCLIBackend()
    config = get_config()
    
    # Use a small model for testing
    model_path = config.get_model_path("granite-4.0-350m-UD-Q6_K_XL.gguf")
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Skipping test - model file required")
        return
    
    try:
        # Initialize backend
        print("Starting CLI backend...")
        await backend.start(model_path)
        print("Backend started successfully")
        
        # Start a generation task
        print("\nStarting generation task...")
        messages = [{"role": "user", "content": "Write a very long story about artificial intelligence."}]
        
        generation_task = asyncio.create_task(
            anext(backend.generate(messages, stream=False))
        )
        
        # Wait a moment to ensure subprocess has started
        await asyncio.sleep(1)
        
        print("Generation started, checking if process is running...")
        async with backend._process_lock:
            if backend._current_process is not None:
                print(f"✓ Subprocess tracked with PID: {backend._current_process.pid}")
            else:
                print("✗ No subprocess tracked!")
                return
        
        # Simulate WebSocket disconnect by cancelling the task
        print("\nSimulating WebSocket disconnect...")
        generation_task.cancel()
        
        try:
            await generation_task
        except asyncio.CancelledError:
            print("✓ Generation task cancelled")
        
        # Now cancel the subprocess
        print("Cancelling subprocess...")
        await backend.cancel_generation()
        
        # Verify subprocess was terminated
        async with backend._process_lock:
            if backend._current_process is None:
                print("✓ Subprocess successfully terminated and cleaned up")
            else:
                print("✗ Subprocess still tracked!")
                return
        
        print("\n✓ Test passed: CLI subprocess cancellation works correctly")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        await backend.stop()


if __name__ == "__main__":
    asyncio.run(test_cli_cancellation())
