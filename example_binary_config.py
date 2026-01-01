#!/usr/bin/env python3
"""
Example: How to use Binary Configuration System

This example demonstrates:
1. Checking current binary configuration
2. Viewing available configurations
3. Manually changing configuration
4. Verifying binary paths
"""

import requests
import json

# API base URL
BASE_URL = "http://127.0.0.1:8000/api"


def get_config():
    """Get current configuration."""
    response = requests.get(f"{BASE_URL}/config")
    return response.json()


def update_config(binary_config):
    """Update binary configuration."""
    response = requests.post(
        f"{BASE_URL}/config",
        json={"binary_config": binary_config}
    )
    return response.json()


def main():
    """Main example."""
    print("=" * 70)
    print("Binary Configuration Example")
    print("=" * 70)
    
    # 1. Get current configuration
    print("\n1. Current Configuration:")
    print("-" * 70)
    config = get_config()
    
    print(f"Binary Config:    {config['binary_config']}")
    print(f"\nSystem Information:")
    for key, value in config['system_info'].items():
        print(f"  {key:15}: {value}")
    
    print(f"\nAvailable Configurations ({len(config['available_binary_configs'])}):")
    for cfg in config['available_binary_configs']:
        marker = "⭐ (current)" if cfg == config['binary_config'] else "   "
        print(f"  {marker} {cfg}")
    
    # 2. Show what would happen if we switch (example only)
    print("\n2. Example: Switching Configuration")
    print("-" * 70)
    
    available = config['available_binary_configs']
    current = config['binary_config']
    
    # Find an alternative config
    alternatives = [c for c in available if c != current]
    
    if alternatives:
        alt_config = alternatives[0]
        print(f"Current:  {current}")
        print(f"Switch to: {alt_config}")
        print("\nTo switch, run:")
        print(f'  curl -X POST {BASE_URL}/config \\')
        print(f'    -H "Content-Type: application/json" \\')
        print(f'    -d \'{{"binary_config": "{alt_config}"}}\'')
        
        # Uncomment to actually switch:
        # print("\nSwitching now...")
        # result = update_config(alt_config)
        # print(f"New config: {result['binary_config']}")
        
        # Switch back:
        # update_config(current)
        # print(f"Switched back to: {current}")
    else:
        print("Only one configuration available. No alternatives to switch to.")
    
    # 3. Configuration recommendations
    print("\n3. Configuration Recommendations:")
    print("-" * 70)
    
    os_type = config['system_info'].get('os', 'unknown')
    arch = config['system_info'].get('architecture', 'unknown')
    gpu = config['system_info'].get('gpu', 'cpu')
    
    print(f"Your System: {os_type} / {arch} / {gpu}")
    print("\nRecommended configurations:")
    
    if os_type == "mac":
        if arch == "arm64":
            print("  ✓ llama-mac-arm64 (Best for Apple Silicon)")
            print("  - llama-mac-x64 (For Intel Macs)")
        else:
            print("  ✓ llama-mac-x64 (Best for Intel Macs)")
            print("  - llama-mac-arm64 (For Apple Silicon)")
    elif os_type == "win":
        if gpu == "vulkan":
            print("  ✓ llama-win-vulkan-x64 (Best for NVIDIA GPUs)")
        elif gpu == "hip-radeon":
            print("  ✓ llama-win-hip-radeon-x64 (Best for AMD GPUs)")
        elif gpu == "sycl":
            print("  ✓ llama-win-sycl-x64 (Best for Intel Arc GPUs)")
        else:
            print("  ✓ llama-win-cpu-x64 (CPU-only mode)")
    
    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server at http://127.0.0.1:8000")
        print("Please make sure the server is running:")
        print("  python3 run_server.py")
    except Exception as e:
        print(f"ERROR: {e}")
