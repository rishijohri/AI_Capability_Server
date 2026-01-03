"""Test script to verify system detection and binary configuration selection."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.utils.system_detector import SystemDetector
from app.config import initialize_config, get_config


def test_system_detection():
    """Test system detection functionality."""
    print("=" * 60)
    print("System Detection Test")
    print("=" * 60)
    
    # Test individual detection methods
    print("\n1. System Information:")
    print("-" * 60)
    system_info = SystemDetector.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Test available configurations
    print("\n2. Available Binary Configurations:")
    print("-" * 60)
    binary_dir = project_root / "binary"
    available_configs = SystemDetector.get_available_configs(binary_dir)
    if available_configs:
        for config in available_configs:
            print(f"  - {config}")
    else:
        print("  No configurations found in binary/llama_binaries/")
    
    # Test auto-detection
    print("\n3. Auto-detected Configuration:")
    print("-" * 60)
    detected_config = SystemDetector.auto_detect_config(binary_dir)
    print(f"  Selected: {detected_config}")
    
    # Verify if it exists
    is_valid = SystemDetector.validate_config(detected_config, binary_dir)
    print(f"  Valid: {is_valid}")
    
    # Test configuration initialization
    print("\n4. Initialize Server Configuration:")
    print("-" * 60)
    config = initialize_config()
    print(f"  Binary Config: {config.binary_config}")
    print(f"  System Info: {config.system_info}")
    print(f"  Available Configs: {config.get_available_binary_configs()}")
    
    # Test binary path generation
    print("\n5. Binary Paths:")
    print("-" * 60)
    binary_names = ["llama-server", "llama-cli", "llama-embedding"]
    for binary_name in binary_names:
        binary_path = config.get_binary_path(binary_name)
        exists = binary_path.exists()
        print(f"  {binary_name}: {binary_path}")
        print(f"    Exists: {exists}")
    
    # Test manual configuration change
    print("\n6. Manual Configuration Change:")
    print("-" * 60)
    available = config.get_available_binary_configs()
    if len(available) > 1:
        # Try to switch to a different config
        alternate_config = available[1] if available[0] == config.binary_config else available[0]
        print(f"  Current: {config.binary_config}")
        print(f"  Switching to: {alternate_config}")
        
        success = config.set_binary_config(alternate_config)
        print(f"  Success: {success}")
        print(f"  New config: {config.binary_config}")
        
        # Switch back
        config.set_binary_config(detected_config)
        print(f"  Switched back to: {config.binary_config}")
    else:
        print("  Only one configuration available, skipping switch test")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_system_detection()
