"""Simple test script to verify system detection (no dependencies)."""

import platform
import subprocess
from pathlib import Path


def test_basic_detection():
    """Test basic system detection."""
    print("=" * 60)
    print("Basic System Detection Test")
    print("=" * 60)
    
    # OS detection
    print("\n1. Operating System:")
    system = platform.system()
    print(f"   Raw: {system}")
    
    if system.lower() == "darwin":
        os_name = "mac"
    elif system.lower() == "windows":
        os_name = "win"
    else:
        os_name = "mac"  # default
    print(f"   Mapped: {os_name}")
    
    # Architecture detection
    print("\n2. CPU Architecture:")
    machine = platform.machine()
    print(f"   Raw: {machine}")
    
    if machine.lower() in ["arm64", "aarch64"]:
        arch = "arm64"
    elif machine.lower() in ["x86_64", "amd64"]:
        arch = "x64"
    else:
        arch = machine.lower()
    print(f"   Mapped: {arch}")
    
    # GPU detection (basic)
    print("\n3. GPU Detection:")
    gpu = None
    if system.lower() == "darwin":
        print("   macOS uses Metal (no GPU suffix needed)")
        gpu = None
    elif system.lower() == "windows":
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            gpu_info = result.stdout.lower()
            print(f"   GPU Info: {gpu_info[:100]}...")
            
            if "amd" in gpu_info or "radeon" in gpu_info:
                gpu = "hip-radeon"
            elif "intel" in gpu_info and "arc" in gpu_info:
                gpu = "sycl"
            elif "nvidia" in gpu_info or "geforce" in gpu_info or "rtx" in gpu_info:
                gpu = "vulkan"
            else:
                gpu = "cpu"
        except Exception as e:
            print(f"   Error detecting GPU: {e}")
            gpu = "cpu"
    else:
        gpu = None
    
    print(f"   Detected GPU type: {gpu or 'N/A'}")
    
    # Recommended config
    print("\n4. Recommended Binary Configuration:")
    if os_name == "mac":
        recommended = f"llama-mac-{arch}"
    else:
        if gpu and gpu != "cpu":
            recommended = f"llama-win-{gpu}-{arch}"
        else:
            recommended = f"llama-win-cpu-{arch}"
    
    print(f"   {recommended}")
    
    # Check if it exists
    print("\n5. Checking Binary Directory:")
    binary_dir = Path(__file__).parent / "binary" / "llama_binaries"
    print(f"   Path: {binary_dir}")
    print(f"   Exists: {binary_dir.exists()}")
    
    if binary_dir.exists():
        configs = [item.name for item in binary_dir.iterdir() if item.is_dir() and item.name.startswith("llama-")]
        print(f"\n   Available configurations ({len(configs)}):")
        for config in sorted(configs):
            exists_marker = "✓" if (binary_dir / config).exists() else "✗"
            is_recommended = "⭐" if config == recommended else "  "
            print(f"     {exists_marker} {is_recommended} {config}")
        
        recommended_path = binary_dir / recommended
        print(f"\n   Recommended config exists: {recommended_path.exists()}")
    else:
        print("   Binary directory not found!")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_detection()
