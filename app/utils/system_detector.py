"""System detection utilities for automatic binary selection."""

import platform
import subprocess
from typing import Optional, Dict, List
from pathlib import Path


class SystemDetector:
    """Detect system properties and select appropriate binary configuration."""
    
    @staticmethod
    def get_os() -> str:
        """Get operating system (mac or win)."""
        system = platform.system().lower()
        if system == "darwin":
            return "mac"
        elif system == "windows":
            return "win"
        else:
            # Default to mac for Linux/other Unix systems
            return "mac"
    
    @staticmethod
    def get_architecture() -> str:
        """Get CPU architecture (arm64, x64, etc)."""
        machine = platform.machine().lower()
        if machine in ["arm64", "aarch64"]:
            return "arm64"
        elif machine in ["x86_64", "amd64"]:
            return "x64"
        else:
            return machine
    
    @staticmethod
    def detect_gpu() -> Optional[str]:
        """
        Detect GPU type if available.
        Returns: 'hip-radeon', 'sycl', 'vulkan', or None for CPU-only
        """
        system = platform.system().lower()
        
        if system == "darwin":
            # macOS uses Metal (no additional GPU detection needed)
            return None
        
        elif system == "windows":
            try:
                # Try to detect AMD GPU (for HIP)
                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                gpu_info = result.stdout.lower()
                
                if "amd" in gpu_info or "radeon" in gpu_info:
                    return "hip-radeon"
                elif "intel" in gpu_info and "arc" in gpu_info:
                    return "sycl"
                elif "nvidia" in gpu_info or "geforce" in gpu_info or "rtx" in gpu_info:
                    # NVIDIA GPUs can use Vulkan
                    return "vulkan"
            except Exception:
                pass
        
        return None
    
    @staticmethod
    def get_available_configs(binary_dir: Path) -> List[str]:
        """
        Get list of available binary configurations in llama_binaries folder.
        
        Args:
            binary_dir: Path to binary directory
            
        Returns:
            List of available configuration folder names
        """
        llama_binaries_dir = binary_dir / "llama_binaries"
        
        if not llama_binaries_dir.exists():
            return []
        
        configs = []
        for item in llama_binaries_dir.iterdir():
            if item.is_dir() and item.name.startswith("llama-"):
                configs.append(item.name)
        
        return sorted(configs)
    
    @staticmethod
    def auto_detect_config(binary_dir: Path) -> str:
        """
        Automatically detect and select the best binary configuration.
        
        Args:
            binary_dir: Path to binary directory
            
        Returns:
            Selected configuration folder name (e.g., 'llama-mac-arm64')
        """
        os_name = SystemDetector.get_os()
        arch = SystemDetector.get_architecture()
        gpu = SystemDetector.detect_gpu()
        
        # Build preferred configuration name
        if os_name == "mac":
            # macOS: llama-mac-arm64 or llama-mac-x64
            preferred = f"llama-mac-{arch}"
        else:
            # Windows: prefer GPU-accelerated versions if available
            if gpu:
                preferred = f"llama-win-{gpu}-{arch}"
            else:
                preferred = f"llama-win-cpu-{arch}"
        
        # Check if preferred configuration exists
        llama_binaries_dir = binary_dir / "llama_binaries"
        preferred_path = llama_binaries_dir / preferred
        
        if preferred_path.exists():
            return preferred
        
        # Fallback: try to find any compatible configuration
        available = SystemDetector.get_available_configs(binary_dir)
        
        if not available:
            # No configurations available, return preferred anyway
            return preferred
        
        # Try to find a configuration matching OS and architecture
        for config in available:
            if f"-{os_name}-" in config and f"-{arch}" in config:
                return config
        
        # Try to find any configuration matching OS
        for config in available:
            if f"-{os_name}-" in config:
                return config
        
        # Last resort: return the first available configuration
        return available[0]
    
    @staticmethod
    def get_system_info() -> Dict[str, str]:
        """
        Get detailed system information.
        
        Returns:
            Dictionary with system details
        """
        return {
            "os": SystemDetector.get_os(),
            "architecture": SystemDetector.get_architecture(),
            "gpu": SystemDetector.detect_gpu() or "cpu",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
    
    @staticmethod
    def validate_config(config_name: str, binary_dir: Path) -> bool:
        """
        Validate that a configuration exists.
        
        Args:
            config_name: Configuration folder name
            binary_dir: Path to binary directory
            
        Returns:
            True if configuration exists, False otherwise
        """
        llama_binaries_dir = binary_dir / "llama_binaries"
        config_path = llama_binaries_dir / config_name
        return config_path.exists() and config_path.is_dir()
