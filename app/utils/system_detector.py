"""System detection utilities for automatic binary selection."""

from __future__ import annotations

import platform
import subprocess
from typing import Optional, Dict, List
from pathlib import Path

# Windows NTSTATUS code returned as a process exit code when the OS cannot
# load a required DLL (e.g. vulkan-1.dll, CUDA/HIP/SYCL runtime DLLs).
# Python's subprocess.returncode exposes the raw DWORD on Windows, so this
# value matches what proc.returncode will contain.
_WIN_STATUS_DLL_NOT_FOUND: int = 0xC0000135  # 3221225781 unsigned


def _win_startupinfo() -> subprocess.STARTUPINFO:
    """Return a STARTUPINFO that hides the child process window on Windows."""
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return si


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
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    startupinfo=_win_startupinfo()
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
    def validate_binary_executable(binary_path: Path) -> bool:
        """Test whether a binary can actually be loaded and executed.

        On Windows, GPU-accelerated llama.cpp builds link against external
        runtime DLLs (e.g. ``vulkan-1.dll``, CUDA/HIP/SYCL runtimes).  When
        those DLLs are absent the OS refuses to load the binary and the process
        exits immediately with NTSTATUS ``0xC0000135`` (STATUS_DLL_NOT_FOUND),
        which Python exposes as ``proc.returncode == 3221225781``.

        This method runs the binary with ``--version`` (a flag supported by all
        llama.cpp tools) and returns ``False`` if the binary fails to load.
        Any other outcome — including a non-zero exit code from argument
        handling — is treated as a successful load.

        Args:
            binary_path: Absolute path to the binary to test.

        Returns:
            ``True`` if the binary can be loaded; ``False`` if it cannot
            (missing DLLs or binary does not exist).
        """
        if not binary_path.exists():
            return False

        try:
            kwargs: Dict[str, object] = {}
            if platform.system() == "Windows":
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = subprocess.SW_HIDE
                kwargs = {
                    "creationflags": subprocess.CREATE_NO_WINDOW,
                    "startupinfo": si,
                }

            result = subprocess.run(
                [str(binary_path), "--version"],
                capture_output=True,
                timeout=15,
                **kwargs,
            )

            # On Windows the raw DWORD exit code is exposed directly.
            # Normalise to unsigned 32-bit so we can compare against the
            # NTSTATUS constants regardless of how Python represents the value.
            exit_code_u32 = result.returncode & 0xFFFFFFFF
            if exit_code_u32 == _WIN_STATUS_DLL_NOT_FOUND:
                return False

            # Any other exit code means the binary loaded successfully.
            return True

        except subprocess.TimeoutExpired:
            # The binary started and kept running past the timeout — that means
            # it successfully loaded all its DLLs.  Treat as a successful load.
            return True
        except (OSError, FileNotFoundError):
            # The file is missing or not executable.  Already guarded by the
            # exists() check above, but handle defensively.
            return False

    @staticmethod
    def auto_detect_config(binary_dir: Path) -> str:
        """
        Automatically detect and select the best binary configuration.

        On Windows the GPU-accelerated builds are preferred when a compatible
        GPU is detected, but only if the binary can actually be loaded (i.e.
        the required runtime DLLs are present).  When the GPU binary cannot
        load, the method falls back to the CPU-only build rather than raising
        an error at startup.
        
        Args:
            binary_dir: Path to binary directory
            
        Returns:
            Selected configuration folder name (e.g., 'llama-mac-arm64')
        """
        os_name = SystemDetector.get_os()
        arch = SystemDetector.get_architecture()
        gpu = SystemDetector.detect_gpu()

        llama_binaries_dir = binary_dir / "llama_binaries"
        # Extension used by binaries in the selected config folder.
        bin_ext = ".exe" if os_name == "win" else ""

        def _binary_ok(config_name: str) -> bool:
            """Return True if the llama-server binary in *config_name* loads."""
            test_bin = llama_binaries_dir / config_name / f"llama-server{bin_ext}"
            return SystemDetector.validate_binary_executable(test_bin)

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
        
        # Check if preferred configuration exists *and* the binary loads.
        preferred_path = llama_binaries_dir / preferred
        if preferred_path.exists():
            if _binary_ok(preferred):
                return preferred
            # Preferred config exists but binary fails to load (missing DLLs).
            # Fall through to find a working alternative.
        
        # Fallback: try to find any compatible configuration
        available = SystemDetector.get_available_configs(binary_dir)
        
        if not available:
            # No configurations available, return preferred anyway
            return preferred

        # On Windows, prioritise the CPU-only build as a safe fallback because
        # it has no external runtime DLL dependencies.
        if os_name == "win":
            cpu_configs = [
                c for c in available
                if f"-{os_name}-cpu-" in c and c.endswith(f"-{arch}")
            ]
            for config in cpu_configs:
                if _binary_ok(config):
                    return config

        # Try to find a working configuration matching OS and architecture
        for config in available:
            if f"-{os_name}-" in config and f"-{arch}" in config:
                if _binary_ok(config):
                    return config
        
        # Try to find any working configuration matching OS
        for config in available:
            if f"-{os_name}-" in config:
                if _binary_ok(config):
                    return config
        
        # Last resort: return the first available configuration (even if we
        # could not validate it — the caller will surface a meaningful error).
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
