"""Resource path utilities for PyInstaller compatibility."""

import sys
import os
import platform
from pathlib import Path


def get_base_path() -> Path:
    """Get the base path for resources.
    
    Returns:
        Path: Base path for resources. When frozen (PyInstaller), returns
              the temporary directory where resources are extracted.
              When running from source, returns the project root directory.
    """
    if getattr(sys, 'frozen', False):
        # Running from PyInstaller bundle
        # sys._MEIPASS is the temp directory where PyInstaller extracts files
        base_path = Path(sys._MEIPASS)
    else:
        # Running from source
        # Go up from app/utils to project root
        base_path = Path(__file__).parent.parent.parent
    
    return base_path


def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to a resource file.
    
    Args:
        relative_path: Path relative to project root (e.g., "model/model.gguf")
    
    Returns:
        Path: Absolute path to the resource
    
    Examples:
        >>> get_resource_path("binary/llama-server")
        PosixPath('/path/to/binary/llama-server')
        
        >>> get_resource_path("model/Qwen3-8B-Q4_K_M.gguf")
        PosixPath('/path/to/model/Qwen3-8B-Q4_K_M.gguf')
    """
    base_path = get_base_path()
    return base_path / relative_path


def get_binary_path(binary_name: str) -> Path:
    """Get absolute path to a binary file.
    
    Args:
        binary_name: Name of the binary (e.g., "llama-server", "llama-cli")
    
    Returns:
        Path: Absolute path to the binary
    
    Note:
        On Windows, automatically appends .exe extension if not already present.
        On macOS/Linux, uses the binary name as-is.
    """
    # Append .exe extension on Windows if not already present
    if platform.system() == "Windows" and not binary_name.endswith(".exe"):
        binary_name = f"{binary_name}.exe"
    
    return get_resource_path(f"binary/{binary_name}")


def get_model_path(model_name: str) -> Path:
    """Get absolute path to a model file.
    
    Args:
        model_name: Name of the model file (e.g., "Qwen3-8B-Q4_K_M.gguf")
    
    Returns:
        Path: Absolute path to the model file
    """
    return get_resource_path(f"model/{model_name}")


def get_data_directory() -> Path:
    """Get the data directory for storing user data.
    
    Returns:
        Path: Path to data directory. When frozen, uses user's home directory.
              When running from source, uses project root.
    """
    if getattr(sys, 'frozen', False):
        # When frozen, store data in user's home directory
        data_dir = Path.home() / ".ai_capability" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    else:
        # When running from source, use project root
        return get_base_path()


def get_face_models_path() -> Path:
    """Get absolute path to the face models directory.
    
    Returns:
        Path: Absolute path to the face models directory (model/)
    """
    return get_resource_path("model")


def ensure_resource_exists(resource_path: Path, resource_type: str = "file") -> bool:
    """Check if a resource exists and log appropriate message.
    
    Args:
        resource_path: Path to the resource
        resource_type: Type of resource ("file", "directory", "binary", "model")
    
    Returns:
        bool: True if resource exists, False otherwise
    """
    exists = resource_path.exists()
    
    if not exists:
        print(f"WARNING: {resource_type} not found: {resource_path}")
        if getattr(sys, 'frozen', False):
            print(f"  Running from PyInstaller bundle. Expected in: {get_base_path()}")
            print(f"  Make sure to include {resource_type} in PyInstaller spec file")
    
    return exists
