"""Directory picker utility for selecting storage folder."""

import sys
import os
import errno
from pathlib import Path
from typing import Optional, Callable, Any
from functools import wraps


def open_directory_picker(initial_dir: Optional[str] = None, title: str = "Select Folder") -> Optional[str]:
    """
    Open a native directory picker dialog to select a folder.
    
    Args:
        initial_dir: Initial directory to open the picker at. If None, uses home directory.
        title: Title of the picker dialog.
    
    Returns:
        Optional[str]: Selected directory path or None if cancelled
    """
    # Determine initial directory
    if initial_dir:
        init_path = Path(initial_dir)
        # If it's a file, use its parent directory
        if init_path.is_file():
            init_path = init_path.parent
        # If directory doesn't exist, go up until we find one that does
        while not init_path.exists() and init_path != init_path.parent:
            init_path = init_path.parent
        initial_dir = str(init_path)
    else:
        initial_dir = str(Path.home())
    
    try:
        # Try tkinter first (cross-platform)
        import tkinter as tk
        from tkinter import filedialog
        
        # Create root window and hide it
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Open directory dialog
        dir_path = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir,
            mustexist=True
        )
        
        # Cleanup
        root.destroy()
        
        return dir_path if dir_path else None
        
    except ImportError:
        # Fallback to AppleScript on macOS
        if sys.platform == "darwin":
            return _open_directory_picker_applescript(initial_dir, title)
        else:
            raise Exception("No directory picker available. Install tkinter or run on macOS.")


def _open_directory_picker_applescript(initial_dir: Optional[str] = None, title: str = "Select Folder") -> Optional[str]:
    """
    macOS-specific directory picker using AppleScript.
    
    Args:
        initial_dir: Initial directory to open the picker at.
        title: Title of the picker dialog.
    
    Returns:
        Optional[str]: Selected directory path or None if cancelled
    """
    import subprocess
    
    # Determine initial directory
    if initial_dir:
        init_path = Path(initial_dir)
        if init_path.is_file():
            init_path = init_path.parent
        while not init_path.exists() and init_path != init_path.parent:
            init_path = init_path.parent
        default_location = f'(POSIX file "{init_path}")'
    else:
        default_location = '(path to home folder)'
    
    applescript = f'''
    tell application "System Events"
        activate
        set theFolder to choose folder with prompt "{title}:" default location {default_location}
        return POSIX path of theFolder
    end tell
    '''
    
    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            dir_path = result.stdout.strip()
            return dir_path if dir_path else None
        else:
            # User cancelled
            return None
            
    except subprocess.TimeoutExpired:
        raise Exception("Directory picker dialog timed out")
    except Exception as e:
        raise Exception(f"Failed to open directory picker: {str(e)}")


def is_permission_error(error: Exception) -> bool:
    """
    Check if an error is a permission denied error.
    
    Args:
        error: Exception to check
        
    Returns:
        bool: True if it's a permission error
    """
    if isinstance(error, PermissionError):
        return True
    if isinstance(error, OSError) and error.errno == errno.EPERM:
        return True
    if isinstance(error, OSError) and error.errno == errno.EACCES:
        return True
    # Check error message for permission-related strings
    error_msg = str(error).lower()
    if 'operation not permitted' in error_msg or 'permission denied' in error_msg:
        return True
    return False


def extract_path_from_error(error: Exception) -> Optional[str]:
    """
    Extract the file/directory path from a permission error.
    
    Args:
        error: Permission error
        
    Returns:
        Optional[str]: Extracted path or None
    """
    # Try to get filename from OSError
    if isinstance(error, OSError) and hasattr(error, 'filename'):
        return error.filename
    
    # Try to extract from error message
    error_msg = str(error)
    
    # Look for common patterns: '/path/to/file' or "/path/to/file"
    import re
    patterns = [
        r"'([/][^']+)'",  # Single quotes
        r'"([/][^"]+)"',  # Double quotes
        r'([/][^\s:]+)',  # Just path starting with /
    ]
    
    for pattern in patterns:
        match = re.search(pattern, error_msg)
        if match:
            return match.group(1)
    
    return None


def handle_permission_error_with_picker(
    error: Exception,
    operation_name: str = "access this location",
    require_metadata: bool = False
) -> Optional[str]:
    """
    Handle a permission error by opening a directory picker dialog.
    
    Args:
        error: The permission error that occurred
        operation_name: Description of the operation for the dialog title
        require_metadata: If True, the selected folder must contain storage_metadata.json
        
    Returns:
        Optional[str]: Selected directory path or None if cancelled
        
    Raises:
        Exception: If no directory is selected or validation fails
    """
    # Extract the path that caused the error
    error_path = extract_path_from_error(error)
    
    title = f"Grant Permission to {operation_name.title()}"
    
    # Open directory picker at the problematic location
    selected_dir = open_directory_picker(
        initial_dir=error_path,
        title=title
    )
    
    if not selected_dir:
        raise Exception("Permission required: No folder selected")
    
    # Validate if metadata is required
    if require_metadata:
        metadata_file = Path(selected_dir) / "storage_metadata.json"
        if not metadata_file.exists():
            raise Exception(
                f"storage_metadata.json not found in {selected_dir}. "
                f"Please select the correct storage folder."
            )
    
    return selected_dir

