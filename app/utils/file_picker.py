"""Permission error utilities for file access.

Note: This module previously contained GUI directory picker functionality.
GUI code has been removed since this server runs as a background helper
process that cannot make GUI calls. The main app handles user interaction
and passes security-scoped bookmarks for sandbox file access.
"""

import errno
import re
from typing import Optional


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

