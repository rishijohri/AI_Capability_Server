"""Permission error handler utilities.

Note: GUI-based fallback handling has been removed since this server runs
as a background helper process that cannot make GUI calls. The main app
handles user interaction and passes security-scoped bookmarks for
sandbox file access.
"""

from typing import Callable, Any, TypeVar
from functools import wraps
import errno

T = TypeVar('T')


def is_permission_error(error: Exception) -> bool:
    """
    Check if an error is a permission denied error.
    
    Args:
        error: Exception to check
        
    Returns:
        bool: True if it's a permission error
    """
    is_perm_error = (
        isinstance(error, PermissionError) or
        (isinstance(error, OSError) and error.errno in [errno.EPERM, errno.EACCES])
    )
    return is_perm_error


def with_permission_check(operation_name: str = "access files"):
    """
    Decorator that provides clear error messages for permission errors.
    
    Since this is a helper process without GUI, permission errors are
    passed back clearly to the caller (main app) which should provide
    proper access via security-scoped bookmarks.
    
    Args:
        operation_name: Description of the operation for error messages
        
    Usage:
        @with_permission_check("model directory")
        def load_model(path):
            return open(path).read()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except (PermissionError, OSError) as e:
                if is_permission_error(e):
                    raise PermissionError(
                        f"Permission denied while trying to {operation_name}. "
                        f"The main app should provide a security-scoped bookmark "
                        f"for the required path. Original error: {str(e)}"
                    ) from e
                raise
        return wrapper
    return decorator

