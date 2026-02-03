"""Permission error handler for automatic file access permission management."""

from pathlib import Path
from typing import Callable, Any, Optional
from functools import wraps
import errno


def with_permission_fallback(
    operation_name: str = "access files",
    require_metadata: bool = False,
    prompt_on_error: bool = True
):
    """
    Decorator that automatically handles permission errors by prompting for folder access.
    
    Args:
        operation_name: Description of the operation (shown in dialog)
        require_metadata: If True, selected folder must contain storage_metadata.json
        prompt_on_error: If True, prompt user on permission error; if False, re-raise
        
    Usage:
        @with_permission_fallback("model directory")
        def load_model(path):
            return open(path).read()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (PermissionError, OSError) as e:
                # Check if it's a permission error
                is_perm_error = (
                    isinstance(e, PermissionError) or
                    (isinstance(e, OSError) and e.errno in [errno.EPERM, errno.EACCES])
                )
                
                if not is_perm_error or not prompt_on_error:
                    raise
                
                # Try to handle with file picker
                try:
                    from app.utils.file_picker import handle_permission_error_with_picker
                    
                    selected_dir = handle_permission_error_with_picker(
                        e,
                        operation_name=operation_name,
                        require_metadata=require_metadata
                    )
                    
                    # If the function takes a path argument, update it
                    # This is a simple heuristic - might need refinement for complex cases
                    if args and isinstance(args[0], (str, Path)):
                        # Replace the first path argument with a path in the selected directory
                        original_path = Path(args[0])
                        new_path = Path(selected_dir) / original_path.name
                        args = (str(new_path),) + args[1:]
                    
                    # Retry the operation
                    return func(*args, **kwargs)
                    
                except ImportError:
                    # File picker not available
                    raise e
                except Exception as picker_error:
                    # Permission prompt failed
                    raise Exception(
                        f"Permission denied and folder selection failed: {str(picker_error)}"
                    ) from e
        
        return wrapper
    return decorator


def safe_file_operation(
    file_path: str,
    operation: Callable[[Path], Any],
    operation_name: str = "access this file",
    prompt_on_error: bool = True
) -> Any:
    """
    Perform a file operation with automatic permission error handling.
    
    Args:
        file_path: Path to the file/directory
        operation: Function that takes a Path and performs the operation
        operation_name: Description of the operation
        prompt_on_error: If True, prompt user on permission error
        
    Returns:
        Result of the operation
        
    Example:
        result = safe_file_operation(
            "/path/to/file.json",
            lambda p: json.load(p.open()),
            "load configuration"
        )
    """
    path = Path(file_path)
    
    try:
        return operation(path)
    except (PermissionError, OSError) as e:
        # Check if it's a permission error
        is_perm_error = (
            isinstance(e, PermissionError) or
            (isinstance(e, OSError) and e.errno in [errno.EPERM, errno.EACCES])
        )
        
        if not is_perm_error or not prompt_on_error:
            raise
        
        # Try to handle with file picker
        try:
            from app.utils.file_picker import handle_permission_error_with_picker
            
            selected_dir = handle_permission_error_with_picker(
                e,
                operation_name=operation_name,
                require_metadata=False
            )
            
            # Update path to be within the selected directory
            if path.is_file() or not path.exists():
                new_path = Path(selected_dir) / path.name
            else:
                new_path = Path(selected_dir)
            
            # Retry the operation
            return operation(new_path)
            
        except ImportError:
            # File picker not available
            raise e
        except Exception as picker_error:
            # Permission prompt failed
            raise Exception(
                f"Permission denied and folder selection failed: {str(picker_error)}"
            ) from e
