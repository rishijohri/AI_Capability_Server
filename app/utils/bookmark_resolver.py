"""macOS security-scoped bookmark resolver.

This module handles resolving security-scoped bookmarks that are passed
from the main macOS app to the helper process. This allows the sandboxed
helper process to gain access to user-selected folders.

Usage:
    from app.utils.bookmark_resolver import resolve_security_scoped_bookmark
    
    path, error = resolve_security_scoped_bookmark(bookmark_base64)
    if error:
        print(f"Failed to resolve bookmark: {error}")
    else:
        # Access files at `path`
"""

import sys
import base64
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def resolve_security_scoped_bookmark(bookmark_base64: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a base64-encoded bookmark to a file path.
    
    Supports both:
    - Minimal bookmarks (created with .minimalBookmark) - shared from main app
    - Security-scoped bookmarks (created with .withSecurityScope) - local only
    
    On macOS, this uses the Foundation framework via PyObjC to resolve
    the bookmark and start accessing the security-scoped resource.
    
    Args:
        bookmark_base64: Base64-encoded bookmark data from the main app
        
    Returns:
        Tuple of (resolved_path, error_message)
        - On success: (path, None)
        - On failure: (None, error_message)
    """
    if sys.platform != "darwin":
        return None, "Security-scoped bookmarks are only supported on macOS"
    
    try:
        # Decode the base64 bookmark data
        bookmark_data = base64.b64decode(bookmark_base64)
        logger.info(f"Decoded bookmark data: {len(bookmark_data)} bytes")
    except Exception as e:
        return None, f"Failed to decode base64 bookmark: {str(e)}"
    
    try:
        # Import PyObjC Foundation framework
        from Foundation import NSURL, NSData
        
        # Convert bytes to NSData
        ns_data = NSData.dataWithBytes_length_(bookmark_data, len(bookmark_data))
        
        # Bookmark resolution options
        # For MINIMAL bookmarks (shared from main app): use NSURLBookmarkResolutionWithoutUI
        # For SECURITY-SCOPED bookmarks: use NSURLBookmarkResolutionWithSecurityScope
        NSURLBookmarkResolutionWithoutUI = 1 << 8  # 256
        NSURLBookmarkResolutionWithSecurityScope = 1 << 10  # 1024
        
        # Try resolving WITHOUT security scope first (for minimal bookmarks)
        # This is the correct approach for bookmarks shared from another app
        resolved_url, is_stale, error = NSURL.URLByResolvingBookmarkData_options_relativeToURL_bookmarkDataIsStale_error_(
            ns_data,
            NSURLBookmarkResolutionWithoutUI,  # Use WithoutUI for minimal bookmarks
            None,  # relativeToURL
            None,  # bookmarkDataIsStale (output parameter)
            None   # error (output parameter)
        )
        
        if error:
            logger.warning(f"Failed to resolve as minimal bookmark: {error.localizedDescription()}")
            # Fallback: try with security scope option (for security-scoped bookmarks)
            resolved_url, is_stale, error = NSURL.URLByResolvingBookmarkData_options_relativeToURL_bookmarkDataIsStale_error_(
                ns_data,
                NSURLBookmarkResolutionWithSecurityScope,
                None,
                None,
                None
            )
            if error:
                return None, f"Failed to resolve bookmark: {error.localizedDescription()}"
        
        if resolved_url is None:
            return None, "Failed to resolve bookmark: URL is None"
        
        # Start accessing the security-scoped resource
        # This is needed even for minimal bookmarks to gain file access
        access_granted = resolved_url.startAccessingSecurityScopedResource()
        if access_granted:
            logger.info("startAccessingSecurityScopedResource returned True - access granted")
        else:
            # For minimal bookmarks, this may return False but access still works
            # because the main app has the resource open
            logger.warning("startAccessingSecurityScopedResource returned False - may need main app to be running")
        
        # Get the file path from the URL
        file_path = resolved_url.path()
        
        if is_stale:
            logger.warning(f"Bookmark for {file_path} is stale - main app should regenerate it")
        
        logger.info(f"Successfully resolved bookmark to: {file_path}")
        return file_path, None
        
    except ImportError:
        return None, "PyObjC Foundation framework not available. Install with: pip install pyobjc-framework-Cocoa"
    except Exception as e:
        return None, f"Failed to resolve bookmark: {str(e)}"


def stop_accessing_security_scoped_resource(url_path: str) -> None:
    """
    Stop accessing a security-scoped resource.
    
    Call this when you're done accessing the resource to release the
    security scope. Note: In practice, for a server process that needs
    ongoing access, you may keep the resource open for the lifetime
    of the process.
    
    Args:
        url_path: The file path that was previously resolved from a bookmark
    """
    if sys.platform != "darwin":
        return
    
    try:
        from Foundation import NSURL
        
        url = NSURL.fileURLWithPath_(url_path)
        url.stopAccessingSecurityScopedResource()
        logger.info(f"Stopped accessing security-scoped resource: {url_path}")
        
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Failed to stop accessing security-scoped resource: {str(e)}")
