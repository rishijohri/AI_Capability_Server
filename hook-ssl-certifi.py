"""
PyInstaller runtime hook for SSL certificate configuration.

This hook ensures that Python uses the bundled certifi certificates
instead of trying to access system paths (which would fail in sandbox).

This runs BEFORE any user code, ensuring SSL is configured correctly
for all libraries (aiohttp, requests, huggingface-hub, etc.)

Note: This hook runs before logging is initialized, so it uses stderr
for output which is captured by macOS Console.
"""

import os
import sys
from pathlib import Path

def setup_ssl_certificates():
    """Configure SSL to use bundled certifi certificates."""
    
    # Determine if running as PyInstaller bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS)
        
        # Look for bundled certifi certificate
        cert_paths = [
            bundle_dir / 'certifi' / 'cacert.pem',
            bundle_dir / 'certifi.pem',
            bundle_dir / 'cacert.pem',
        ]
        
        for cert_path in cert_paths:
            if cert_path.exists():
                # Set environment variables for SSL certificate location
                # These are used by requests, aiohttp, urllib3, and other HTTPS libraries
                os.environ['SSL_CERT_FILE'] = str(cert_path)
                os.environ['REQUESTS_CA_BUNDLE'] = str(cert_path)
                os.environ['CURL_CA_BUNDLE'] = str(cert_path)
                
                # Log to stderr (visible in macOS Console)
                print(f"[HOOK-SSL] Using bundled certificates: {cert_path}", file=sys.stderr)
                return
        
        # If no bundled cert found, warn but allow system to try
        print("[HOOK-SSL] Warning: No bundled certifi found, will attempt system certificates", file=sys.stderr)
    else:
        # Running in development mode - use system certifi
        print("[HOOK-SSL] Development mode: using system certifi", file=sys.stderr)

# Execute SSL setup immediately when this hook runs
setup_ssl_certificates()
