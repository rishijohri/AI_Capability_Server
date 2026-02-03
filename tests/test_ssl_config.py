#!/usr/bin/env python3
"""
Test SSL certificate configuration in bundled app.

This script verifies that:
1. Bundled certifi certificates are found
2. Environment variables are set correctly
3. HTTPS requests work without accessing system paths
"""

import os
import sys
from pathlib import Path

def test_ssl_config():
    """Test SSL certificate configuration."""
    
    print("=" * 60)
    print("SSL Certificate Configuration Test")
    print("=" * 60)
    
    # Check if running as bundle
    is_bundled = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    print(f"\nRunning as PyInstaller bundle: {is_bundled}")
    
    if is_bundled:
        bundle_dir = Path(sys._MEIPASS)
        print(f"Bundle directory: {bundle_dir}")
        
        # Check for bundled certifi
        cert_paths = [
            bundle_dir / 'certifi' / 'cacert.pem',
            bundle_dir / 'certifi.pem',
            bundle_dir / 'cacert.pem',
        ]
        
        print("\nSearching for bundled certificates:")
        for cert_path in cert_paths:
            exists = "✓ FOUND" if cert_path.exists() else "✗ NOT FOUND"
            print(f"  {exists}: {cert_path}")
    
    # Check environment variables
    print("\nSSL Environment Variables:")
    ssl_vars = ['SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE']
    for var in ssl_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var}: {value}")
    
    # Try importing certifi
    print("\nCertifi Module:")
    try:
        import certifi
        cert_path = certifi.where()
        exists = Path(cert_path).exists()
        print(f"  certifi.where(): {cert_path}")
        print(f"  File exists: {exists}")
    except ImportError:
        print("  ✗ certifi module not found")
    
    # Test HTTPS request
    print("\nTesting HTTPS Request:")
    try:
        import urllib.request
        import ssl
        
        # Create SSL context (will use environment variables)
        context = ssl.create_default_context()
        
        # Try a simple HTTPS request
        url = "https://www.google.com"
        print(f"  Attempting: {url}")
        
        response = urllib.request.urlopen(url, timeout=5, context=context)
        print(f"  ✓ SUCCESS: Status {response.status}")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_ssl_config()
