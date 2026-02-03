#!/usr/bin/env python3
"""
Startup script for AI Server.
Can be run directly or built with PyInstaller.
"""

import sys
import os
from pathlib import Path

# ============================================================
# EARLY INITIALIZATION (Before any other imports)
# ============================================================

# 1. Configure Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 2. Initialize logging FIRST (to catch all subsequent operations)
from app.utils.logging_config import initialize_logging, get_logger
initialize_logging(debug=os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'))

logger = get_logger(__name__)
logger.info("Starting visarc_ai_server initialization...")

# 3. SSL Certificate Configuration (Critical for sandboxed apps)
def configure_ssl_certificates():
    """Ensure bundled SSL certificates are used instead of system paths."""
    
    # Check if running as PyInstaller bundle
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundle_dir = Path(sys._MEIPASS)
        
        # Try to find bundled certifi certificate
        cert_paths = [
            bundle_dir / 'certifi' / 'cacert.pem',
            bundle_dir / 'certifi.pem',
            bundle_dir / 'cacert.pem',
        ]
        
        for cert_path in cert_paths:
            if cert_path.exists():
                # Set SSL environment variables to use bundled certificates
                os.environ['SSL_CERT_FILE'] = str(cert_path)
                os.environ['REQUESTS_CA_BUNDLE'] = str(cert_path)
                os.environ['CURL_CA_BUNDLE'] = str(cert_path)
                logger.info(f"SSL: Using bundled certificates: {cert_path}")
                return True
        
        logger.warning("SSL: No bundled certifi found, will attempt system certificates")
        return False
    else:
        logger.info("SSL: Development mode - using system certifi")
        return True

# Configure SSL
try:
    configure_ssl_certificates()
except Exception as e:
    logger.error(f"SSL configuration failed: {e}", exc_info=True)

# 4. Import and run main
logger.info("Loading application modules...")
try:
    from app.main import main
    logger.info("Application modules loaded successfully")
except Exception as e:
    logger.critical(f"Failed to import application modules: {e}", exc_info=True)
    sys.exit(1)

if __name__ == "__main__":
    try:
        logger.info("Starting main application...")
        main()
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        sys.exit(1)
