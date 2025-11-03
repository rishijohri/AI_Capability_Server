#!/usr/bin/env python3
"""
Startup script for AI Server.
Can be run directly or built with PyInstaller.
"""

import sys
from pathlib import Path

# Add app directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run main
from app.main import main

if __name__ == "__main__":
    main()
