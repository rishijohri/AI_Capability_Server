#!/usr/bin/env python3
"""
Test script to verify logging configuration works correctly.

Run this to ensure logs appear in macOS Console:
1. Terminal: python tests/test_logging.py
2. Console.app: Filter by "process:python" or "test_logging"
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Initialize logging
from app.utils.logging_config import initialize_logging, get_logger

print("Initializing logging system...")
initialize_logging(debug=True)

logger = get_logger(__name__)

def test_log_levels():
    """Test all log levels."""
    logger.info("=" * 60)
    logger.info("LOGGING SYSTEM TEST")
    logger.info("=" * 60)
    
    logger.debug("This is a DEBUG message - detailed troubleshooting info")
    logger.info("This is an INFO message - normal operation")
    logger.warning("This is a WARNING message - something unexpected")
    logger.error("This is an ERROR message - recoverable error")
    
    logger.info("All log levels tested successfully")


def test_exception_logging():
    """Test exception logging."""
    logger.info("Testing exception logging...")
    
    try:
        raise ValueError("This is a test exception - it's intentional!")
    except ValueError as e:
        logger.error(f"Caught exception: {e}", exc_info=True)
    
    logger.info("Exception logging test complete")


def test_crash_handler():
    """Test crash handler (commented out by default)."""
    logger.info("To test crash handler, uncomment the raise statement below")
    logger.info("Then check Console.app for UNCAUGHT EXCEPTION logs")
    
    # Uncomment to test crash handler:
    # raise RuntimeError("Test crash - this should be logged as CRITICAL")


def test_console_visibility():
    """Test that logs are visible in Console.app."""
    logger.info("=" * 60)
    logger.info("CONSOLE VISIBILITY TEST")
    logger.info("=" * 60)
    logger.info("Open Console.app and filter by:")
    logger.info("  process:python")
    logger.info("or")
    logger.info("  process:test_logging")
    logger.info("")
    logger.info("You should see this message with timestamp and log level")
    logger.info("")
    logger.info("Waiting 3 seconds...")
    
    time.sleep(1)
    logger.info("1 second elapsed")
    time.sleep(1)
    logger.info("2 seconds elapsed")
    time.sleep(1)
    logger.info("3 seconds elapsed")
    
    logger.info("=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LOGGING SYSTEM TEST")
    print("=" * 60)
    print("\n1. Open Console.app (Applications → Utilities → Console)")
    print("2. In the search bar, enter: process:python")
    print("3. You should see logs appear below")
    print("\n" + "=" * 60 + "\n")
    
    input("Press Enter to start the test...")
    
    test_log_levels()
    print()
    
    test_exception_logging()
    print()
    
    test_console_visibility()
    print()
    
    test_crash_handler()
    
    print("\n" + "=" * 60)
    print("Check Console.app to verify all logs appeared!")
    print("=" * 60 + "\n")
