"""
Enhanced logging configuration for macOS Console app visibility.

This module provides comprehensive logging for the AI Server that:
- Outputs to stdout/stderr (captured by macOS Console)
- Includes timestamps, process info, and severity levels
- Catches uncaught exceptions and crashes
- Provides diagnostic information for debugging
"""

import logging
import sys
import os
import traceback
import signal
import atexit
from pathlib import Path
from datetime import datetime
from typing import Optional


class ConsoleFormatter(logging.Formatter):
    """Custom formatter optimized for macOS Console app."""
    
    # ANSI color codes (visible in terminal, stripped in Console)
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        """Format log record for Console app."""
        # Add color for terminal output
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            levelname = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.COLORS['RESET']}"
        else:
            levelname = record.levelname
        
        # Format: [TIMESTAMP] [LEVEL] [MODULE] Message
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Build log message
        parts = [
            f"[{timestamp}]",
            f"[{levelname}]",
            f"[{record.name}]",
            record.getMessage()
        ]
        
        message = " ".join(parts)
        
        # Add exception info if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        
        return message


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configure comprehensive logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to also log to a file
        log_file: Path to log file (if log_to_file=True)
    
    Returns:
        Root logger instance
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler (stdout for INFO+, stderr for WARNING+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(console_handler)
    
    # Create error handler (stderr for ERROR+)
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(ConsoleFormatter())
    root_logger.addHandler(error_handler)
    
    # Optional file handler
    if log_to_file and log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Capture everything in file
            file_handler.setFormatter(ConsoleFormatter())
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            root_logger.warning(f"Failed to setup file logging: {e}")
    
    return root_logger


def log_system_info():
    """Log detailed system information for diagnostics."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("VISARC AI SERVER STARTING")
    logger.info("=" * 60)
    
    # Python environment
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Platform: {sys.platform}")
    
    # Process info
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Parent Process ID: {os.getppid()}")
    
    # PyInstaller bundle info
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        logger.info(f"Running as PyInstaller bundle")
        logger.info(f"Bundle directory: {sys._MEIPASS}")
        logger.info(f"Executable: {sys.executable}")
    else:
        logger.info(f"Running in development mode")
        logger.info(f"Working directory: {os.getcwd()}")
    
    # Environment variables (sanitized)
    logger.info("Environment variables:")
    for key in ['SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'PATH']:
        value = os.environ.get(key, 'NOT SET')
        logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


def exception_handler(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for uncaught exceptions.
    Logs all unhandled exceptions before the program crashes.
    """
    logger = logging.getLogger(__name__)
    
    # Ignore KeyboardInterrupt (Ctrl+C)
    if issubclass(exc_type, KeyboardInterrupt):
        logger.info("Received keyboard interrupt, shutting down gracefully")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Log the exception
    logger.critical("=" * 60)
    logger.critical("UNCAUGHT EXCEPTION - APPLICATION CRASH")
    logger.critical("=" * 60)
    logger.critical(f"Exception type: {exc_type.__name__}")
    logger.critical(f"Exception message: {exc_value}")
    logger.critical("Traceback:")
    
    # Format and log full traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in tb_lines:
        for subline in line.rstrip().split('\n'):
            logger.critical(f"  {subline}")
    
    logger.critical("=" * 60)
    logger.critical("Application will now terminate")
    logger.critical("=" * 60)
    
    # Call default handler to actually exit
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def signal_handler(signum, frame):
    """Handle termination signals."""
    logger = logging.getLogger(__name__)
    
    signal_names = {
        signal.SIGTERM: 'SIGTERM',
        signal.SIGINT: 'SIGINT',
        signal.SIGHUP: 'SIGHUP' if hasattr(signal, 'SIGHUP') else 'UNKNOWN'
    }
    
    signal_name = signal_names.get(signum, f'SIGNAL_{signum}')
    
    logger.warning("=" * 60)
    logger.warning(f"Received signal: {signal_name}")
    logger.warning("Initiating graceful shutdown...")
    logger.warning("=" * 60)
    
    # Exit gracefully
    sys.exit(0)


def setup_crash_handlers():
    """Setup handlers for crashes and signals."""
    # Install global exception handler
    sys.excepthook = exception_handler
    
    # Install signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)
    
    # Register exit handler
    atexit.register(on_exit)
    
    logger = logging.getLogger(__name__)
    logger.debug("Crash handlers installed successfully")


def on_exit():
    """Called when the application exits."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("VISARC AI SERVER SHUTDOWN COMPLETE")
    logger.info(f"Uptime: {datetime.now()}")
    logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize logging on module import
_initialized = False

def initialize_logging(debug: bool = False):
    """
    Initialize the logging system.
    
    Args:
        debug: Enable debug logging
    """
    global _initialized
    
    if _initialized:
        return
    
    try:
        # Determine log level
        level = logging.DEBUG if debug else logging.INFO
        
        # Setup logging
        setup_logging(level=level)
        
        # Log system info
        log_system_info()
        
        # Setup crash handlers
        setup_crash_handlers()
        
        _initialized = True
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging system initialized (level: {logging.getLevelName(level)})")
        
    except Exception as e:
        # If logging setup fails, write to stderr
        import sys
        import traceback
        sys.stderr.write(f"CRITICAL: Logging initialization failed: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        raise
