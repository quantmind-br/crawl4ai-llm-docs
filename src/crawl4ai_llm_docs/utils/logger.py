"""Logging configuration utilities."""
import logging
import sys
from pathlib import Path
from typing import Optional

from platformdirs import PlatformDirs


def setup_logging(
    name: str = "crawl4ai_llm_docs",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        try:
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    return logger


def get_log_file_path(app_name: str = "crawl4ai-llm-docs") -> Path:
    """Get default log file path.
    
    Args:
        app_name: Application name for directory
        
    Returns:
        Path to log file
    """
    dirs = PlatformDirs(app_name)
    log_dir = Path(dirs.user_log_dir)
    return log_dir / f"{app_name}.log"