"""Configuration manager with cross-platform support."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from platformdirs import PlatformDirs
from pydantic import ValidationError

from .models import AppConfig, CrawlerConfig


logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration with cross-platform support."""
    
    def __init__(self, app_name: str = "crawl4ai-llm-docs"):
        """Initialize configuration manager.
        
        Args:
            app_name: Application name for directory creation
        """
        self.app_name = app_name
        self.dirs = PlatformDirs(app_name)
        self.config_dir = Path(self.dirs.user_config_dir)
        self.config_file = self.config_dir / "config.json"
        
        # Ensure config directory exists
        self._ensure_config_directory()
        
        # Config instances
        self._app_config: Optional[AppConfig] = None
        self._crawler_config: Optional[CrawlerConfig] = None
    
    def _ensure_config_directory(self) -> None:
        """Ensure configuration directory exists."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Config directory ensured: {self.config_dir}")
        except PermissionError as e:
            logger.error(f"Cannot create config directory: {e}")
            raise
    
    @property
    def app_config(self) -> AppConfig:
        """Get application configuration, loading if necessary."""
        if self._app_config is None:
            self._app_config = self.load_config()
        return self._app_config
    
    @property
    def crawler_config(self) -> CrawlerConfig:
        """Get crawler configuration."""
        if self._crawler_config is None:
            self._crawler_config = CrawlerConfig()
        return self._crawler_config
    
    def load_config(self) -> AppConfig:
        """Load configuration from file or create default.
        
        Returns:
            AppConfig instance
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                logger.info(f"Loaded configuration from {self.config_file}")
                return AppConfig(**config_data)
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to load config: {e}")
                logger.info("Creating default configuration")
        
        # Create default configuration with test credentials
        config = AppConfig.get_test_config()
        return config
    
    def save_config(self, config: Optional[AppConfig] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save. Uses current if None.
        """
        if config:
            self._app_config = config
        
        # If no config instance, try to get the current one
        if not self._app_config:
            self._app_config = self.app_config
        
        if not self._app_config:
            logger.error("No configuration to save")
            return
        
        try:
            config_dict = self._app_config.model_dump()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except (IOError, OSError) as e:
            logger.error(f"Failed to save config: {e}")
            raise
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values.
        
        Args:
            **kwargs: Configuration values to update
        """
        try:
            current_dict = self.app_config.model_dump()
            current_dict.update(kwargs)
            
            # Validate new configuration
            new_config = AppConfig(**current_dict)
            self._app_config = new_config
            
            logger.debug(f"Updated configuration: {list(kwargs.keys())}")
            
        except ValidationError as e:
            logger.error(f"Invalid configuration update: {e}")
            raise
    
    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._app_config = AppConfig().get_test_config()
        logger.info("Configuration reset to defaults")
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information for display.
        
        Returns:
            Dictionary with configuration information
        """
        return {
            "config_file": str(self.config_file),
            "config_dir": str(self.config_dir),
            "app_name": self.app_name,
            "config_exists": self.config_file.exists(),
            "directories": {
                "config": str(self.config_dir),
                "data": str(self.dirs.user_data_dir),
                "cache": str(self.dirs.user_cache_dir),
                "logs": str(self.dirs.user_log_dir)
            }
        }
    
    def is_configured(self) -> bool:
        """Check if application is configured.
        
        Returns:
            True if configuration is valid (with API key)
        """
        try:
            config = self.load_config()
            return bool(config.api_key)
        except (ValidationError, ValueError, json.JSONDecodeError):
            return False
    
    def validate_api_connection(self) -> bool:
        """Validate API connection with current configuration.
        
        Returns:
            True if API connection is valid
        """
        try:
            config = self.app_config
            # Basic validation - check required fields
            return bool(config.api_key and config.base_url and config.model)
        except Exception as e:
            logger.error(f"API validation failed: {e}")
            return False
    
    def validate_parallel_processing_config(self) -> bool:
        """Validate parallel processing configuration.
        
        Returns:
            True if parallel processing configuration is valid
        """
        try:
            config = self.app_config
            parallel_config = config.parallel_processing
            
            # Validate concurrency settings
            if not (1 <= parallel_config.max_concurrent_requests <= 20):
                logger.error(f"Invalid max_concurrent_requests: {parallel_config.max_concurrent_requests}")
                return False
            
            # Validate rate limiting settings
            if not (0.0 <= parallel_config.rate_limit_buffer_percent <= 0.5):
                logger.error(f"Invalid rate_limit_buffer_percent: {parallel_config.rate_limit_buffer_percent}")
                return False
            
            # Validate progress tracking settings
            if not (1 <= parallel_config.progress_update_interval <= 60):
                logger.error(f"Invalid progress_update_interval: {parallel_config.progress_update_interval}")
                return False
            
            # Validate timeout settings
            if not (5 <= parallel_config.request_timeout <= 300):
                logger.error(f"Invalid request_timeout: {parallel_config.request_timeout}")
                return False
            
            # Validate error threshold
            if not (0.05 <= parallel_config.error_threshold_percent <= 0.5):
                logger.error(f"Invalid error_threshold_percent: {parallel_config.error_threshold_percent}")
                return False
            
            logger.debug("Parallel processing configuration is valid")
            return True
            
        except Exception as e:
            logger.error(f"Parallel processing validation failed: {e}")
            return False