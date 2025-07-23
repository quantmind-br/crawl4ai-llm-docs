"""Tests for configuration system."""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from crawl4ai_llm_docs.config.manager import ConfigManager
from crawl4ai_llm_docs.config.models import AppConfig, CrawlerConfig
from crawl4ai_llm_docs.exceptions import ConfigurationError


class TestAppConfig:
    """Test AppConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AppConfig().get_test_config()
        
        assert config.api_key == "PLACEHOLDER_API_KEY_REMOVED"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gemini-2.5-flash"
        assert config.max_workers == 4
        assert config.temperature == 0.1
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = AppConfig(
            api_key="test-key",
            base_url="https://api.example.com/v1"
        )
        assert config.api_key == "test-key"
        
        # Invalid base URL
        with pytest.raises(ValueError):
            AppConfig(
                api_key="test-key",
                base_url="invalid-url"
            )
    
    def test_chunk_overlap_validation(self):
        """Test chunk overlap validation."""
        # Valid configuration
        config = AppConfig(
            api_key="test-key",
            chunk_size=1000,
            chunk_overlap=200
        )
        assert config.chunk_overlap == 200
        
        # Invalid configuration - overlap >= chunk_size
        with pytest.raises(ValueError):
            AppConfig(
                api_key="test-key",
                chunk_size=1000,
                chunk_overlap=1000
            )


class TestCrawlerConfig:
    """Test CrawlerConfig model."""
    
    def test_default_crawler_config(self):
        """Test default crawler configuration."""
        config = CrawlerConfig()
        
        assert config.user_agent_mode == "random"
        assert config.headless is True
        assert config.magic_mode is True
        assert config.word_count_threshold == 200


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Setup test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    @patch('crawl4ai_llm_docs.config.manager.PlatformDirs')
    def test_config_manager_init(self, mock_platform_dirs):
        """Test ConfigManager initialization."""
        mock_dirs = Mock()
        mock_dirs.user_config_dir = str(self.temp_path)
        mock_platform_dirs.return_value = mock_dirs
        
        manager = ConfigManager()
        
        assert manager.app_name == "crawl4ai-llm-docs"
        assert manager.config_dir == self.temp_path
    
    @patch('crawl4ai_llm_docs.config.manager.PlatformDirs')
    def test_load_config_from_file(self, mock_platform_dirs):
        """Test loading configuration from existing file."""
        mock_dirs = Mock()
        mock_dirs.user_config_dir = str(self.temp_path)
        mock_platform_dirs.return_value = mock_dirs
        
        # Create config file
        config_file = self.temp_path / "config.json"
        config_data = {
            "api_key": "test-key",
            "base_url": "https://api.example.com/v1",
            "model": "gpt-4"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        manager = ConfigManager()
        config = manager.load_config()
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com/v1"
        assert config.model == "gpt-4"
    
    @patch('crawl4ai_llm_docs.config.manager.PlatformDirs')
    def test_load_config_no_file(self, mock_platform_dirs):
        """Test loading configuration when no file exists."""
        mock_dirs = Mock()
        mock_dirs.user_config_dir = str(self.temp_path)
        mock_platform_dirs.return_value = mock_dirs
        
        manager = ConfigManager()
        config = manager.load_config()
        
        # Should return test configuration
        assert config.api_key == "PLACEHOLDER_API_KEY_REMOVED"
        assert config.base_url == "https://api.openai.com/v1"
    
    @patch('crawl4ai_llm_docs.config.manager.PlatformDirs')
    def test_save_config(self, mock_platform_dirs):
        """Test saving configuration to file."""
        mock_dirs = Mock()
        mock_dirs.user_config_dir = str(self.temp_path)
        mock_platform_dirs.return_value = mock_dirs
        
        manager = ConfigManager()
        config = AppConfig(api_key="new-key", base_url="https://api.example.com/v1")
        
        manager.save_config(config)
        
        # Verify file was created
        config_file = self.temp_path / "config.json"
        assert config_file.exists()
        
        # Verify content
        with open(config_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["api_key"] == "new-key"
    
    @patch('crawl4ai_llm_docs.config.manager.PlatformDirs')
    def test_update_config(self, mock_platform_dirs):
        """Test updating configuration."""
        mock_dirs = Mock()
        mock_dirs.user_config_dir = str(self.temp_path)
        mock_platform_dirs.return_value = mock_dirs
        
        manager = ConfigManager()
        
        # Update configuration
        manager.update_config(api_key="updated-key", model="gpt-4")
        
        config = manager.app_config
        assert config.api_key == "updated-key"
        assert config.model == "gpt-4"
    
    @patch('crawl4ai_llm_docs.config.manager.PlatformDirs')
    def test_is_configured(self, mock_platform_dirs):
        """Test configuration validation."""
        mock_dirs = Mock()
        mock_dirs.user_config_dir = str(self.temp_path)
        mock_platform_dirs.return_value = mock_dirs
        
        manager = ConfigManager()
        
        # Should be configured with test config
        assert manager.is_configured() is True
    
    @patch('crawl4ai_llm_docs.config.manager.PlatformDirs')
    def test_get_config_info(self, mock_platform_dirs):
        """Test getting configuration information."""
        mock_dirs = Mock()
        mock_dirs.user_config_dir = str(self.temp_path)
        mock_dirs.user_data_dir = str(self.temp_path / "data")
        mock_dirs.user_cache_dir = str(self.temp_path / "cache")
        mock_dirs.user_log_dir = str(self.temp_path / "logs")
        mock_platform_dirs.return_value = mock_dirs
        
        manager = ConfigManager()
        info = manager.get_config_info()
        
        assert "config_file" in info
        assert "directories" in info
        assert "config_exists" in info
        assert info["app_name"] == "crawl4ai-llm-docs"
    
    @patch('crawl4ai_llm_docs.config.manager.PlatformDirs')
    def test_validate_api_connection(self, mock_platform_dirs):
        """Test API connection validation."""
        mock_dirs = Mock()
        mock_dirs.user_config_dir = str(self.temp_path)
        mock_platform_dirs.return_value = mock_dirs
        
        manager = ConfigManager()
        
        # Should validate successfully with test config
        assert manager.validate_api_connection() is True