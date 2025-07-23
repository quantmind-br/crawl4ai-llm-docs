"""Tests for CLI interface."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from click.testing import CliRunner

from crawl4ai_llm_docs.cli import main
from crawl4ai_llm_docs.config.manager import ConfigManager
from crawl4ai_llm_docs.config.models import AppConfig


class TestCLI:
    """Test CLI functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()
    
    def test_version_option(self):
        """Test --version option."""
        result = self.runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '0.1.0' in result.output
    
    def test_help_option(self):
        """Test --help option."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Process documentation URLs' in result.output
        assert '--config' in result.output
        assert '--info' in result.output
    
    @patch('crawl4ai_llm_docs.cli.ConfigManager')
    def test_config_option(self, mock_config_manager):
        """Test --config option for interactive configuration."""
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        # Mock interactive configuration
        with patch('crawl4ai_llm_docs.cli.interactive_configuration') as mock_config:
            result = self.runner.invoke(main, ['--config'])
            assert result.exit_code == 0
            mock_config.assert_called_once_with(mock_manager)
    
    @patch('crawl4ai_llm_docs.cli.ConfigManager')
    def test_info_option(self, mock_config_manager):
        """Test --info option for showing configuration."""
        mock_manager = Mock()
        mock_config_manager.return_value = mock_manager
        
        # Mock display configuration
        with patch('crawl4ai_llm_docs.cli.display_config_info') as mock_display:
            result = self.runner.invoke(main, ['--info'])
            assert result.exit_code == 0
            mock_display.assert_called_once_with(mock_manager)
    
    @patch('crawl4ai_llm_docs.cli.process_urls_file')
    @patch('crawl4ai_llm_docs.cli.ConfigManager')
    def test_process_existing_file(self, mock_config_manager, mock_process):
        """Test processing with existing URLs file."""
        # Create a temporary file
        with self.runner.isolated_filesystem():
            test_file = Path('test_urls.txt')
            test_file.write_text('https://example.com')
            
            result = self.runner.invoke(main, [str(test_file)])
            assert result.exit_code == 0
            mock_process.assert_called_once()
    
    def test_nonexistent_file(self):
        """Test error handling for non-existent file."""
        result = self.runner.invoke(main, ['nonexistent.txt'])
        assert result.exit_code == 1
        assert 'URLs file not found' in result.output
    
    @patch('crawl4ai_llm_docs.cli.get_urls_file_interactively')
    @patch('crawl4ai_llm_docs.cli.process_urls_file')
    @patch('crawl4ai_llm_docs.cli.ConfigManager')
    def test_interactive_file_selection(self, mock_config_manager, mock_process, mock_get_file):
        """Test interactive file selection when no file provided."""
        mock_file = Path('interactive.txt')
        mock_get_file.return_value = mock_file
        
        with patch.object(Path, 'exists', return_value=True):
            result = self.runner.invoke(main, [])
            assert result.exit_code == 0
            mock_get_file.assert_called_once()
            mock_process.assert_called_once()


class TestInteractiveConfiguration:
    """Test interactive configuration functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()
    
    @patch('crawl4ai_llm_docs.cli.Prompt')
    @patch('crawl4ai_llm_docs.cli.Confirm')
    def test_interactive_configuration_success(self, mock_confirm, mock_prompt):
        """Test successful interactive configuration."""
        # Mock user inputs
        mock_prompt.ask.side_effect = [
            'test-api-key',
            'https://api.example.com/v1',
            'gpt-4',
            '4'
        ]
        mock_confirm.ask.return_value = False
        
        # Mock config manager
        config_manager = Mock()
        config_manager.app_config = AppConfig().get_test_config()
        
        from crawl4ai_llm_docs.cli import interactive_configuration
        
        # Should not raise exception
        interactive_configuration(config_manager)
        config_manager.save_config.assert_called_once()
    
    @patch('crawl4ai_llm_docs.cli.Prompt')
    def test_get_urls_file_interactively_success(self, mock_prompt):
        """Test interactive URLs file selection - success case."""
        mock_prompt.ask.return_value = 'test_urls.txt'
        
        from crawl4ai_llm_docs.cli import get_urls_file_interactively
        
        with patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'is_file', return_value=True):
            
            result = get_urls_file_interactively()
            assert result == Path('test_urls.txt')
    
    @patch('crawl4ai_llm_docs.cli.Prompt')
    @patch('crawl4ai_llm_docs.cli.Confirm')
    def test_get_urls_file_interactively_retry(self, mock_confirm, mock_prompt):
        """Test interactive URLs file selection - retry on invalid file."""
        mock_prompt.ask.side_effect = ['invalid.txt', 'valid.txt']
        mock_confirm.ask.return_value = True  # Try again
        
        from crawl4ai_llm_docs.cli import get_urls_file_interactively
        
        def mock_exists(self):
            return str(self) == 'valid.txt'
        
        with patch.object(Path, 'exists', mock_exists), \
             patch.object(Path, 'is_file', return_value=True):
            
            result = get_urls_file_interactively()
            assert result == Path('valid.txt')


class TestProcessUrlsFile:
    """Test URL file processing functionality."""
    
    @patch('crawl4ai_llm_docs.cli.DocumentationProcessor')
    @patch('crawl4ai_llm_docs.cli.DocumentationScraper')
    @patch('crawl4ai_llm_docs.cli.FileHandler')
    def test_process_urls_file_success(self, mock_file_handler, mock_scraper, mock_processor):
        """Test successful URL file processing."""
        # Setup mocks
        config_manager = Mock()
        config_manager.is_configured.return_value = True
        config_manager.app_config = AppConfig().get_test_config()
        config_manager.crawler_config = Mock()
        
        mock_handler = Mock()
        mock_file_handler.return_value = mock_handler
        mock_handler.read_urls_file.return_value = ['https://example.com']
        
        mock_scraper_instance = Mock()
        mock_scraper.return_value = mock_scraper_instance
        mock_scraper_instance.scrape_urls.return_value = [Mock(success=True, markdown='# Test')]
        
        mock_processor_instance = Mock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.consolidate_documentation.return_value = '# Consolidated'
        
        from crawl4ai_llm_docs.cli import process_urls_file
        
        # Should not raise exception
        urls_file = Path('test.txt')
        process_urls_file(urls_file, config_manager)
        
        mock_handler.read_urls_file.assert_called_once_with(urls_file)
        mock_scraper_instance.scrape_urls.assert_called_once()
        mock_processor_instance.consolidate_documentation.assert_called_once()
        mock_handler.write_markdown_file.assert_called_once()
    
    @patch('crawl4ai_llm_docs.cli.ConfigManager')
    def test_process_urls_file_not_configured(self, mock_config_manager):
        """Test processing when not configured."""
        mock_manager = Mock()
        mock_manager.is_configured.return_value = False
        mock_config_manager.return_value = mock_manager
        
        from crawl4ai_llm_docs.cli import process_urls_file
        
        urls_file = Path('test.txt')
        
        # Should handle gracefully (print message and return)
        process_urls_file(urls_file, mock_manager)
        mock_manager.is_configured.assert_called_once()