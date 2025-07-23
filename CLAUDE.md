# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup and Dependencies

This is a Python CLI application that scrapes documentation websites using Crawl4AI and processes content with OpenAI-compatible APIs. Always use virtual environments for development:

- Use `uv venv` to create virtual environments 
- Use `uv pip install -e .` for development installation
- Use `uv pip install -e .[dev]` to install with development dependencies

Key dependencies include:
- crawl4ai>=0.7.0 (web scraping with anti-detection)
- openai>=1.0.0 (LLM API integration)
- click>=8.0.0 (CLI framework)
- pydantic>=2.0.0 (configuration models)
- rich>=13.0.0 (terminal output formatting)

## Development Commands

**Installation and Setup:**
```bash
# Create virtual environment
uv venv

# Install in development mode with dev dependencies
uv pip install -e .[dev]
```

**Testing:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_cli.py

# Run with verbose output
pytest -v

# Run async tests
pytest tests/ -v
```

**Code Quality:**
```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

**Running the Application:**
```bash
# Run via module
python -m crawl4ai_llm_docs

# Or use entry point after installation
crawl4ai-llm-docs

# Show help
crawl4ai-llm-docs --help

# Configure API settings
crawl4ai-llm-docs --config

# Show configuration info
crawl4ai-llm-docs --info
```

## Project Architecture

This is a modular CLI application with the following key components:

**Core Processing Pipeline:**
1. **CLI Interface** (`src/crawl4ai_llm_docs/cli.py`) - Click-based CLI with Rich formatting
2. **Configuration Management** (`src/crawl4ai_llm_docs/config/`) - Persistent config with Pydantic models
3. **Web Scraping** (`src/crawl4ai_llm_docs/core/scraper.py`) - Async scraping with Crawl4AI and anti-detection
4. **LLM Processing** (`src/crawl4ai_llm_docs/core/processor.py`) - Documentation consolidation via OpenAI-compatible APIs
5. **File Operations** (`src/crawl4ai_llm_docs/utils/file_handler.py`) - URL file reading and markdown output

**Configuration System:**
- Cross-platform config storage in `~/.crawl4ai-llm-docs/config.json`
- Supports OpenAI-compatible APIs (OpenAI, local models, etc.)
- Configurable via interactive CLI or direct file editing

**Key Design Patterns:**
- Async/await for concurrent web scraping with rate limiting
- Retry logic with exponential backoff for reliability
- Token counting and chunking for large documentation sets
- Rich console output for better user experience
- Cross-platform compatibility (Windows/Linux)

## Architecture Components

**Entry Points:**
- `src/crawl4ai_llm_docs/cli.py:main()` - Main CLI interface
- `src/crawl4ai_llm_docs/__main__.py` - Module execution with Windows UTF-8 fixes

**Core Classes:**
- `DocumentationScraper` - Handles batch URL scraping with Crawl4AI
- `DocumentationProcessor` - LLM-based content consolidation 
- `ConfigManager` - Cross-platform configuration management
- `ScrapedDocument` - Data class for scraped content with metadata

**Error Handling:**
- Custom exception hierarchy in `exceptions.py`
- Graceful fallbacks for missing dependencies
- Comprehensive logging throughout the pipeline

## File Structure

```
src/crawl4ai_llm_docs/
├── __init__.py                 # Package initialization
├── __main__.py                 # Module entry point
├── cli.py                      # Main CLI interface
├── exceptions.py               # Custom exceptions
├── config/
│   ├── __init__.py
│   ├── manager.py             # Configuration management
│   └── models.py              # Pydantic configuration models
├── core/
│   ├── __init__.py
│   ├── scraper.py             # Web scraping with Crawl4AI
│   ├── processor.py           # LLM processing
│   └── optimizer.py           # Content optimization (placeholder)
└── utils/
    ├── __init__.py
    ├── file_handler.py        # File I/O operations
    └── logger.py              # Logging utilities
```

## Testing Strategy

Tests are organized by component with comprehensive mocking:
- `tests/test_cli.py` - CLI interface and user interactions
- `tests/test_config.py` - Configuration management
- `tests/fixtures/` - Test data and sample files

Use pytest with async support for testing the scraping pipeline.