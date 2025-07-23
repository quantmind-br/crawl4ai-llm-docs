# crawl4ai-llm-docs

A cross-platform Python CLI application that scrapes documentation websites using Crawl4AI, processes content with OpenAI-compatible APIs, and outputs LLM-optimized markdown files.

## Features

- Web scraping with anti-detection capabilities using Crawl4AI
- LLM-powered content consolidation via OpenAI-compatible APIs
- Cross-platform compatibility (Windows/Linux)
- Interactive CLI interface
- Configuration management with persistent storage
- LLM-optimized markdown output
- pipx installable

## Installation

```bash
pipx install crawl4ai-llm-docs
```

## Usage

1. Create a text file with URLs (one per line)
2. Run the tool:
   ```bash
   crawl4ai-llm-docs
   ```
3. Follow the interactive prompts to configure and process your documentation

## Configuration

The tool stores configuration in `~/.crawl4ai-llm-docs/config.json` for persistent settings across sessions.

## License

MIT License