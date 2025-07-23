# Security Configuration

This document outlines secure configuration practices for crawl4ai-llm-docs.

## Environment Variables

This application requires API credentials that should NEVER be hardcoded. Instead, use environment variables:

### Required Environment Variables

```bash
# Your API key (required)
export CRAWL4AI_API_KEY="your_actual_api_key_here"

# Optional: API base URL (defaults to OpenAI)
export CRAWL4AI_BASE_URL="https://api.openai.com/v1"

# Optional: Model name (defaults to gpt-3.5-turbo)
export CRAWL4AI_MODEL="gpt-3.5-turbo"
```

### Setup Instructions

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit .env with your actual credentials:**
   ```bash
   # Edit the .env file and replace placeholder values
   nano .env
   ```

3. **Load environment variables:**
   ```bash
   # Option 1: Source the .env file
   source .env
   
   # Option 2: Use with python-dotenv
   pip install python-dotenv
   ```

### Security Best Practices

- ✅ **DO**: Use environment variables for all secrets
- ✅ **DO**: Keep .env files in .gitignore
- ✅ **DO**: Use different API keys for development/production
- ✅ **DO**: Rotate API keys regularly
- ❌ **DON'T**: Commit API keys to version control
- ❌ **DON'T**: Share .env files
- ❌ **DON'T**: Use production keys in development

### API Key Security

- Store API keys securely (password manager, secure vault)
- Monitor API usage for unusual activity
- Revoke compromised keys immediately
- Use minimum required permissions

### Configuration Validation

The application will validate that required environment variables are set and fail safely if they're missing:

```python
ValueError: CRAWL4AI_API_KEY environment variable is required for test configuration.
```

## Reporting Security Issues

If you discover a security vulnerability, please report it privately to:
- Email: security@example.com (replace with actual contact)
- Do not open public issues for security vulnerabilities