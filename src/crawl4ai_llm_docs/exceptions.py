"""Custom exceptions for crawl4ai-llm-docs."""


class CrawlAILLMDocsError(Exception):
    """Base exception for crawl4ai-llm-docs."""
    pass


class FileOperationError(CrawlAILLMDocsError):
    """Raised when file operations fail."""
    pass


class ConfigurationError(CrawlAILLMDocsError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(CrawlAILLMDocsError):
    """Raised when input validation fails."""
    pass


class ScrapingError(CrawlAILLMDocsError):
    """Raised when web scraping operations fail."""
    pass


class ProcessingError(CrawlAILLMDocsError):
    """Raised when LLM processing operations fail."""
    pass


class APIError(CrawlAILLMDocsError):
    """Raised when API operations fail."""
    pass


# Parallel processing specific exceptions
class RateLimitExceededException(CrawlAILLMDocsError):
    """Raised when API rate limits are exceeded."""
    pass


class CircuitBreakerOpenError(CrawlAILLMDocsError):
    """Raised when circuit breaker is open due to consecutive failures."""
    pass


class ConcurrencyLimitException(CrawlAILLMDocsError):
    """Raised when concurrency limits are exceeded."""
    pass


class SessionPoolExhaustedException(CrawlAILLMDocsError):
    """Raised when HTTP session pool is exhausted."""
    pass