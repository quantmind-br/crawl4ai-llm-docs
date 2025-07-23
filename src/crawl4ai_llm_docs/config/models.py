"""Pydantic configuration models for crawl4ai-llm-docs."""
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class ParallelProcessingConfig(BaseSettings):
    """Configuration for parallel processing parameters."""
    
    # Core concurrency settings
    max_concurrent_requests: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum number of concurrent LLM requests"
    )
    
    # Rate limiting configuration
    enable_adaptive_rate_limiting: bool = Field(
        default=True,
        description="Enable adaptive rate limiting based on API headers"
    )
    rate_limit_buffer_percent: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Buffer percentage for rate limiting (10% = stay 10% below limits)"
    )
    
    # Progress tracking
    progress_update_interval: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Progress update interval in seconds"
    )
    
    # Session management
    enable_session_pooling: bool = Field(
        default=True,
        description="Enable HTTP session pooling for connection reuse"
    )
    max_connections_per_host: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum connections per host in the session pool"
    )
    
    # Timeout configuration
    request_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Request timeout in seconds"
    )
    
    # Error handling
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Enable circuit breaker for error handling"
    )
    error_threshold_percent: float = Field(
        default=0.15,
        ge=0.05,
        le=0.5,
        description="Error rate threshold for circuit breaker (15% = trip at 15% error rate)"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "CRAWL4AI_PARALLEL_"
        case_sensitive = False


class AppConfig(BaseSettings):
    """Main application configuration with validation."""
    
    # LLM API Configuration
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI-compatible API"
    )
    model: str = Field(
        default="gpt-4o",
        description="LLM model to use for processing"
    )
    
    # Crawling Configuration
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum number of concurrent crawling workers"
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for failed requests"
    )
    
    # Processing Configuration
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=8000,
        description="Chunk size for large document processing"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks"
    )
    max_tokens: int = Field(
        default=8000,
        ge=100,
        le=32000,
        description="Maximum tokens per API request"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Temperature for LLM processing"
    )
    
    # Application Settings
    debug: bool = Field(default=False, description="Enable debug logging")
    output_format: str = Field(
        default="markdown",
        pattern="^(markdown|md)$",
        description="Output format"
    )
    
    # Parallel Processing Configuration
    parallel_processing: ParallelProcessingConfig = Field(
        default_factory=ParallelProcessingConfig,
        description="Parallel processing configuration"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "CRAWL4AI_"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = values.get('chunk_size', 1000)
        if v >= chunk_size:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v
    
    @validator('base_url')
    def validate_base_url(cls, v):
        """Ensure base URL is properly formatted."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return v.rstrip('/')
    
    @classmethod
    def get_test_config(cls) -> 'AppConfig':
        """Get configuration with test API credentials."""
        return cls(
            api_key="sk-Z2WHmAXDKW8f31iss5iPrA",
            base_url="https://api.quantmind.com.br/v1",
            model="gemini/gemini-2.5-flash-lite-preview-06-17",
            max_tokens=16000,  # Doubled to reduce API calls and improve performance
            parallel_processing=ParallelProcessingConfig(
                max_concurrent_requests=2,  # Reduced to be more respectful to API rate limits
                enable_adaptive_rate_limiting=True,
                progress_update_interval=3  # Slightly longer to reduce noise
            )
        )


class CrawlerConfig(BaseSettings):
    """Configuration for crawl4ai specific settings."""
    
    # Browser Configuration
    user_agent_mode: str = Field(
        default="random",
        pattern="^(random|fixed)$",
        description="User agent mode for browser"
    )
    headless: bool = Field(default=True, description="Run browser in headless mode")
    viewport_width: int = Field(default=1920, ge=800, le=1920)
    viewport_height: int = Field(default=1080, ge=600, le=1080)
    use_persistent_context: bool = Field(
        default=True,
        description="Use persistent browser context"
    )
    
    # Content Filtering
    word_count_threshold: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Minimum word count for extracted content"
    )
    magic_mode: bool = Field(
        default=True,
        description="Enable crawl4ai magic mode for anti-detection"
    )
    enable_cache: bool = Field(
        default=True,
        description="Enable crawling cache"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "CRAWL4AI_CRAWLER_"