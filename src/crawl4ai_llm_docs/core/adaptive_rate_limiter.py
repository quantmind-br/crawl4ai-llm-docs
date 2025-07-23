"""
Adaptive rate limiter with API header parsing for intelligent throttling.
"""
import asyncio
import time
import logging
import random
from typing import Dict, Optional, List
from collections import deque
from dataclasses import dataclass

from ..config.models import ParallelProcessingConfig
from ..exceptions import RateLimitExceededException

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Information about current rate limits."""
    requests_remaining: Optional[int] = None
    requests_limit: Optional[int] = None
    tokens_remaining: Optional[int] = None
    tokens_limit: Optional[int] = None
    reset_time: Optional[float] = None
    retry_after: Optional[float] = None

class AdaptiveRateLimiter:
    """Intelligent rate limiting based on API response headers."""
    
    def __init__(self, config: ParallelProcessingConfig):
        self.config = config
        self.current_delay = 0.1  # Start with minimal delay
        self.request_times = deque(maxlen=100)
        self.error_count = 0
        self.consecutive_successes = 0
        self.rate_limit_info = RateLimitInfo()
        self._lock = asyncio.Lock()
        
        # Circuit breaker state
        self.circuit_breaker_open = False
        self.circuit_breaker_open_time = 0
        self.circuit_breaker_timeout = 60  # 1 minute timeout
        
        logger.info(f"AdaptiveRateLimiter initialized with buffer {config.rate_limit_buffer_percent:.1%}")
        
    async def acquire(self) -> None:
        """Acquire permission to make API request."""
        async with self._lock:
            # Check circuit breaker
            if self.circuit_breaker_open:
                if time.time() - self.circuit_breaker_open_time > self.circuit_breaker_timeout:
                    self.circuit_breaker_open = False
                    self.error_count = 0
                    logger.info("Circuit breaker reset - allowing requests")
                else:
                    remaining_time = self.circuit_breaker_timeout - (time.time() - self.circuit_breaker_open_time)
                    raise RateLimitExceededException(f"Circuit breaker open for {remaining_time:.1f}s more")
            
            # Apply current delay
            if self.current_delay > 0:
                logger.debug(f"Rate limiting delay: {self.current_delay:.2f}s")
                await asyncio.sleep(self.current_delay)
                
            # Record request time
            self.request_times.append(time.time())
            
    def on_response(self, response_headers: Dict[str, str], response_time: float) -> None:
        """Update rate limiting based on API response."""
        self.consecutive_successes += 1
        
        # Parse rate limit headers (OpenAI format)
        self._parse_rate_limit_headers(response_headers)
        
        # Adaptive adjustment based on rate limit info
        if self.rate_limit_info.requests_remaining is not None:
            remaining_ratio = self.rate_limit_info.requests_remaining / max(self.rate_limit_info.requests_limit or 1, 1)
            
            # Proactive throttling when approaching limits
            if remaining_ratio < self.config.rate_limit_buffer_percent:
                # Getting close to rate limit - increase delay
                self.current_delay = min(self.current_delay * 1.5, 10.0)
                logger.warning(f"Approaching rate limit ({remaining_ratio:.1%} remaining), increasing delay to {self.current_delay:.2f}s")
            elif remaining_ratio > 0.5:
                # Plenty of headroom - reduce delay
                self.current_delay = max(0.1, self.current_delay * 0.9)
                
        # Reduce delay on consistent success
        if self.consecutive_successes > 5:
            self.current_delay = max(0.1, self.current_delay * 0.95)
            self.consecutive_successes = 0
            logger.debug(f"Reducing delay due to success streak: {self.current_delay:.2f}s")
            
    def on_error(self, error: Exception, error_type: str = "unknown") -> None:
        """Handle API errors with appropriate backoff."""
        self.error_count += 1
        self.consecutive_successes = 0
        
        if error_type == "rate_limit" or "rate limit" in str(error).lower():
            # Significant backoff for rate limit errors
            self.current_delay = min(self.current_delay * 2, 30.0)
            logger.warning(f"Rate limit error, increasing delay to {self.current_delay:.2f}s")
            
            # Extract retry-after if available
            if hasattr(error, 'retry_after'):
                self.current_delay = max(self.current_delay, float(error.retry_after))
                
        elif error_type == "connection" or "connection" in str(error).lower():
            # Moderate backoff for connection errors
            self.current_delay = min(self.current_delay * 1.5, 15.0)
            logger.warning(f"Connection error, increasing delay to {self.current_delay:.2f}s")
            
        elif self.error_count > 3:
            # General error backoff
            self.current_delay = min(self.current_delay * 1.2, 10.0)
            logger.warning(f"Multiple errors ({self.error_count}), increasing delay to {self.current_delay:.2f}s")
            
        # Circuit breaker check
        if self.config.enable_circuit_breaker:
            recent_requests = len([t for t in self.request_times if time.time() - t < 60])  # Last minute
            if recent_requests > 0:
                error_rate = self.error_count / (recent_requests + self.error_count)
                if error_rate > self.config.error_threshold_percent:
                    self.circuit_breaker_open = True
                    self.circuit_breaker_open_time = time.time()
                    logger.error(f"Circuit breaker opened due to high error rate: {error_rate:.1%}")
                    
    def _parse_rate_limit_headers(self, headers: Dict[str, str]) -> None:
        """Parse rate limit information from API headers."""
        # OpenAI-style headers
        if 'x-ratelimit-remaining-requests' in headers:
            try:
                self.rate_limit_info.requests_remaining = int(headers['x-ratelimit-remaining-requests'])
                self.rate_limit_info.requests_limit = int(headers.get('x-ratelimit-limit-requests', 0))
            except (ValueError, TypeError):
                pass
                
        if 'x-ratelimit-remaining-tokens' in headers:
            try:
                self.rate_limit_info.tokens_remaining = int(headers['x-ratelimit-remaining-tokens'])
                self.rate_limit_info.tokens_limit = int(headers.get('x-ratelimit-limit-tokens', 0))
            except (ValueError, TypeError):
                pass
                
        if 'x-ratelimit-reset-requests' in headers:
            try:
                self.rate_limit_info.reset_time = float(headers['x-ratelimit-reset-requests'])
            except (ValueError, TypeError):
                pass
                
        if 'retry-after' in headers:
            try:
                self.rate_limit_info.retry_after = float(headers['retry-after'])
            except (ValueError, TypeError):
                pass
                
    def get_current_delay(self) -> float:
        """Get current delay for monitoring."""
        return self.current_delay
        
    def get_rate_limit_status(self) -> Dict[str, any]:
        """Get current rate limiting status."""
        recent_requests = len([t for t in self.request_times if time.time() - t < 60])
        
        return {
            "current_delay": self.current_delay,
            "error_count": self.error_count,
            "consecutive_successes": self.consecutive_successes,
            "requests_last_minute": recent_requests,
            "circuit_breaker_open": self.circuit_breaker_open,
            "rate_limit_info": {
                "requests_remaining": self.rate_limit_info.requests_remaining,
                "requests_limit": self.rate_limit_info.requests_limit,
                "tokens_remaining": self.rate_limit_info.tokens_remaining,
                "tokens_limit": self.rate_limit_info.tokens_limit,
            }
        }
        
    def reset_statistics(self) -> None:
        """Reset error statistics (useful for testing)."""
        self.error_count = 0
        self.consecutive_successes = 0
        self.circuit_breaker_open = False
        self.current_delay = 0.1
        logger.info("Rate limiter statistics reset")