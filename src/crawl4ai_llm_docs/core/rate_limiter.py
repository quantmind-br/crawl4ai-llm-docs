"""Adaptive rate limiter with API header parsing and circuit breaker."""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..config.models import ParallelProcessingConfig


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker tripped
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RateLimitState:
    """Current rate limit state from API headers."""
    requests_remaining: Optional[int] = None
    tokens_remaining: Optional[int] = None
    reset_time: Optional[float] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class RequestRecord:
    """Record of a single request for rate limiting."""
    timestamp: float
    tokens_used: int
    success: bool
    response_time: float = 0.0


class AdaptiveRateLimiter:
    """Adaptive rate limiter with circuit breaker and API header parsing."""
    
    def __init__(self, config: ParallelProcessingConfig):
        """Initialize rate limiter with configuration.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config
        
        # Rate limiting state
        self.rate_limit_state = RateLimitState()
        self.request_history: List[RequestRecord] = []
        self._lock = asyncio.Lock()
        
        # Circuit breaker state
        self.circuit_state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.circuit_open_time: Optional[float] = None
        self.circuit_test_time = 30.0  # Test circuit after 30 seconds
        
        # Default rate limits (will be updated from API headers)
        self.default_rpm = 60  # requests per minute
        self.default_tpm = 40000  # tokens per minute
        
        # Rate limiting window (60 seconds)
        self.window_duration = 60.0
    
    async def acquire(self, estimated_tokens: int = 1000) -> bool:
        """Acquire permission to make a request.
        
        Args:
            estimated_tokens: Estimated token usage for the request
            
        Returns:
            True if request can proceed
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
        """
        async with self._lock:
            # Check circuit breaker
            await self._check_circuit_breaker()
            
            if self.circuit_state == CircuitState.OPEN:
                from ..exceptions import CircuitBreakerOpenError
                raise CircuitBreakerOpenError("Circuit breaker is open due to consecutive failures")
            
            # Update rate limit state if available
            current_time = time.time()
            
            # Clean old requests from sliding window
            self._clean_old_requests(current_time)
            
            # Check if we can make the request
            if await self._should_throttle(current_time, estimated_tokens):
                # Calculate wait time
                wait_time = await self._calculate_wait_time(current_time, estimated_tokens)
                if wait_time > 0:
                    logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
            
            # Record the request attempt
            request_record = RequestRecord(
                timestamp=current_time,
                tokens_used=estimated_tokens,
                success=True  # Will be updated after response
            )
            self.request_history.append(request_record)
            
            return True
    
    def _clean_old_requests(self, current_time: float) -> None:
        """Remove requests older than the window duration."""
        cutoff_time = current_time - self.window_duration
        self.request_history = [
            record for record in self.request_history 
            if record.timestamp > cutoff_time
        ]
    
    async def _should_throttle(self, current_time: float, estimated_tokens: int) -> bool:
        """Determine if we should throttle the request.
        
        Args:
            current_time: Current timestamp
            estimated_tokens: Estimated token usage
            
        Returns:
            True if request should be throttled
        """
        if not self.config.enable_adaptive_rate_limiting:
            return False
        
        # Count recent requests and tokens
        recent_requests = len(self.request_history)
        recent_tokens = sum(record.tokens_used for record in self.request_history)
        
        # Use API header limits if available, otherwise use defaults
        rpm_limit = self._get_effective_rpm_limit()
        tpm_limit = self._get_effective_tpm_limit()
        
        # Apply buffer to stay below limits
        buffer = self.config.rate_limit_buffer_percent
        effective_rpm_limit = rpm_limit * (1 - buffer)
        effective_tpm_limit = tpm_limit * (1 - buffer)
        
        # Check if adding this request would exceed limits
        if recent_requests >= effective_rpm_limit:
            logger.debug(f"Request throttling: {recent_requests} >= {effective_rpm_limit} RPM")
            return True
        
        if recent_tokens + estimated_tokens >= effective_tpm_limit:
            logger.debug(f"Token throttling: {recent_tokens + estimated_tokens} >= {effective_tpm_limit} TPM")
            return True
        
        return False
    
    def _get_effective_rpm_limit(self) -> int:
        """Get effective requests per minute limit."""
        if (self.rate_limit_state.requests_remaining is not None and 
            self.rate_limit_state.reset_time is not None):
            # Calculate RPM from API headers
            time_until_reset = self.rate_limit_state.reset_time - time.time()
            if time_until_reset > 0:
                return max(1, int(self.rate_limit_state.requests_remaining * 60 / time_until_reset))
        
        return self.default_rpm
    
    def _get_effective_tpm_limit(self) -> int:
        """Get effective tokens per minute limit."""
        if (self.rate_limit_state.tokens_remaining is not None and 
            self.rate_limit_state.reset_time is not None):
            # Calculate TPM from API headers
            time_until_reset = self.rate_limit_state.reset_time - time.time()
            if time_until_reset > 0:
                return max(100, int(self.rate_limit_state.tokens_remaining * 60 / time_until_reset))
        
        return self.default_tpm
    
    async def _calculate_wait_time(self, current_time: float, estimated_tokens: int) -> float:
        """Calculate how long to wait before next request.
        
        Args:
            current_time: Current timestamp
            estimated_tokens: Estimated token usage
            
        Returns:
            Wait time in seconds
        """
        if not self.request_history:
            return 0.0
        
        # Time until oldest request is outside the window
        oldest_request = min(record.timestamp for record in self.request_history)
        time_until_window_reset = self.window_duration - (current_time - oldest_request)
        
        # If we have API reset time, use that too
        if self.rate_limit_state.reset_time:
            api_reset_wait = self.rate_limit_state.reset_time - current_time
            if api_reset_wait > 0:
                time_until_window_reset = min(time_until_window_reset, api_reset_wait)
        
        return max(0.0, time_until_window_reset)
    
    async def _check_circuit_breaker(self) -> None:
        """Check and update circuit breaker state."""
        current_time = time.time()
        
        if self.circuit_state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if (self.circuit_open_time and 
                current_time - self.circuit_open_time >= self.circuit_test_time):
                self.circuit_state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to half-open state")
        
        elif self.circuit_state == CircuitState.HALF_OPEN:
            # In half-open state, allow limited requests to test recovery
            pass
    
    def update_from_response(self, headers: Dict[str, str], tokens_used: int, 
                           success: bool, response_time: float = 0.0) -> None:
        """Update rate limiter state from API response.
        
        Args:
            headers: HTTP response headers
            tokens_used: Actual tokens used in the request
            success: Whether the request was successful
            response_time: Response time in seconds
        """
        current_time = time.time()
        
        # Update the most recent request record
        if self.request_history:
            self.request_history[-1].tokens_used = tokens_used
            self.request_history[-1].success = success
            self.request_history[-1].response_time = response_time
        
        # Parse rate limit headers (OpenAI style)
        self._parse_rate_limit_headers(headers, current_time)
        
        # Update circuit breaker state
        self._update_circuit_breaker(success)
        
        logger.debug(f"Updated rate limiter: success={success}, tokens={tokens_used}, "
                    f"circuit_state={self.circuit_state.value}")
    
    def _parse_rate_limit_headers(self, headers: Dict[str, str], current_time: float) -> None:
        """Parse rate limit information from response headers.
        
        Args:
            headers: HTTP response headers
            current_time: Current timestamp
        """
        # OpenAI-style headers
        if 'x-ratelimit-remaining-requests' in headers:
            try:
                self.rate_limit_state.requests_remaining = int(headers['x-ratelimit-remaining-requests'])
            except ValueError:
                pass
        
        if 'x-ratelimit-remaining-tokens' in headers:
            try:
                self.rate_limit_state.tokens_remaining = int(headers['x-ratelimit-remaining-tokens'])
            except ValueError:
                pass
        
        # Parse reset time (different formats supported)
        for reset_header in ['x-ratelimit-reset-requests', 'x-ratelimit-reset']:
            if reset_header in headers:
                try:
                    reset_value = headers[reset_header]
                    if 'ms' in reset_value:
                        # Milliseconds from now
                        reset_ms = int(reset_value.replace('ms', ''))
                        self.rate_limit_state.reset_time = current_time + (reset_ms / 1000)
                    elif reset_value.isdigit():
                        # Seconds from now
                        self.rate_limit_state.reset_time = current_time + int(reset_value)
                    else:
                        # Unix timestamp
                        self.rate_limit_state.reset_time = float(reset_value)
                    break
                except (ValueError, TypeError):
                    continue
        
        self.rate_limit_state.last_updated = current_time
    
    def _update_circuit_breaker(self, success: bool) -> None:
        """Update circuit breaker state based on request success.
        
        Args:
            success: Whether the request was successful
        """
        if success:
            # Reset failure count on success
            self.consecutive_failures = 0
            
            # Close circuit if it was half-open
            if self.circuit_state == CircuitState.HALF_OPEN:
                self.circuit_state = CircuitState.CLOSED
                logger.info("Circuit breaker closed after successful request")
        
        else:
            # Increment failure count
            self.consecutive_failures += 1
            
            # Check if we should open the circuit
            error_threshold = int(self.config.error_threshold_percent * 100)  # Convert to percentage
            if self.consecutive_failures >= error_threshold:
                self.circuit_state = CircuitState.OPEN
                self.circuit_open_time = time.time()
                logger.warning(f"Circuit breaker opened after {self.consecutive_failures} consecutive failures")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics.
        
        Returns:
            Dictionary with rate limiter statistics
        """
        current_time = time.time()
        self._clean_old_requests(current_time)
        
        recent_requests = len(self.request_history)
        recent_tokens = sum(record.tokens_used for record in self.request_history)
        successful_requests = sum(1 for record in self.request_history if record.success)
        
        avg_response_time = 0.0
        if self.request_history:
            avg_response_time = sum(record.response_time for record in self.request_history) / len(self.request_history)
        
        return {
            'circuit_state': self.circuit_state.value,
            'consecutive_failures': self.consecutive_failures,
            'recent_requests': recent_requests,
            'recent_tokens': recent_tokens,
            'success_rate': successful_requests / recent_requests if recent_requests > 0 else 1.0,
            'average_response_time': avg_response_time,
            'effective_rpm_limit': self._get_effective_rpm_limit(),
            'effective_tpm_limit': self._get_effective_tpm_limit(),
            'rate_limit_state': {
                'requests_remaining': self.rate_limit_state.requests_remaining,
                'tokens_remaining': self.rate_limit_state.tokens_remaining,
                'reset_time': self.rate_limit_state.reset_time,
                'last_updated': self.rate_limit_state.last_updated
            }
        }