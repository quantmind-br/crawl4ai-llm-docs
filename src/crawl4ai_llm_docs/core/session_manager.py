"""HTTP session manager with connection pooling and resource management."""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.models import ParallelProcessingConfig


logger = logging.getLogger(__name__)


class SessionManager:
    """HTTP session manager with connection pooling and automatic cleanup."""
    
    def __init__(self, config: ParallelProcessingConfig):
        """Initialize session manager with configuration.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        
        # Connection configuration
        self.connector_config = {
            'limit': self.config.max_connections_per_host * 5,  # Total pool size
            'limit_per_host': self.config.max_connections_per_host,
            'ttl_dns_cache': 300,  # DNS cache TTL in seconds
            'use_dns_cache': True,
            'keepalive_timeout': 30,
            'enable_cleanup_closed': True
        }
        
        # Timeout configuration
        self.timeout_config = aiohttp.ClientTimeout(
            total=self.config.request_timeout,
            connect=min(10, self.config.request_timeout // 3),
            sock_read=self.config.request_timeout - 5
        )
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling.
        
        Returns:
            Configured aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    await self._create_session()
        
        return self._session
    
    async def _create_session(self) -> None:
        """Create new HTTP session with connection pooling."""
        # Create TCP connector with connection pooling
        connector = aiohttp.TCPConnector(**self.connector_config)
        
        # Create session with connector and timeout
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout_config,
            headers={
                'User-Agent': 'crawl4ai-llm-docs/1.0'
            }
        )
        
        logger.debug(f"Created HTTP session with connection pool: "
                    f"max_connections={self.connector_config['limit']}, "
                    f"max_per_host={self.connector_config['limit_per_host']}")
    
    @asynccontextmanager
    async def request(self, method: str, url: str, **kwargs):
        """Context manager for making HTTP requests with automatic session management.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for aiohttp request
            
        Yields:
            aiohttp.ClientResponse
        """
        session = await self.get_session()
        
        try:
            async with session.request(method, url, **kwargs) as response:
                yield response
        except Exception as e:
            logger.debug(f"HTTP request failed: {method} {url} - {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def make_request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request arguments
            
        Returns:
            HTTP response
            
        Raises:
            aiohttp.ClientError: If request fails after retries
        """
        async with self.request(method, url, **kwargs) as response:
            # Read response content to ensure connection can be reused
            await response.read()
            return response
    
    async def close(self) -> None:
        """Close HTTP session and cleanup connections."""
        if self._session and not self._session.closed:
            await self._session.close()
            
            # Wait for proper cleanup
            await asyncio.sleep(0.1)
            
            logger.debug("HTTP session closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.close()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        if not self._session or self._session.closed:
            return {'status': 'closed'}
        
        connector = self._session.connector
        
        return {
            'status': 'active',
            'total_connections': len(connector._conns),
            'available_connections': sum(len(conns) for conns in connector._conns.values()),
            'limit': connector.limit,
            'limit_per_host': connector.limit_per_host
        }
    
    async def test_connection(self) -> bool:
        """Test HTTP session functionality.
        
        Returns:
            True if session is working correctly
        """
        try:
            session = await self.get_session()
            
            # Test with a simple HTTP request
            async with session.get('https://httpbin.org/get') as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Session test failed: {e}")
            return False


# Global session manager instance will be created when needed
_session_manager: Optional[SessionManager] = None
_session_lock = asyncio.Lock()


async def get_global_session_manager(config: ParallelProcessingConfig) -> SessionManager:
    """Get or create global session manager instance.
    
    Args:
        config: Parallel processing configuration
        
    Returns:
        Global SessionManager instance
    """
    global _session_manager
    
    if _session_manager is None:
        async with _session_lock:
            if _session_manager is None:
                _session_manager = SessionManager(config)
    
    return _session_manager


async def cleanup_global_session_manager() -> None:
    """Cleanup global session manager."""
    global _session_manager
    
    if _session_manager:
        await _session_manager.close()
        _session_manager = None