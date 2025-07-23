"""Parallel LLM processor with coordinated session management, rate limiting, and progress tracking."""
import asyncio
import time
import logging
import json
from typing import List, Dict, Any, Optional, Callable, Tuple, Awaitable
from dataclasses import dataclass
import aiohttp

from .session_manager import SessionManager, get_global_session_manager
from .rate_limiter import AdaptiveRateLimiter
from .progress_tracker import ProgressTracker
from ..config.models import ParallelProcessingConfig
from ..exceptions import (
    ProcessingError, 
    RateLimitExceededException, 
    CircuitBreakerOpenError,
    ConcurrencyLimitException,
    SessionPoolExhaustedException
)


logger = logging.getLogger(__name__)


@dataclass
class ProcessingItem:
    """Individual item to be processed."""
    id: str
    data: Dict[str, Any]
    estimated_tokens: int = 1000
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ProcessingResult:
    """Result of processing an individual item."""
    item_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: int = 0
    response_time: float = 0.0
    retry_count: int = 0


class ParallelLLMProcessor:
    """Coordinated parallel processor for LLM operations."""
    
    def __init__(self, config: ParallelProcessingConfig, processor_function: Callable):
        """Initialize parallel processor.
        
        Args:
            config: Parallel processing configuration
            processor_function: Async function to process individual items
                               Signature: async def process(session, item) -> ProcessingResult
        """
        self.config = config
        self.processor_function = processor_function
        
        # Core components
        self.session_manager: Optional[SessionManager] = None
        self.rate_limiter = AdaptiveRateLimiter(config)
        self.progress_tracker = ProgressTracker(config)
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        
        # Processing state
        self.active_tasks: List[asyncio.Task] = []
        self.total_items = 0
        self.completed_items = 0
        self.failed_items = 0
        self.skipped_items = 0
        
        # Processing control
        self._stop_processing = False
        self._processing_started = False
        
        logger.debug(f"Initialized ParallelLLMProcessor with max_concurrent={config.max_concurrent_requests}")
    
    async def process_items(self, items: List[ProcessingItem]) -> List[ProcessingResult]:
        """Process a list of items in parallel.
        
        Args:
            items: List of items to process
            
        Returns:
            List of processing results
            
        Raises:
            ProcessingError: If processing setup fails
        """
        if not items:
            logger.warning("No items to process")
            return []
        
        self.total_items = len(items)
        logger.info(f"Starting parallel processing of {self.total_items} items")
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start progress tracking
            await self.progress_tracker.start_tracking(
                self.total_items, 
                f"Processing {self.total_items} items"
            )
            
            # Process items
            results = await self._coordinate_processing(items)
            
            # Final metrics update
            await self._update_final_metrics()
            
            logger.info(f"Completed processing: {len(results)} results, "
                       f"{self.completed_items} successful, {self.failed_items} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise ProcessingError(f"Parallel processing failed: {e}")
        
        finally:
            await self._cleanup()
    
    async def _initialize_components(self) -> None:
        """Initialize all processing components."""
        try:
            # Initialize session manager
            if self.config.enable_session_pooling:
                self.session_manager = await get_global_session_manager(self.config)
            else:
                self.session_manager = SessionManager(self.config)
            
            logger.debug("Initialized processing components")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise ProcessingError(f"Component initialization failed: {e}")
    
    async def _coordinate_processing(self, items: List[ProcessingItem]) -> List[ProcessingResult]:
        """Coordinate the parallel processing of items.
        
        Args:
            items: Items to process
            
        Returns:
            List of processing results
        """
        # Add items to processing queue
        for item in items:
            await self.processing_queue.put(item)
        
        # Create worker tasks
        worker_tasks = []
        for i in range(self.config.max_concurrent_requests):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            worker_tasks.append(task)
            self.active_tasks.append(task)
        
        # Create result collector task
        collector_task = asyncio.create_task(self._result_collector())
        self.active_tasks.append(collector_task)
        
        # Wait for all items to be processed
        results = []
        try:
            # Wait for processing to complete
            while self.completed_items + self.failed_items + self.skipped_items < self.total_items:
                await asyncio.sleep(0.1)
                
                # Update progress tracker with queue status
                await self.progress_tracker.set_queue_size(self.processing_queue.qsize())
                await self.progress_tracker.set_active_tasks(len([t for t in worker_tasks if not t.done()]))
            
            # Signal workers to stop
            self._stop_processing = True
            
            # Add sentinel values to stop workers
            for _ in worker_tasks:
                await self.processing_queue.put(None)
            
            # Wait for workers to finish
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            
            # Collect final results
            collector_task.cancel()
            try:
                await collector_task
            except asyncio.CancelledError:
                pass
            
            # Collect remaining results
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    results.append(result)
                except asyncio.QueueEmpty:
                    break
            
            logger.debug(f"Collected {len(results)} final results")
            return results
            
        except Exception as e:
            logger.error(f"Processing coordination failed: {e}")
            self._stop_processing = True
            
            # Cancel all tasks
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
            
            raise
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Worker loop for processing items.
        
        Args:
            worker_id: Unique identifier for this worker
        """
        logger.debug(f"Worker {worker_id} started")
        
        while not self._stop_processing:
            try:
                # Get next item from queue
                item = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                
                # Sentinel to stop worker
                if item is None:
                    break
                
                # Process the item
                result = await self._process_single_item(item, worker_id)
                
                # Add result to result queue
                await self.result_queue.put(result)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.TimeoutError:
                # Continue checking for stop signal
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                continue
        
        logger.debug(f"Worker {worker_id} finished")
    
    async def _process_single_item(self, item: ProcessingItem, worker_id: str) -> ProcessingResult:
        """Process a single item with rate limiting and error handling.
        
        Args:
            item: Item to process
            worker_id: ID of the processing worker
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Acquire concurrency semaphore
            async with self.semaphore:
                # Apply rate limiting
                try:
                    await self.rate_limiter.acquire(item.estimated_tokens)
                except CircuitBreakerOpenError as e:
                    logger.warning(f"Circuit breaker open for item {item.id}: {e}")
                    await self._record_failure(item, str(e), 0.0)
                    return ProcessingResult(item.id, False, error=str(e))
                
                # Get session for processing
                if not self.session_manager:
                    raise SessionPoolExhaustedException("Session manager not initialized")
                
                session = await self.session_manager.get_session()
                
                # Process the item
                try:
                    result = await self.processor_function(session, item)
                    
                    # Calculate response time
                    response_time = time.time() - start_time
                    result.response_time = response_time
                    
                    # Update rate limiter with actual usage
                    self.rate_limiter.update_from_response(
                        headers={},  # Headers would come from actual API response
                        tokens_used=result.tokens_used or item.estimated_tokens,
                        success=result.success,
                        response_time=response_time
                    )
                    
                    # Update progress tracking
                    if result.success:
                        await self._record_success(item, result.tokens_used or 0, response_time)
                    else:
                        await self._record_failure(item, result.error or "Unknown error", response_time)
                    
                    logger.debug(f"Worker {worker_id} processed item {item.id}: success={result.success}")
                    return result
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    error_msg = f"Processing error: {e}"
                    logger.error(f"Worker {worker_id} failed to process item {item.id}: {error_msg}")
                    
                    # Update rate limiter with failure
                    self.rate_limiter.update_from_response(
                        headers={},
                        tokens_used=0,
                        success=False,
                        response_time=response_time
                    )
                    
                    await self._record_failure(item, error_msg, response_time)
                    
                    # Check if item should be retried
                    if item.retry_count < item.max_retries:
                        item.retry_count += 1
                        logger.debug(f"Retrying item {item.id} (attempt {item.retry_count})")
                        await asyncio.sleep(min(2 ** item.retry_count, 10))  # Exponential backoff
                        await self.processing_queue.put(item)
                        return ProcessingResult(item.id, False, error="Retrying", retry_count=item.retry_count)
                    
                    return ProcessingResult(item.id, False, error=error_msg, retry_count=item.retry_count)
        
        except ConcurrencyLimitException as e:
            logger.warning(f"Concurrency limit reached for item {item.id}: {e}")
            await self._record_failure(item, str(e), 0.0)
            return ProcessingResult(item.id, False, error=str(e))
        
        except Exception as e:
            logger.error(f"Unexpected error processing item {item.id}: {e}")
            await self._record_failure(item, str(e), 0.0)
            return ProcessingResult(item.id, False, error=str(e))
    
    async def _record_success(self, item: ProcessingItem, tokens_used: int, response_time: float) -> None:
        """Record successful processing."""
        self.completed_items += 1
        await self.progress_tracker.update_progress(
            completed=1,
            response_time=response_time,
            tokens=tokens_used
        )
    
    async def _record_failure(self, item: ProcessingItem, error: str, response_time: float) -> None:
        """Record failed processing."""
        if item.retry_count >= item.max_retries:
            self.failed_items += 1
            await self.progress_tracker.update_progress(
                failed=1,
                response_time=response_time,
                error=error
            )
    
    async def _record_skip(self, item: ProcessingItem, reason: str) -> None:
        """Record skipped processing."""
        self.skipped_items += 1
        await self.progress_tracker.update_progress(skipped=1)
    
    async def _result_collector(self) -> None:
        """Background task to collect and log processing results."""
        collected_count = 0
        
        while not self._stop_processing:
            try:
                # Check for results periodically
                await asyncio.sleep(1)
                
                temp_results = []
                while not self.result_queue.empty():
                    try:
                        result = self.result_queue.get_nowait()
                        temp_results.append(result)
                        collected_count += 1
                    except asyncio.QueueEmpty:
                        break
                
                if temp_results:
                    logger.debug(f"Result collector processed {len(temp_results)} results "
                               f"(total: {collected_count})")
                
            except Exception as e:
                logger.error(f"Result collector error: {e}")
                continue
    
    async def _update_final_metrics(self) -> None:
        """Update final processing metrics."""
        # Get rate limiter statistics
        rate_stats = self.rate_limiter.get_statistics()
        
        # Get session statistics
        session_stats = {}
        if self.session_manager:
            session_stats = self.session_manager.get_connection_stats()
        
        # Log final statistics
        logger.info(f"Final processing statistics:")
        logger.info(f"  Total items: {self.total_items}")
        logger.info(f"  Completed: {self.completed_items}")
        logger.info(f"  Failed: {self.failed_items}")
        logger.info(f"  Skipped: {self.skipped_items}")
        logger.info(f"  Success rate: {self.completed_items / self.total_items * 100:.1f}%")
        logger.info(f"  Rate limiter: {rate_stats}")
        logger.info(f"  Session pool: {session_stats}")
    
    async def _cleanup(self) -> None:
        """Cleanup all resources."""
        try:
            # Stop progress tracking
            await self.progress_tracker.stop_tracking()
            
            # Cancel any remaining tasks
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
            
            # Cleanup session manager if not global
            if self.session_manager and not self.config.enable_session_pooling:
                await self.session_manager.close()
            
            logger.debug("Parallel processor cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        rate_stats = self.rate_limiter.get_statistics()
        progress_stats = self.progress_tracker.get_summary()
        
        session_stats = {}
        if self.session_manager:
            session_stats = self.session_manager.get_connection_stats()
        
        return {
            'processing': {
                'total_items': self.total_items,
                'completed': self.completed_items,
                'failed': self.failed_items,
                'skipped': self.skipped_items,
                'success_rate': self.completed_items / self.total_items if self.total_items > 0 else 0,
                'active_tasks': len([t for t in self.active_tasks if not t.done()]),
                'queue_size': self.processing_queue.qsize() if hasattr(self, 'processing_queue') else 0
            },
            'rate_limiting': rate_stats,
            'progress': progress_stats,
            'session_pool': session_stats
        }
    
    async def test_parallel_processing(self, num_items: int = 10) -> List[ProcessingResult]:
        """Test parallel processing functionality.
        
        Args:
            num_items: Number of test items to process
            
        Returns:
            List of processing results
        """
        logger.info(f"Testing parallel processing with {num_items} items")
        
        # Create test processor function
        async def test_processor(session, item):
            # Simulate processing time
            await asyncio.sleep(0.1 + (int(item.id) % 3) * 0.1)
            
            # Simulate occasional failures
            if int(item.id) % 7 == 0:
                return ProcessingResult(item.id, False, error="Test error", tokens_used=0)
            
            return ProcessingResult(
                item.id, 
                True, 
                data={'processed': True, 'value': int(item.id) * 2},
                tokens_used=100 + (int(item.id) % 5) * 50
            )
        
        # Create test items
        test_items = [
            ProcessingItem(
                id=str(i),
                data={'test_data': f'item_{i}'},
                estimated_tokens=100 + (i % 5) * 50
            )
            for i in range(num_items)
        ]
        
        # Create test processor
        original_processor = self.processor_function
        self.processor_function = test_processor
        
        try:
            results = await self.process_items(test_items)
            logger.info(f"Test completed: {len(results)} results")
            return results
        finally:
            self.processor_function = original_processor