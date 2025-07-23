"""
Pipeline Coordinator - Orchestrates parallel scraping and processing operations.
Manages the complete parallel processing pipeline with adaptive coordination.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from rich.console import Console

from .scraper import DocumentationScraper, ScrapedDocument
from .adaptive_rate_limiter import AdaptiveRateLimiter
from .intelligent_chunker import IntelligentChunker, DocumentChunk
from .progress_tracker import ProgressTracker
from ..config.models import AppConfig
from ..exceptions import (
    ConcurrencyLimitException, 
    SessionPoolExhaustedException,
    CircuitBreakerOpenError
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineTask:
    """Represents a task in the processing pipeline."""
    task_id: str
    urls: List[str]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass 
class PipelineMetrics:
    """Comprehensive pipeline performance metrics."""
    scraping_time: float = 0.0
    processing_time: float = 0.0
    total_urls: int = 0
    successful_scrapes: int = 0
    failed_scrapes: int = 0
    successful_processes: int = 0
    failed_processes: int = 0
    concurrent_peak: int = 0
    queue_peak: int = 0
    rate_limit_hits: int = 0
    efficiency_ratio: float = 0.0  # actual_time / theoretical_optimal_time


class PipelineCoordinator:
    """Advanced pipeline coordinator for parallel scraping and processing."""
    
    def __init__(self, config: AppConfig, console: Optional[Console] = None):
        """Initialize pipeline coordinator.
        
        Args:
            config: Application configuration
            console: Rich console for output
        """
        self.config = config
        self.console = console or Console()
        
        # Import here to avoid circular import
        from ..config.models import CrawlerConfig
        
        # Core components
        self.scraper = DocumentationScraper(CrawlerConfig())
        self.rate_limiter = AdaptiveRateLimiter(config.parallel_processing)
        self.chunker = IntelligentChunker(config)
        self.progress_tracker = ProgressTracker(config.parallel_processing, console)
        
        # Pipeline coordination
        self._scraping_semaphore = asyncio.Semaphore(config.max_workers)
        self._processing_semaphore = asyncio.Semaphore(config.parallel_processing.max_concurrent_requests)
        
        # Task management
        self._scraping_queue: asyncio.Queue = asyncio.Queue()
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._result_queue: asyncio.Queue = asyncio.Queue()
        
        # State tracking
        self._active_scraping_tasks = 0
        self._active_processing_tasks = 0
        self._shutdown_event = asyncio.Event()
        self._metrics = PipelineMetrics()
        
        # Worker pools
        self._scraping_workers: List[asyncio.Task] = []
        self._processing_workers: List[asyncio.Task] = []
        
        logger.info(f"PipelineCoordinator initialized with {config.max_workers} scrapers, "
                   f"{config.parallel_processing.max_concurrent_requests} processors")
    
    async def process_urls(self, 
                          urls: List[str],
                          processor_func: Callable[[List[DocumentChunk]], Any],
                          max_retries: int = 2) -> Dict[str, Any]:
        """Process a list of URLs through the complete pipeline.
        
        Args:
            urls: List of URLs to process
            processor_func: Function to process document chunks
            max_retries: Maximum retry attempts for failed operations
            
        Returns:
            Dictionary with processing results and metrics
        """
        if not urls:
            return {"error": "No URLs provided", "results": [], "metrics": {}}
        
        logger.info(f"Starting pipeline processing for {len(urls)} URLs")
        self._metrics = PipelineMetrics(total_urls=len(urls))
        
        try:
            # Start progress tracking
            await self.progress_tracker.start_tracking(len(urls), "Pipeline Processing")
            
            # Initialize pipeline workers
            await self._start_workers()
            
            # Queue all URLs for scraping
            for i, url in enumerate(urls):
                task = PipelineTask(
                    task_id=f"url_{i}",
                    urls=[url],
                    priority=0
                )
                await self._scraping_queue.put(task)
            
            # Start coordination and monitoring
            coordinator_task = asyncio.create_task(self._coordinate_pipeline())
            results_collector = asyncio.create_task(self._collect_results(processor_func))
            
            # Wait for all URLs to be processed
            await self._wait_for_completion(len(urls))
            
            # Stop workers and collect final results
            await self._shutdown_workers()
            coordinator_task.cancel()
            
            # Get final results
            try:
                await coordinator_task
            except asyncio.CancelledError:
                pass
                
            final_results = await results_collector
            
            # Stop progress tracking
            final_metrics = await self.progress_tracker.stop_tracking()
            
            # Calculate efficiency metrics
            self._calculate_efficiency_metrics()
            
            return {
                "results": final_results,
                "metrics": self._get_metrics_summary(),
                "progress_metrics": final_metrics.get_summary() if hasattr(final_metrics, 'get_summary') else {}
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            await self._emergency_shutdown()
            raise
    
    async def _start_workers(self) -> None:
        """Start scraping and processing worker pools."""
        # Start scraping workers
        for i in range(self.config.max_workers):
            worker = asyncio.create_task(self._scraping_worker(f"scraper_{i}"))
            self._scraping_workers.append(worker)
        
        # Start processing workers 
        for i in range(self.config.parallel_processing.max_concurrent_requests):
            worker = asyncio.create_task(self._processing_worker(f"processor_{i}"))
            self._processing_workers.append(worker)
        
        logger.info(f"Started {len(self._scraping_workers)} scraping workers and "
                   f"{len(self._processing_workers)} processing workers")
    
    async def _scraping_worker(self, worker_id: str) -> None:
        """Worker for scraping operations."""
        logger.debug(f"Scraping worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get next scraping task
                try:
                    task = await asyncio.wait_for(
                        self._scraping_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process scraping task
                async with self._scraping_semaphore:
                    self._active_scraping_tasks += 1
                    await self.progress_tracker.set_active_tasks(
                        self._active_scraping_tasks + self._active_processing_tasks
                    )
                    
                    try:
                        start_time = time.time()
                        task.started_at = start_time
                        
                        # Perform scraping
                        scraped_docs = await self.scraper.scrape_urls_async(task.urls)
                        
                        scraping_time = time.time() - start_time
                        self._metrics.scraping_time += scraping_time
                        
                        # Update metrics
                        successful = sum(1 for doc in scraped_docs if doc.success)
                        failed = len(scraped_docs) - successful
                        
                        self._metrics.successful_scrapes += successful
                        self._metrics.failed_scrapes += failed
                        
                        # Queue for processing if successful
                        if successful > 0:
                            processing_task = PipelineTask(
                                task_id=f"process_{task.task_id}",
                                urls=task.urls,
                                result=scraped_docs,
                                created_at=task.created_at,
                                started_at=start_time
                            )
                            await self._processing_queue.put(processing_task)
                        
                        # Update progress
                        await self.progress_tracker.update_progress(
                            completed=successful,
                            failed=failed,
                            response_time=scraping_time
                        )
                        
                        task.completed_at = time.time()
                        task.result = scraped_docs
                        
                    except Exception as e:
                        logger.error(f"Scraping failed for {task.urls}: {e}")
                        task.error = str(e)
                        self._metrics.failed_scrapes += len(task.urls)
                        
                        await self.progress_tracker.update_progress(
                            failed=len(task.urls),
                            error=str(e)
                        )
                        
                        # Retry logic
                        if task.retry_count < 2:  # Max 2 retries
                            task.retry_count += 1
                            await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                            await self._scraping_queue.put(task)
                    
                    finally:
                        self._active_scraping_tasks -= 1
                        self._scraping_queue.task_done()
                        await self.progress_tracker.set_active_tasks(
                            self._active_scraping_tasks + self._active_processing_tasks
                        )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scraping worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.debug(f"Scraping worker {worker_id} stopped")
    
    async def _processing_worker(self, worker_id: str) -> None:
        """Worker for LLM processing operations."""
        logger.debug(f"Processing worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get next processing task
                try:
                    task = await asyncio.wait_for(
                        self._processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process LLM task with rate limiting
                async with self._processing_semaphore:
                    self._active_processing_tasks += 1
                    await self.progress_tracker.set_active_tasks(
                        self._active_scraping_tasks + self._active_processing_tasks
                    )
                    
                    try:
                        # Apply rate limiting
                        await self.rate_limiter.acquire()
                        
                        start_time = time.time()
                        
                        # Create chunks from scraped documents
                        scraped_docs = task.result
                        if not scraped_docs:
                            continue
                        
                        chunks = self.chunker.create_chunks(scraped_docs)
                        if not chunks:
                            continue
                        
                        # Processing will be handled by the processor function
                        task.result = chunks
                        task.completed_at = time.time()
                        
                        processing_time = time.time() - start_time
                        self._metrics.processing_time += processing_time
                        self._metrics.successful_processes += 1
                        
                        # Update rate limiter on success
                        self.rate_limiter.on_response({}, processing_time)
                        
                        # Queue for final collection
                        await self._result_queue.put(task)
                        
                    except Exception as e:
                        logger.error(f"Processing failed for task {task.task_id}: {e}")
                        task.error = str(e)
                        self._metrics.failed_processes += 1
                        
                        # Update rate limiter on error
                        self.rate_limiter.on_error(e, "processing")
                        
                        await self.progress_tracker.update_progress(error=str(e))
                        
                        # Retry logic for processing
                        if task.retry_count < 1:  # Less retries for processing
                            task.retry_count += 1
                            await asyncio.sleep(5)  # Fixed delay for processing retries
                            await self._processing_queue.put(task)
                    
                    finally:
                        self._active_processing_tasks -= 1
                        self._processing_queue.task_done()
                        await self.progress_tracker.set_active_tasks(
                            self._active_scraping_tasks + self._active_processing_tasks
                        )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.debug(f"Processing worker {worker_id} stopped")
    
    async def _coordinate_pipeline(self) -> None:
        """Coordinate pipeline operations and resource management."""
        while not self._shutdown_event.is_set():
            try:
                # Update queue sizes
                await self.progress_tracker.set_queue_size(
                    self._scraping_queue.qsize() + self._processing_queue.qsize()
                )
                
                # Track peak utilization
                current_concurrent = self._active_scraping_tasks + self._active_processing_tasks
                self._metrics.concurrent_peak = max(self._metrics.concurrent_peak, current_concurrent)
                
                current_queue = self._scraping_queue.qsize() + self._processing_queue.qsize()
                self._metrics.queue_peak = max(self._metrics.queue_peak, current_queue)
                
                # Adaptive throttling based on error rates
                rate_limit_status = self.rate_limiter.get_rate_limit_status()
                if rate_limit_status["circuit_breaker_open"]:
                    logger.warning("Circuit breaker open - pausing pipeline")
                    await asyncio.sleep(5)
                    self._metrics.rate_limit_hits += 1
                
                await asyncio.sleep(2)  # Coordination update interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pipeline coordination error: {e}")
                await asyncio.sleep(1)
    
    async def _collect_results(self, processor_func: Callable[[List[DocumentChunk]], Any]) -> List[Any]:
        """Collect and process final results."""
        results = []
        processed_chunks = []
        
        while not self._shutdown_event.is_set():
            try:
                # Get completed processing task
                try:
                    task = await asyncio.wait_for(
                        self._result_queue.get(),
                        timeout=1.0  
                    )
                except asyncio.TimeoutError:
                    continue
                
                if task.result and isinstance(task.result, list):
                    processed_chunks.extend(task.result)
                    
                    # Process chunks when we have a good batch
                    if len(processed_chunks) >= 3:  # Process in batches
                        try:
                            batch_result = await asyncio.get_event_loop().run_in_executor(
                                None, processor_func, processed_chunks
                            )
                            results.append(batch_result)
                            processed_chunks = []  # Clear batch
                        except Exception as e:
                            logger.error(f"Batch processing failed: {e}")
                
                self._result_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Result collection error: {e}")
                await asyncio.sleep(1)
        
        # Process any remaining chunks
        if processed_chunks:
            try:
                final_result = await asyncio.get_event_loop().run_in_executor(
                    None, processor_func, processed_chunks
                )
                results.append(final_result)
            except Exception as e:
                logger.error(f"Final batch processing failed: {e}")
        
        return results
    
    async def _wait_for_completion(self, total_urls: int) -> None:
        """Wait for all tasks to complete."""
        timeout_per_url = 60  # 1 minute per URL max
        total_timeout = total_urls * timeout_per_url
        
        start_time = time.time()
        
        while time.time() - start_time < total_timeout:
            # Check if all queues are empty and no active tasks
            if (self._scraping_queue.empty() and 
                self._processing_queue.empty() and
                self._active_scraping_tasks == 0 and
                self._active_processing_tasks == 0):
                logger.info("All tasks completed successfully")
                return
            
            await asyncio.sleep(1)
        
        logger.warning(f"Pipeline timeout after {total_timeout}s")
    
    async def _shutdown_workers(self) -> None:
        """Gracefully shutdown all workers."""
        logger.info("Shutting down pipeline workers...")
        self._shutdown_event.set()
        
        # Cancel all workers
        all_workers = self._scraping_workers + self._processing_workers
        for worker in all_workers:
            worker.cancel()
        
        # Wait for workers to finish
        if all_workers:
            await asyncio.gather(*all_workers, return_exceptions=True)
        
        logger.info("All workers shut down")
    
    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown in case of critical errors."""
        logger.error("Emergency pipeline shutdown initiated")
        await self._shutdown_workers()
        await self.progress_tracker.stop_tracking()
    
    def _calculate_efficiency_metrics(self) -> None:
        """Calculate pipeline efficiency metrics."""
        total_time = self._metrics.scraping_time + self._metrics.processing_time
        
        if total_time > 0 and self._metrics.total_urls > 0:
            # Theoretical optimal: if everything was sequential
            theoretical_time = self._metrics.total_urls * 10  # Assume 10s per URL sequentially
            self._metrics.efficiency_ratio = theoretical_time / total_time
        
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "scraping_time": self._metrics.scraping_time,
            "processing_time": self._metrics.processing_time,
            "total_urls": self._metrics.total_urls,
            "successful_scrapes": self._metrics.successful_scrapes,
            "failed_scrapes": self._metrics.failed_scrapes,
            "successful_processes": self._metrics.successful_processes,
            "failed_processes": self._metrics.failed_processes,
            "concurrent_peak": self._metrics.concurrent_peak,
            "queue_peak": self._metrics.queue_peak,
            "rate_limit_hits": self._metrics.rate_limit_hits,
            "efficiency_ratio": self._metrics.efficiency_ratio,
            "scraping_success_rate": (
                self._metrics.successful_scrapes / max(1, self._metrics.total_urls)
            ),
            "processing_success_rate": (
                self._metrics.successful_processes / 
                max(1, self._metrics.successful_processes + self._metrics.failed_processes)
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Get pipeline health status."""
        rate_limit_status = self.rate_limiter.get_rate_limit_status()
        
        return {
            "pipeline_active": not self._shutdown_event.is_set(),
            "active_scraping_tasks": self._active_scraping_tasks,
            "active_processing_tasks": self._active_processing_tasks,
            "scraping_queue_size": self._scraping_queue.qsize(),
            "processing_queue_size": self._processing_queue.qsize(),
            "result_queue_size": self._result_queue.qsize(),
            "rate_limiter": rate_limit_status,
            "workers": {
                "scraping_workers": len(self._scraping_workers),
                "processing_workers": len(self._processing_workers)
            }
        }