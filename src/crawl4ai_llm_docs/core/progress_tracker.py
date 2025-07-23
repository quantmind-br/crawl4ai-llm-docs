"""Real-time progress tracking with performance metrics and ETA calculation."""
import asyncio
import time
import logging
import psutil
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel

from ..config.models import ParallelProcessingConfig


logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Comprehensive metrics for parallel processing."""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    # Performance metrics
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    tokens_processed: int = 0
    tokens_per_second: float = 0.0
    
    # Error tracking
    error_rate: float = 0.0
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    
    # Resource utilization
    active_tasks: int = 0
    queue_size: int = 0
    memory_usage_mb: float = 0.0


class ProgressTracker:
    """Advanced progress tracking for parallel LLM operations."""
    
    def __init__(self, config: ParallelProcessingConfig, console: Optional[Console] = None):
        """Initialize progress tracker.
        
        Args:
            config: Parallel processing configuration
            console: Rich console instance (creates new one if None)
        """
        self.config = config
        self.console = console or Console()
        self.metrics = ProcessingMetrics()
        
        # Rich progress display
        self._progress: Optional[Progress] = None
        self._task_id: Optional[TaskID] = None
        
        # Tracking data
        self._lock = asyncio.Lock()
        self._response_times: List[float] = []
        self._token_history: List[Tuple[float, int]] = []  # (timestamp, tokens)
        self._error_history: List[Tuple[float, str]] = []  # (timestamp, error)
        
        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
        
        # Display configuration
        self.enable_detailed_logging = True
        self.show_memory_usage = True
    
    async def start_tracking(self, total_items: int, description: str = "Processing") -> None:
        """Initialize tracking for a processing session.
        
        Args:
            total_items: Total number of items to process
            description: Description for the progress bar
        """
        async with self._lock:
            self.metrics = ProcessingMetrics(total_items=total_items)
            self._stop_monitoring = False
            
            # Initialize Rich progress display
            self._progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TextColumn("{task.completed}/{task.total}"),
                "•",
                TextColumn("{task.fields[rate]:.1f} req/s"),
                "•",
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=2
            )
            
            self._task_id = self._progress.add_task(
                description,
                total=total_items,
                rate=0.0
            )
            
            # Start progress display
            self._progress.start()
            
            # Start monitoring task
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"Started tracking for {total_items} items: {description}")
    
    async def update_progress(self,
                            completed: int = 0,
                            failed: int = 0,
                            skipped: int = 0,
                            response_time: Optional[float] = None,
                            tokens: Optional[int] = None,
                            error: Optional[str] = None) -> None:
        """Update progress metrics.
        
        Args:
            completed: Number of completed items
            failed: Number of failed items
            skipped: Number of skipped items
            response_time: Response time for the request
            tokens: Number of tokens processed
            error: Error message if any
        """
        async with self._lock:
            current_time = time.time()
            
            # Update counters
            self.metrics.completed_items += completed
            self.metrics.failed_items += failed
            self.metrics.skipped_items += skipped
            self.metrics.last_update = current_time
            
            # Track response times
            if response_time is not None:
                self._response_times.append(response_time)
                # Keep only recent response times
                if len(self._response_times) > 100:
                    self._response_times = self._response_times[-100:]
            
            # Track token usage
            if tokens is not None:
                self.metrics.tokens_processed += tokens
                self._token_history.append((current_time, tokens))
                # Clean old token history (keep last minute)
                cutoff = current_time - 60
                self._token_history = [(t, c) for t, c in self._token_history if t > cutoff]
            
            # Track errors
            if error is not None:
                self.metrics.last_error = error
                self.metrics.consecutive_errors += 1
                self._error_history.append((current_time, error))
                # Keep recent errors
                if len(self._error_history) > 50:
                    self._error_history = self._error_history[-50:]
            else:
                self.metrics.consecutive_errors = 0
            
            # Update Rich progress
            if self._progress and self._task_id is not None:
                total_processed = (self.metrics.completed_items + 
                                 self.metrics.failed_items + 
                                 self.metrics.skipped_items)
                
                self._progress.update(
                    self._task_id,
                    completed=total_processed,
                    rate=self.metrics.requests_per_second
                )
    
    async def set_active_tasks(self, count: int) -> None:
        """Update active task count.
        
        Args:
            count: Number of active tasks
        """
        async with self._lock:
            self.metrics.active_tasks = count
    
    async def set_queue_size(self, size: int) -> None:
        """Update queue size.
        
        Args:
            size: Current queue size
        """
        async with self._lock:
            self.metrics.queue_size = size
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring and metrics calculation."""
        update_interval = self.config.progress_update_interval
        
        while not self._stop_monitoring:
            try:
                await self._calculate_metrics()
                await self._display_detailed_info()
                await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _calculate_metrics(self) -> None:
        """Calculate derived metrics."""
        async with self._lock:
            current_time = time.time()
            elapsed = current_time - self.metrics.start_time
            
            if elapsed > 0:
                total_processed = (self.metrics.completed_items + 
                                 self.metrics.failed_items + 
                                 self.metrics.skipped_items)
                
                # Calculate requests per second
                self.metrics.requests_per_second = total_processed / elapsed
                
                # Calculate average response time
                if self._response_times:
                    self.metrics.average_response_time = sum(self._response_times) / len(self._response_times)
                
                # Calculate tokens per second
                if self._token_history:
                    recent_tokens = sum(count for _, count in self._token_history)
                    time_window = min(60, elapsed)
                    self.metrics.tokens_per_second = recent_tokens / time_window
                
                # Calculate error rate
                if total_processed > 0:
                    self.metrics.error_rate = self.metrics.failed_items / total_processed
                
                # Update memory usage
                if self.show_memory_usage:
                    try:
                        process = psutil.Process()
                        self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                    except Exception:
                        pass  # Ignore memory monitoring errors
    
    async def _display_detailed_info(self) -> None:
        """Display detailed progress information."""
        if not self.enable_detailed_logging:
            return
        
        async with self._lock:
            # Only show detailed info periodically
            if time.time() - self.metrics.last_update < self.config.progress_update_interval:
                return
            
            # Calculate progress percentage
            total_processed = (self.metrics.completed_items + 
                             self.metrics.failed_items + 
                             self.metrics.skipped_items)
            progress_pct = (total_processed / self.metrics.total_items * 100 
                          if self.metrics.total_items > 0 else 0)
            
            # Calculate ETA
            eta_str = "Unknown"
            if self.metrics.requests_per_second > 0:
                remaining = self.metrics.total_items - total_processed
                eta_seconds = remaining / self.metrics.requests_per_second
                eta_str = str(timedelta(seconds=int(eta_seconds)))
            
            # Create summary table
            table = Table(title="Processing Statistics", show_header=False)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Progress", f"{progress_pct:.1f}% ({total_processed}/{self.metrics.total_items})")
            table.add_row("Rate", f"{self.metrics.requests_per_second:.2f} req/s")
            table.add_row("Avg Response Time", f"{self.metrics.average_response_time:.2f}s")
            table.add_row("Tokens/sec", f"{self.metrics.tokens_per_second:.0f}")
            table.add_row("Error Rate", f"{self.metrics.error_rate:.2%}")
            table.add_row("Active Tasks", str(self.metrics.active_tasks))
            
            if self.show_memory_usage:
                table.add_row("Memory Usage", f"{self.metrics.memory_usage_mb:.1f} MB")
            
            table.add_row("ETA", eta_str)
            
            # Display table (only if console is not showing progress bar)
            if not self._progress:
                self.console.print(table)
    
    async def stop_tracking(self) -> ProcessingMetrics:
        """Stop tracking and return final metrics.
        
        Returns:
            Final processing metrics
        """
        self._stop_monitoring = True
        
        # Stop monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop progress display
        if self._progress:
            self._progress.stop()
            self._progress = None
        
        # Calculate final metrics
        await self._calculate_metrics()
        
        # Display final summary
        await self._display_final_summary()
        
        logger.info(f"Tracking completed. Final metrics: {self.get_summary()}")
        return self.metrics
    
    async def _display_final_summary(self) -> None:
        """Display final processing summary."""
        elapsed = time.time() - self.metrics.start_time
        
        # Create final summary
        summary_text = f"""
[bold green]Processing Complete![/bold green]

[cyan]Results:[/cyan]
• Total: {self.metrics.total_items}
• Completed: {self.metrics.completed_items}
• Failed: {self.metrics.failed_items}
• Skipped: {self.metrics.skipped_items}
• Success Rate: {(self.metrics.completed_items / self.metrics.total_items * 100):.1f}%

[cyan]Performance:[/cyan]
• Processing Time: {timedelta(seconds=int(elapsed))}
• Average Rate: {self.metrics.requests_per_second:.2f} req/s
• Total Tokens: {self.metrics.tokens_processed:,}
• Tokens/sec: {self.metrics.tokens_per_second:.0f}
"""
        
        panel = Panel(summary_text, title="Processing Summary", border_style="green")
        self.console.print(panel, markup=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics.
        
        Returns:
            Dictionary with current metrics
        """
        return {
            "total_items": self.metrics.total_items,
            "completed": self.metrics.completed_items,
            "failed": self.metrics.failed_items,
            "skipped": self.metrics.skipped_items,
            "success_rate": (self.metrics.completed_items / self.metrics.total_items 
                           if self.metrics.total_items > 0 else 0),
            "requests_per_second": self.metrics.requests_per_second,
            "average_response_time": self.metrics.average_response_time,
            "tokens_processed": self.metrics.tokens_processed,
            "tokens_per_second": self.metrics.tokens_per_second,
            "error_rate": self.metrics.error_rate,
            "consecutive_errors": self.metrics.consecutive_errors,
            "elapsed_time": time.time() - self.metrics.start_time,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "active_tasks": self.metrics.active_tasks,
            "queue_size": self.metrics.queue_size
        }
    
    async def test_tracking(self, total_items: int) -> None:
        """Test progress tracking functionality.
        
        Args:
            total_items: Number of items to simulate
        """
        await self.start_tracking(total_items, "Testing Progress Tracker")
        
        try:
            for i in range(total_items):
                # Simulate processing
                await asyncio.sleep(0.1)
                
                # Simulate different outcomes
                if i % 10 == 9:
                    await self.update_progress(failed=1, error="Test error")
                else:
                    await self.update_progress(
                        completed=1,
                        response_time=0.5 + (i % 3) * 0.2,
                        tokens=100 + (i % 5) * 50
                    )
                
                await self.set_active_tasks(min(4, total_items - i))
                await self.set_queue_size(max(0, total_items - i - 1))
        
        finally:
            await self.stop_tracking()