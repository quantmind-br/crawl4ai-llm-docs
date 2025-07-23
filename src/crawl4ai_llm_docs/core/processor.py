"""Documentation processor using OpenAI-compatible APIs with optimized parallel architecture."""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import tiktoken

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError
except ImportError:
    logging.warning("OpenAI library not installed. Please install with: pip install openai>=1.0.0")
    OpenAI = None
    APIError = None
    RateLimitError = None
    APIConnectionError = None

from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .scraper import ScrapedDocument
from .pipeline_coordinator import PipelineCoordinator
from .content_preservation_processor import ContentPreservationProcessor
from .intelligent_chunker import IntelligentChunker
from .adaptive_rate_limiter import AdaptiveRateLimiter
from .progress_tracker import ProgressTracker
from ..config.models import AppConfig
from ..exceptions import ProcessingError, APIError as CustomAPIError


logger = logging.getLogger(__name__)


class DocumentationProcessor:
    """Advanced documentation processor with optimized parallel architecture."""
    
    def __init__(self, config: AppConfig, console: Optional[Console] = None, enable_parallel: bool = None):
        """Initialize processor with optimized parallel architecture.
        
        Args:
            config: Application configuration
            console: Rich console for output (creates new if None)
            enable_parallel: Enable parallel processing (auto-detect if None)
        """
        self.config = config
        self.console = console or Console()
        self.client = self._create_client()
        self._validate_dependencies()
        
        # Determine if parallel processing should be enabled
        self.enable_parallel = self._should_enable_parallel(enable_parallel)
        
        # Initialize optimized parallel architecture components
        self.rate_limiter = AdaptiveRateLimiter(config.parallel_processing)
        self.chunker = IntelligentChunker(config)
        self.content_processor = ContentPreservationProcessor(config, self.rate_limiter)
        self.pipeline_coordinator: Optional[PipelineCoordinator] = None
        
        if self.enable_parallel:
            self.pipeline_coordinator = PipelineCoordinator(config, self.console)
            
        logger.info(f"DocumentationProcessor initialized with optimized architecture - "
                   f"parallel_processing={self.enable_parallel}")
    
    def _validate_dependencies(self) -> None:
        """Validate that OpenAI dependencies are available."""
        if OpenAI is None:
            raise ImportError(
                "OpenAI library is not installed. Please install with: pip install openai>=1.0.0"
            )
    
    def _should_enable_parallel(self, enable_parallel: Optional[bool]) -> bool:
        """Determine if parallel processing should be enabled.
        
        Args:
            enable_parallel: Explicit setting (None for auto-detect)
            
        Returns:
            True if parallel processing should be enabled
        """
        if enable_parallel is not None:
            return enable_parallel
        
        # Auto-detect based on configuration and document count
        parallel_config = self.config.parallel_processing
        
        # Enable if adaptive rate limiting is enabled and we have concurrent capacity
        return (parallel_config.max_concurrent_requests > 1 and 
                parallel_config.enable_adaptive_rate_limiting)
    
    def _create_client(self) -> 'OpenAI':
        """Create OpenAI client with custom configuration.
        
        Returns:
            OpenAI client instance
        """
        if OpenAI is None:
            raise ImportError("OpenAI library not available")
        
        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=30.0,
            max_retries=3
        )
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text for the specified model.
        
        Args:
            text: Text to count tokens for
            model: Model name (uses config model if None)
            
        Returns:
            Number of tokens
        """
        try:
            model_name = model or self.config.model
            
            # Map model names to tiktoken encodings
            encoding_map = {
                'gpt-4o': 'o200k_base',
                'gpt-4o-mini': 'o200k_base',
                'gpt-4': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base',
                'gemini-2.5-flash': 'cl100k_base',  # Fallback for custom models
            }
            
            encoding_name = encoding_map.get(model_name, 'cl100k_base')
            enc = tiktoken.get_encoding(encoding_name)
            
            return len(enc.encode(text))
            
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using character approximation.")
            # Fallback: approximate tokens as chars/4
            return len(text) // 4
    
    
    
    
    def consolidate_documentation(self, documents: List[ScrapedDocument]) -> str:
        """Consolidate scraped documentation using optimized parallel architecture.
        
        Args:
            documents: List of scraped documents
            
        Returns:
            Consolidated markdown content with preserved technical information
            
        Raises:
            ProcessingError: If processing fails
        """
        if not documents:
            raise ProcessingError("No documents provided for consolidation")
        
        # Filter successful documents
        successful_docs = [doc for doc in documents if doc.success and doc.markdown]
        
        if not successful_docs:
            raise ProcessingError("No valid documents found for consolidation")
        
        logger.info(f"Starting optimized consolidation of {len(successful_docs)} documents "
                   f"(parallel={self.enable_parallel})")
        
        try:
            # Get initial content statistics
            total_content = "\n".join(doc.markdown for doc in successful_docs)
            total_tokens = self.count_tokens(total_content)
            
            logger.info(f"Total content: {total_tokens} tokens from {len(successful_docs)} documents")
            
            if len(successful_docs) == 1:
                # Single document - use content preservation processor directly
                logger.info("Processing single document with content preservation")
                consolidated_content = asyncio.run(self._process_single_document(successful_docs[0]))
                
            elif self.enable_parallel and len(successful_docs) > 1:
                # Use optimized parallel pipeline
                consolidated_content = asyncio.run(self._parallel_consolidate_optimized(successful_docs))
                
            else:
                # Use sequential processing with intelligent chunking
                consolidated_content = asyncio.run(self._sequential_consolidate_optimized(successful_docs))
            
            # Validate output quality
            if not consolidated_content or len(consolidated_content.strip()) < 100:
                raise ProcessingError("Generated content is too short or empty")
            
            # Check content preservation ratio
            output_tokens = self.count_tokens(consolidated_content)
            preservation_ratio = output_tokens / max(1, total_tokens)
            
            logger.info(f"Consolidation complete. Generated {len(consolidated_content)} characters "
                       f"({output_tokens} tokens), preservation ratio: {preservation_ratio:.2%}")
            
            # Warn if preservation ratio is too low
            if preservation_ratio < 0.7:
                logger.warning(f"Low content preservation ratio: {preservation_ratio:.2%}. "
                              "Consider reviewing content preservation settings.")
            
            return consolidated_content
            
        except (CustomAPIError, ProcessingError):
            raise
        except Exception as e:
            error_msg = f"Optimized consolidation failed: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
    
    async def _process_single_document(self, document: ScrapedDocument) -> str:
        """Process a single document with content preservation.
        
        Args:
            document: Single document to process
            
        Returns:
            Processed content
        """
        # Create chunk for single document
        chunk = self.chunker.create_chunks([document])
        
        if not chunk:
            logger.warning("No chunks created from single document")
            return document.markdown
        
        # Process with content preservation
        result = await self.content_processor.process_chunks_async(chunk)
        return result.cleaned_content
    
    async def _parallel_consolidate_optimized(self, documents: List[ScrapedDocument]) -> str:
        """Optimized parallel consolidation using the new pipeline architecture.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Consolidated content
        """
        logger.info(f"Using optimized parallel pipeline for {len(documents)} documents")
        
        if not self.pipeline_coordinator:
            raise ProcessingError("Pipeline coordinator not initialized")
        
        # Create URL list for pipeline processing
        urls = [doc.url for doc in documents]
        
        # Define processor function for chunks
        def process_chunks(chunks):
            """Process document chunks and return consolidated content."""
            try:
                # Run content preservation processing
                result = asyncio.run(self.content_processor.process_chunks_async(chunks))
                return result.cleaned_content
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                # Fall back to concatenating original content
                return "\n\n---\n\n".join(
                    doc.markdown for chunk in chunks for doc in chunk.documents
                )
        
        # Use pipeline coordinator for parallel processing with scraping and processing
        pipeline_result = await self.pipeline_coordinator.process_urls(
            urls, 
            process_chunks,
            max_retries=2
        )
        
        if not pipeline_result.get("results"):
            raise ProcessingError("Pipeline processing produced no results")
        
        # Combine all processing results
        final_content = "\n\n---\n\n".join(
            str(result) for result in pipeline_result["results"] if result
        )
        
        # Log pipeline metrics
        metrics = pipeline_result.get("metrics", {})
        logger.info(f"Pipeline metrics: efficiency_ratio={metrics.get('efficiency_ratio', 0):.2f}, "
                   f"scraping_time={metrics.get('scraping_time', 0):.2f}s, "
                   f"processing_time={metrics.get('processing_time', 0):.2f}s")
        
        return final_content
    
    async def _sequential_consolidate_optimized(self, documents: List[ScrapedDocument]) -> str:
        """Optimized sequential processing using intelligent chunking.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Consolidated content
        """
        logger.info(f"Using optimized sequential processing for {len(documents)} documents")
        
        # Create intelligent chunks
        chunks = self.chunker.create_chunks(documents)
        
        if not chunks:
            logger.warning("No chunks created from documents")
            return "\n\n---\n\n".join(doc.markdown for doc in documents)
        
        # Log chunking statistics
        chunking_stats = self.chunker.get_chunking_stats(chunks)
        logger.info(f"Chunking stats: {chunking_stats['total_chunks']} chunks, "
                   f"efficiency_gain: {chunking_stats['efficiency_gain']:.2f}x, "
                   f"token_utilization: {chunking_stats['token_utilization']:.2%}")
        
        # Process chunks with content preservation
        result = await self.content_processor.process_chunks_async(chunks)
        
        return result.cleaned_content

    
    def get_processing_stats(self, documents: List[ScrapedDocument]) -> Dict[str, Any]:
        """Get comprehensive statistics about processing input and architecture.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with comprehensive statistics
        """
        successful_docs = [doc for doc in documents if doc.success and doc.markdown]
        
        total_content = "\n".join(doc.markdown for doc in successful_docs)
        total_tokens = self.count_tokens(total_content)
        total_chars = len(total_content)
        
        # Get intelligent chunking preview
        chunking_preview = {}
        if successful_docs:
            chunks = self.chunker.create_chunks(successful_docs)
            chunking_preview = self.chunker.get_chunking_stats(chunks)
        
        # Get rate limiter status
        rate_limiter_status = self.rate_limiter.get_rate_limit_status()
        
        # Get content processor statistics
        processor_stats = self.content_processor.get_processing_statistics()
        
        stats = {
            "input_analysis": {
                "total_documents": len(documents),
                "processable_documents": len(successful_docs),
                "total_characters": total_chars,
                "estimated_tokens": total_tokens,
                "average_tokens_per_doc": total_tokens / len(successful_docs) if successful_docs else 0
            },
            "architecture_config": {
                "model": self.config.model,
                "max_tokens_per_request": self.config.max_tokens,
                "parallel_processing_enabled": self.enable_parallel,
                "max_concurrent_requests": self.config.parallel_processing.max_concurrent_requests,
                "adaptive_rate_limiting": self.config.parallel_processing.enable_adaptive_rate_limiting,
                "chunk_target_tokens": self.config.chunk_target_tokens,
                "chunk_max_tokens": self.config.chunk_max_tokens
            },
            "intelligent_chunking": chunking_preview,
            "rate_limiter_status": rate_limiter_status,
            "content_processor_stats": processor_stats,
            "estimated_performance": {
                "expected_chunks": chunking_preview.get("total_chunks", 1),
                "efficiency_gain": chunking_preview.get("efficiency_gain", 1.0),
                "estimated_cost_reduction": chunking_preview.get("estimated_cost_reduction", 0.0),
                "sequential_vs_parallel_speedup": self.config.parallel_processing.max_concurrent_requests if self.enable_parallel else 1
            }
        }
        
        return stats
    
    def get_parallel_processing_stats(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive parallel architecture statistics.
        
        Returns:
            Dictionary with parallel architecture statistics or None
        """
        if not self.enable_parallel:
            return None
        
        stats = {}
        
        # Pipeline coordinator statistics
        if self.pipeline_coordinator:
            stats["pipeline_coordinator"] = asyncio.run(
                self.pipeline_coordinator.health_check()
            )
        
        # Rate limiter statistics
        stats["rate_limiter"] = self.rate_limiter.get_rate_limit_status()
        
        # Content processor statistics
        stats["content_processor"] = self.content_processor.get_processing_statistics()
        
        # Architecture configuration
        stats["architecture_config"] = {
            "max_concurrent_requests": self.config.parallel_processing.max_concurrent_requests,
            "adaptive_rate_limiting": self.config.parallel_processing.enable_adaptive_rate_limiting,
            "session_pooling": self.config.parallel_processing.enable_session_pooling,
            "circuit_breaker": self.config.parallel_processing.enable_circuit_breaker,
            "chunk_target_tokens": self.config.chunk_target_tokens,
            "chunk_max_tokens": self.config.chunk_max_tokens
        }
        
        return stats
    
    async def test_parallel_processing(self, num_test_urls: int = 5) -> Dict[str, Any]:
        """Test the optimized parallel processing architecture.
        
        Args:
            num_test_urls: Number of test URLs to process
            
        Returns:
            Test results and comprehensive statistics
        """
        if not self.enable_parallel:
            return {
                'status': 'skipped',
                'reason': 'Parallel processing not enabled'
            }
        
        logger.info(f"Testing optimized parallel architecture with {num_test_urls} URLs")
        
        try:
            # Create test URLs (using httpbin for testing)
            test_urls = [
                f"https://httpbin.org/json?test={i}" 
                for i in range(num_test_urls)
            ]
            
            start_time = time.time()
            
            # Test the complete pipeline
            if self.pipeline_coordinator:
                def dummy_processor(chunks):
                    return f"Processed {len(chunks)} chunks successfully"
                
                result = await self.pipeline_coordinator.process_urls(
                    test_urls,
                    dummy_processor,
                    max_retries=1
                )
                
                processing_time = time.time() - start_time
                
                # Get comprehensive statistics
                architecture_stats = self.get_parallel_processing_stats()
                
                return {
                    'status': 'completed',
                    'test_urls': num_test_urls,
                    'processing_time': processing_time,
                    'pipeline_result': result,
                    'architecture_statistics': architecture_stats,
                    'performance_metrics': {
                        'urls_per_second': num_test_urls / processing_time if processing_time > 0 else 0,
                        'efficiency_ratio': result.get('metrics', {}).get('efficiency_ratio', 0),
                        'concurrent_peak': result.get('metrics', {}).get('concurrent_peak', 0)
                    }
                }
            else:
                # Test individual components
                test_results = {}
                
                # Test rate limiter
                for i in range(3):
                    await self.rate_limiter.acquire()
                    await asyncio.sleep(0.1)
                test_results['rate_limiter'] = 'working'
                
                # Test content processor
                test_content = "# Test Document\n\nThis is a test of the content preservation processor."
                from .scraper import ScrapedDocument
                test_doc = ScrapedDocument(
                    url="https://test.example.com",
                    title="Test Document",
                    markdown=test_content,
                    success=True
                )
                
                content_test = await self.content_processor.test_content_preservation(test_content)
                test_results['content_processor'] = content_test
                
                return {
                    'status': 'completed',
                    'component_tests': test_results,
                    'processing_time': time.time() - start_time,
                    'architecture_statistics': self.get_parallel_processing_stats()
                }
                
        except Exception as e:
            logger.error(f"Parallel architecture test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'architecture_statistics': self.get_parallel_processing_stats()
            }
    
    def get_architecture_health(self) -> Dict[str, Any]:
        """Get health status of all architecture components.
        
        Returns:
            Dictionary with health status of each component
        """
        health = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Rate limiter health
            rate_status = self.rate_limiter.get_rate_limit_status()
            health["components"]["rate_limiter"] = {
                "status": "healthy" if not rate_status["circuit_breaker_open"] else "degraded",
                "current_delay": rate_status["current_delay"],
                "error_count": rate_status["error_count"]
            }
            
            # Content processor health
            processor_stats = self.content_processor.get_processing_statistics()
            health["components"]["content_processor"] = {
                "status": "healthy",
                "chunks_processed": processor_stats["total_chunks_processed"],
                "average_preservation": processor_stats["average_preservation_ratio"]
            }
            
            # Pipeline coordinator health (if enabled)
            if self.pipeline_coordinator:
                pipeline_health = asyncio.run(self.pipeline_coordinator.health_check())
                health["components"]["pipeline_coordinator"] = {
                    "status": "healthy" if pipeline_health["pipeline_active"] else "inactive",
                    "active_tasks": pipeline_health["active_scraping_tasks"] + pipeline_health["active_processing_tasks"],
                    "queue_size": pipeline_health["scraping_queue_size"] + pipeline_health["processing_queue_size"]
                }
            
            # Check for any degraded components
            degraded_components = [
                name for name, component in health["components"].items() 
                if component["status"] != "healthy"
            ]
            
            if degraded_components:
                health["overall_status"] = "degraded"
                health["degraded_components"] = degraded_components
                
        except Exception as e:
            health["overall_status"] = "error"
            health["error"] = str(e)
        
        return health