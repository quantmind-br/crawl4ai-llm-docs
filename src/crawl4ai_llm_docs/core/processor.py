"""Documentation processor using OpenAI-compatible APIs."""
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

from tenacity import retry, stop_after_attempt, wait_random_exponential

from .scraper import ScrapedDocument
from .parallel_processor import ParallelLLMProcessor, ProcessingItem, ProcessingResult
from ..config.models import AppConfig
from ..exceptions import ProcessingError, APIError as CustomAPIError


logger = logging.getLogger(__name__)


class DocumentationProcessor:
    """Processes scraped documentation using LLM APIs."""
    
    def __init__(self, config: AppConfig, enable_parallel: bool = None):
        """Initialize processor with configuration.
        
        Args:
            config: Application configuration
            enable_parallel: Enable parallel processing (auto-detect if None)
        """
        self.config = config
        self.client = self._create_client()
        self._validate_dependencies()
        
        # Determine if parallel processing should be enabled
        self.enable_parallel = self._should_enable_parallel(enable_parallel)
        
        # Initialize parallel processor if enabled
        self._parallel_processor: Optional[ParallelLLMProcessor] = None
        if self.enable_parallel:
            self._parallel_processor = ParallelLLMProcessor(
                self.config.parallel_processing,
                self._parallel_process_chunk
            )
            
        logger.info(f"DocumentationProcessor initialized - parallel_processing={self.enable_parallel}")
    
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
        
        # Auto-detect based on configuration
        parallel_config = self.config.parallel_processing
        
        # Enable if max_concurrent_requests > 1 and adaptive rate limiting is enabled
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
    
    def _create_consolidation_prompt(self, documents: List[ScrapedDocument]) -> str:
        """Create prompt for documentation consolidation.
        
        Args:
            documents: List of scraped documents
            
        Returns:
            Consolidation prompt
        """
        # Check if these are already processed chunks vs raw documents
        is_processed_chunks = all(doc.url.startswith("processed_chunk_") for doc in documents)
        
        if is_processed_chunks:
            prompt = """You are a documentation expert. Consolidate the following pre-processed documentation chunks into a single, comprehensive, well-structured markdown document.

These chunks have already been individually processed and cleaned. Your task is to:

1. Merge all chunks into a cohesive, comprehensive document
2. Remove any redundancy between chunks while preserving all important information
3. Create a logical flow and clear structure across all sections
4. Use proper markdown formatting with hierarchical headings (H1, H2, H3, etc.)
5. Ensure smooth transitions between sections
6. Maintain technical accuracy and preserve all details
7. Create a comprehensive table of contents if appropriate
8. Ensure the final document reads as a unified whole, not separate chunks

The final document should be complete, well-organized, and suitable for both human reading and AI consumption.

Pre-processed documentation chunks to consolidate:

"""
        else:
            prompt = """You are a documentation expert. Consolidate the following documentation sections into a single, coherent, well-structured markdown document optimized for LLM consumption.

Requirements:
1. Remove redundancy and duplicate information
2. Maintain technical accuracy and all important details
3. Create logical flow and clear structure
4. Use proper markdown formatting with clear headings
5. Preserve code examples and their syntax highlighting
6. Keep important links and references
7. Organize content hierarchically with H1, H2, H3 headings
8. Ensure the final document is comprehensive yet concise

Guidelines:
- Start with a clear title and overview
- Group related topics together
- Use consistent terminology throughout
- Preserve all technical specifications and parameters
- Include practical examples where available
- Remove navigation elements and boilerplate text

The consolidated document should be optimized for reading by other AI systems while remaining human-readable.

Documentation sections to consolidate:

"""
        
        # Add documents with source information
        for i, doc in enumerate(documents, 1):
            if doc.success and doc.markdown:
                if is_processed_chunks:
                    prompt += f"\n---\n## Chunk {i}\nContent:\n{doc.markdown}\n"
                else:
                    prompt += f"\n---\n## Source {i}: {doc.title}\nURL: {doc.url}\nContent:\n{doc.markdown}\n"
        
        prompt += "\n---\n\nConsolidated Documentation:"
        
        return prompt
    
    def _chunk_documents(self, documents: List[ScrapedDocument]) -> List[List[ScrapedDocument]]:
        """Chunk documents with one document per chunk strategy.
        
        Args:
            documents: List of scraped documents
            
        Returns:
            List of document chunks (each chunk contains exactly one document)
        """
        # Strategy: Each page/document becomes its own chunk
        chunks = []
        
        # Filter successful documents
        valid_docs = [doc for doc in documents if doc.success and doc.markdown]
        
        # Create one chunk per document
        for doc in valid_docs:
            doc_tokens = self.count_tokens(doc.markdown)
            chunks.append([doc])
            logger.debug(f"Created chunk for {doc.url} ({doc_tokens} tokens)")
        
        logger.info(f"Created {len(chunks)} chunks (1 document per chunk) from {len(documents)} total documents")
        return chunks
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    def _process_chunk(self, documents: List[ScrapedDocument]) -> str:
        """Process a chunk of documents with retry logic.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Consolidated content
            
        Raises:
            CustomAPIError: If API call fails
        """
        try:
            prompt = self._create_consolidation_prompt(documents)
            
            # Count tokens for logging
            input_tokens = self.count_tokens(prompt)
            logger.debug(f"Processing chunk with {input_tokens} input tokens")
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical documentation writer. Your task is to consolidate multiple documentation sources into a single, well-structured, and comprehensive document."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content = response.choices[0].message.content
            
            if response.usage:
                logger.info(
                    f"API usage - Input: {response.usage.prompt_tokens}, "
                    f"Output: {response.usage.completion_tokens}, "
                    f"Total: {response.usage.total_tokens}"
                )
            
            return content
            
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            raise CustomAPIError(f"Rate limit exceeded: {e}")
        except APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            raise CustomAPIError(f"API connection error: {e}")
        except APIError as e:
            logger.error(f"API error: {e}")
            raise CustomAPIError(f"API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during processing: {e}")
            raise ProcessingError(f"Processing failed: {e}")
    
    async def _parallel_process_chunk(self, session, item: ProcessingItem) -> ProcessingResult:
        """Process a single chunk in parallel mode.
        
        Args:
            session: HTTP session (not used for OpenAI client)
            item: Processing item with document data
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        try:
            # Extract documents from processing item
            documents_data = item.data.get('documents', [])
            documents = [ScrapedDocument(**doc) for doc in documents_data]
            
            # Process the chunk using existing logic
            content = self._process_chunk(documents)
            
            # Calculate tokens used
            tokens_used = self.count_tokens(content)
            response_time = time.time() - start_time
            
            return ProcessingResult(
                item_id=item.id,
                success=True,
                data={'content': content},
                tokens_used=tokens_used,
                response_time=response_time
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Parallel chunk processing failed for item {item.id}: {error_msg}")
            
            return ProcessingResult(
                item_id=item.id,
                success=False,
                error=error_msg,
                response_time=response_time
            )
    
    def consolidate_documentation(self, documents: List[ScrapedDocument]) -> str:
        """Consolidate scraped documentation into a single markdown document.
        
        Args:
            documents: List of scraped documents
            
        Returns:
            Consolidated markdown content
            
        Raises:
            ProcessingError: If processing fails
        """
        if not documents:
            raise ProcessingError("No documents provided for consolidation")
        
        # Filter successful documents
        successful_docs = [doc for doc in documents if doc.success and doc.markdown]
        
        if not successful_docs:
            raise ProcessingError("No valid documents found for consolidation")
        
        logger.info(f"Starting consolidation of {len(successful_docs)} documents (parallel={self.enable_parallel})")
        
        try:
            total_content = "\n".join(doc.markdown for doc in successful_docs)
            total_tokens = self.count_tokens(total_content)
            
            logger.info(f"Total content: {total_tokens} tokens from {len(successful_docs)} documents")
            
            if len(successful_docs) == 1:
                # Single document - process directly (no parallel benefit)
                logger.info("Processing single document")
                consolidated_content = self._process_chunk(successful_docs)
            elif self.enable_parallel and len(successful_docs) > 1:
                # Use parallel processing for multiple documents
                consolidated_content = asyncio.run(self._parallel_consolidate(successful_docs))
            else:
                # Fall back to sequential processing
                consolidated_content = self._sequential_consolidate(successful_docs)
            
            # Basic validation of output
            if not consolidated_content or len(consolidated_content.strip()) < 100:
                raise ProcessingError("Generated content is too short or empty")
            
            logger.info(f"Consolidation complete. Generated {len(consolidated_content)} characters")
            return consolidated_content
            
        except (CustomAPIError, ProcessingError):
            raise
        except Exception as e:
            error_msg = f"Consolidation failed: {str(e)}"
            logger.error(error_msg)
            raise ProcessingError(error_msg)
    
    def _sequential_consolidate(self, documents: List[ScrapedDocument]) -> str:
        """Sequential consolidation (original logic).
        
        Args:
            documents: List of documents to process
            
        Returns:
            Consolidated content
        """
        logger.info(f"Using sequential processing for {len(documents)} documents")
        
        chunks = self._chunk_documents(documents)
        chunk_results = []
        
        for i, chunk in enumerate(chunks, 1):
            doc = chunk[0]  # Each chunk has exactly one document
            logger.info(f"Processing chunk {i}/{len(chunks)}: {doc.title} ({doc.url})")
            chunk_result = self._process_chunk(chunk)
            chunk_results.append(chunk_result)
            
            # Add delay between chunks to avoid rate limiting
            if i < len(chunks):
                time.sleep(1)
        
        # Simple concatenation of processed chunks (no further LLM consolidation) 
        logger.info(f"Concatenating {len(chunk_results)} processed chunks without compression")
        
        # Join all processed chunks with clear separators
        final_content = "\n\n---\n\n".join(chunk_results)
        
        logger.info(f"Generated final document with {len(final_content)} characters")
        return final_content
    
    async def _parallel_consolidate(self, documents: List[ScrapedDocument]) -> str:
        """Parallel consolidation using ParallelLLMProcessor.
        
        Args:
            documents: List of documents to process
            
        Returns:
            Consolidated content
        """
        logger.info(f"Using parallel processing for {len(documents)} documents")
        
        if not self._parallel_processor:
            raise ProcessingError("Parallel processor not initialized")
        
        # Create processing items for each document chunk
        chunks = self._chunk_documents(documents)
        processing_items = []
        
        for i, chunk in enumerate(chunks):
            doc = chunk[0]  # Each chunk has exactly one document
            estimated_tokens = self.count_tokens(doc.markdown)
            
            # Serialize document data for parallel processing
            item = ProcessingItem(
                id=f"chunk_{i}",
                data={
                    'documents': [{
                        'url': doc.url,
                        'title': doc.title,
                        'content': doc.content,
                        'markdown': doc.markdown,
                        'success': doc.success,
                        'error': doc.error,
                        'scraped_at': doc.scraped_at,
                        'word_count': doc.word_count
                    }]
                },
                estimated_tokens=estimated_tokens
            )
            processing_items.append(item)
        
        # Process chunks in parallel
        results = await self._parallel_processor.process_items(processing_items)
        
        # Extract successful chunk results
        chunk_results = []
        for result in results:
            if result.success and result.data:
                chunk_results.append(result.data['content'])
            else:
                logger.warning(f"Chunk {result.item_id} failed: {result.error}")
        
        if not chunk_results:
            raise ProcessingError("No chunks were successfully processed")
        
        # Simple concatenation of processed chunks (no further LLM consolidation)
        logger.info(f"Concatenating {len(chunk_results)} processed chunks without compression")
        
        # Join all processed chunks with clear separators
        final_content = "\n\n---\n\n".join(chunk_results)
        
        logger.info(f"Generated final document with {len(final_content)} characters")
        return final_content
    
    def get_processing_stats(self, documents: List[ScrapedDocument]) -> Dict[str, Any]:
        """Get statistics about processing input.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary with statistics
        """
        successful_docs = [doc for doc in documents if doc.success and doc.markdown]
        
        total_content = "\n".join(doc.markdown for doc in successful_docs)
        total_tokens = self.count_tokens(total_content)
        total_chars = len(total_content)
        
        stats = {
            "total_documents": len(documents),
            "processable_documents": len(successful_docs),
            "total_characters": total_chars,
            "estimated_tokens": total_tokens,
            "chunks_needed": max(1, (total_tokens + self.config.max_tokens - 1000 - 1) // (self.config.max_tokens - 1000)),
            "model": self.config.model,
            "max_tokens_per_request": self.config.max_tokens,
            "parallel_processing_enabled": self.enable_parallel,
            "max_concurrent_requests": self.config.parallel_processing.max_concurrent_requests if self.enable_parallel else 1
        }
        
        return stats
    
    def get_parallel_processing_stats(self) -> Optional[Dict[str, Any]]:
        """Get parallel processing statistics if available.
        
        Returns:
            Dictionary with parallel processing statistics or None
        """
        if not self._parallel_processor:
            return None
        
        return self._parallel_processor.get_processing_statistics()
    
    async def test_parallel_processing(self, num_test_items: int = 5) -> Dict[str, Any]:
        """Test parallel processing functionality.
        
        Args:
            num_test_items: Number of test items to process
            
        Returns:
            Test results and statistics
        """
        if not self.enable_parallel or not self._parallel_processor:
            return {
                'status': 'skipped',
                'reason': 'Parallel processing not enabled'
            }
        
        logger.info(f"Testing parallel processing with {num_test_items} items")
        
        try:
            results = await self._parallel_processor.test_parallel_processing(num_test_items)
            stats = self._parallel_processor.get_processing_statistics()
            
            return {
                'status': 'completed',
                'test_items': num_test_items,
                'results_count': len(results),
                'successful_results': len([r for r in results if r.success]),
                'failed_results': len([r for r in results if not r.success]),
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Parallel processing test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }