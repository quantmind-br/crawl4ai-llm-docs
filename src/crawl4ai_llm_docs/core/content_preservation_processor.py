"""
Content Preservation Processor - Focuses on cleaning and organizing content instead of summarizing.
Preserves all technical information while improving readability and structure.
"""
import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from openai import OpenAI, AsyncOpenAI

from .intelligent_chunker import DocumentChunk
from .adaptive_rate_limiter import AdaptiveRateLimiter
from ..config.models import AppConfig
from ..exceptions import ProcessingError, APIError

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of content processing operation."""
    cleaned_content: str
    original_token_count: int
    processed_token_count: int
    processing_time: float
    preservation_ratio: float  # How much content was preserved
    chunk_ids: List[str]
    metadata: Dict[str, Any]


class ContentPreservationProcessor:
    """LLM processor focused on content preservation rather than summarization."""
    
    # Content cleaning prompt that preserves all information
    CONTENT_CLEANING_PROMPT = """You are a technical documentation processor. Your task is to clean and organize documentation content while preserving ALL technical information.

CRITICAL REQUIREMENTS:
1. PRESERVE ALL CODE EXAMPLES - Never remove or shorten code blocks
2. PRESERVE ALL TECHNICAL DETAILS - Keep all parameters, return values, examples
3. PRESERVE ALL API REFERENCES - Keep all method signatures, class definitions
4. PRESERVE ALL CONFIGURATION OPTIONS - Keep all settings, flags, parameters
5. PRESERVE ALL INSTALLATION INSTRUCTIONS - Keep all commands, steps, requirements

YOUR TASKS:
1. Remove duplicate content and redundant explanations
2. Fix formatting and markdown structure
3. Organize content logically with clear headings
4. Clean up HTML artifacts and malformed markdown
5. Standardize code block formatting
6. Remove navigation elements, footers, and non-content text
7. Merge related sections that were split artificially

WHAT NOT TO DO:
- Never summarize or condense technical content
- Never remove code examples or snippets
- Never remove parameter lists or API details
- Never remove configuration examples
- Never remove installation steps or commands
- Never change technical terminology or exact wording

OUTPUT FORMAT:
Return clean, well-structured markdown that maintains 100% of the technical information while improving readability and organization.

Here is the content to process:

{content}

Cleaned content:"""

    # Alternative prompt for highly technical content
    TECHNICAL_CLEANING_PROMPT = """You are processing technical documentation. Your goal is content organization and cleanup, NOT summarization.

PRESERVE EVERYTHING:
- All code examples and snippets
- All API methods, parameters, and return types  
- All configuration options and settings
- All installation commands and requirements
- All troubleshooting information
- All examples and tutorials

CLEAN AND ORGANIZE:
- Remove HTML artifacts and broken markup
- Fix markdown formatting issues
- Organize with clear section headers
- Remove duplicate/redundant text
- Remove navigation menus and footers
- Standardize code block language tags
- Improve paragraph structure

Content to clean and organize:

{content}

Organized content:"""

    def __init__(self, config: AppConfig, rate_limiter: Optional[AdaptiveRateLimiter] = None):
        """Initialize content preservation processor.
        
        Args:
            config: Application configuration
            rate_limiter: Optional rate limiter for API calls
        """
        self.config = config
        self.rate_limiter = rate_limiter
        
        # Initialize OpenAI clients
        self.sync_client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.async_client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # Processing statistics
        self.total_chunks_processed = 0
        self.total_processing_time = 0.0
        self.total_tokens_processed = 0
        self.preservation_scores = []
        
        logger.info(f"ContentPreservationProcessor initialized with model {config.model}")
    
    async def process_chunks_async(self, chunks: List[DocumentChunk]) -> ProcessingResult:
        """Process document chunks asynchronously with content preservation.
        
        Args:
            chunks: List of document chunks to process
            
        Returns:
            ProcessingResult with preserved and cleaned content
        """
        if not chunks:
            logger.warning("No chunks provided for processing")
            return ProcessingResult(
                cleaned_content="",
                original_token_count=0,
                processed_token_count=0,
                processing_time=0.0,
                preservation_ratio=0.0,
                chunk_ids=[],
                metadata={"error": "No chunks provided"}
            )
        
        logger.info(f"Processing {len(chunks)} chunks asynchronously")
        start_time = time.time()
        
        try:
            # Process chunks in parallel with rate limiting
            semaphore = asyncio.Semaphore(self.config.parallel_processing.max_concurrent_requests)
            
            async def process_single_chunk(chunk: DocumentChunk) -> tuple[DocumentChunk, str]:
                async with semaphore:
                    if self.rate_limiter:
                        await self.rate_limiter.acquire()
                    
                    try:
                        cleaned = await self._clean_chunk_content_async(chunk)
                        
                        if self.rate_limiter:
                            response_time = time.time()
                            self.rate_limiter.on_response({}, response_time - time.time())
                        
                        return chunk, cleaned
                        
                    except Exception as e:
                        if self.rate_limiter:
                            self.rate_limiter.on_error(e, "processing")
                        logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
                        return chunk, chunk.combined_content  # Return original on error
            
            # Process all chunks
            results = await asyncio.gather(
                *[process_single_chunk(chunk) for chunk in chunks],
                return_exceptions=True
            )
            
            # Collect results
            cleaned_sections = []
            chunk_ids = []
            original_tokens = 0
            processing_errors = 0
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Chunk processing exception: {result}")
                    processing_errors += 1
                    continue
                
                chunk, cleaned_content = result
                cleaned_sections.append(cleaned_content)
                chunk_ids.append(chunk.chunk_id)
                original_tokens += chunk.token_count
            
            # Combine all cleaned content
            final_content = self._combine_cleaned_sections(cleaned_sections, chunks)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            processed_tokens = self._estimate_token_count(final_content)
            preservation_ratio = min(1.0, processed_tokens / max(1, original_tokens))
            
            # Update statistics
            self.total_chunks_processed += len(chunks)
            self.total_processing_time += processing_time
            self.total_tokens_processed += processed_tokens
            self.preservation_scores.append(preservation_ratio)
            
            logger.info(f"Processed {len(chunks)} chunks in {processing_time:.2f}s, "
                       f"preservation ratio: {preservation_ratio:.2%}")
            
            return ProcessingResult(
                cleaned_content=final_content,
                original_token_count=original_tokens,
                processed_token_count=processed_tokens,
                processing_time=processing_time,
                preservation_ratio=preservation_ratio,
                chunk_ids=chunk_ids,
                metadata={
                    "chunks_processed": len(chunks),
                    "processing_errors": processing_errors,
                    "avg_chunk_tokens": original_tokens / len(chunks) if chunks else 0,
                    "model_used": self.config.model
                }
            )
            
        except Exception as e:
            logger.error(f"Async chunk processing failed: {e}")
            raise ProcessingError(f"Failed to process chunks: {e}")
    
    def process_chunks_sync(self, chunks: List[DocumentChunk]) -> ProcessingResult:
        """Process document chunks synchronously (fallback method).
        
        Args:
            chunks: List of document chunks to process
            
        Returns:
            ProcessingResult with preserved and cleaned content
        """
        if not chunks:
            return ProcessingResult(
                cleaned_content="",
                original_token_count=0,
                processed_token_count=0,
                processing_time=0.0,
                preservation_ratio=0.0,
                chunk_ids=[],
                metadata={"error": "No chunks provided"}
            )
        
        logger.info(f"Processing {len(chunks)} chunks synchronously")
        start_time = time.time()
        
        cleaned_sections = []
        chunk_ids = []
        original_tokens = 0
        
        try:
            for chunk in chunks:
                try:
                    cleaned = self._clean_chunk_content_sync(chunk)
                    cleaned_sections.append(cleaned)
                    chunk_ids.append(chunk.chunk_id)
                    original_tokens += chunk.token_count
                    
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk.chunk_id}: {e}")
                    # Use original content on error
                    cleaned_sections.append(chunk.combined_content)
                    chunk_ids.append(chunk.chunk_id)
                    original_tokens += chunk.token_count
            
            # Combine results
            final_content = self._combine_cleaned_sections(cleaned_sections, chunks)
            
            processing_time = time.time() - start_time
            processed_tokens = self._estimate_token_count(final_content)
            preservation_ratio = min(1.0, processed_tokens / max(1, original_tokens))
            
            return ProcessingResult(
                cleaned_content=final_content,
                original_token_count=original_tokens,
                processed_token_count=processed_tokens,
                processing_time=processing_time,
                preservation_ratio=preservation_ratio,
                chunk_ids=chunk_ids,
                metadata={
                    "chunks_processed": len(chunks),
                    "sync_processing": True,
                    "model_used": self.config.model
                }
            )
            
        except Exception as e:
            logger.error(f"Sync chunk processing failed: {e}")
            raise ProcessingError(f"Failed to process chunks: {e}")
    
    async def _clean_chunk_content_async(self, chunk: DocumentChunk) -> str:
        """Clean content of a single chunk asynchronously."""
        content = chunk.combined_content.strip()
        if not content:
            return ""
        
        # Choose appropriate prompt based on content type
        prompt = self._select_cleaning_prompt(content)
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical documentation processor focused on content preservation and organization."
                    },
                    {
                        "role": "user", 
                        "content": prompt.format(content=content)
                    }
                ],
                max_tokens=min(self.config.max_tokens, len(content.split()) * 2),  # Allow room for cleaning
                temperature=self.config.temperature,
                timeout=self.config.parallel_processing.request_timeout
            )
            
            cleaned_content = response.choices[0].message.content.strip()
            
            # Validate that content wasn't over-condensed
            preservation_check = self._validate_content_preservation(content, cleaned_content)
            
            if preservation_check < 0.7:  # Less than 70% preserved is concerning
                logger.warning(f"Low preservation ratio {preservation_check:.2%} for chunk {chunk.chunk_id}")
                # Return original if too much was lost
                return content
            
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Async content cleaning failed for chunk {chunk.chunk_id}: {e}")
            raise APIError(f"API call failed: {e}")
    
    def _clean_chunk_content_sync(self, chunk: DocumentChunk) -> str:
        """Clean content of a single chunk synchronously."""
        content = chunk.combined_content.strip()
        if not content:
            return ""
        
        prompt = self._select_cleaning_prompt(content)
        
        try:
            response = self.sync_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical documentation processor focused on content preservation and organization."
                    },
                    {
                        "role": "user",
                        "content": prompt.format(content=content)
                    }
                ],
                max_tokens=min(self.config.max_tokens, len(content.split()) * 2),
                temperature=self.config.temperature,
                timeout=self.config.parallel_processing.request_timeout
            )
            
            cleaned_content = response.choices[0].message.content.strip()
            
            # Validate preservation
            preservation_check = self._validate_content_preservation(content, cleaned_content)
            
            if preservation_check < 0.7:
                logger.warning(f"Low preservation ratio {preservation_check:.2%} for chunk {chunk.chunk_id}")
                return content
            
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Sync content cleaning failed for chunk {chunk.chunk_id}: {e}")
            raise APIError(f"API call failed: {e}")
    
    def _select_cleaning_prompt(self, content: str) -> str:
        """Select appropriate cleaning prompt based on content characteristics."""
        # Check for high technical content density
        technical_indicators = [
            "```", "def ", "class ", "function", "import ", "from ",
            "npm install", "pip install", "curl", "wget", "git clone",
            "http://", "https://", "api.", "config", "settings"
        ]
        
        technical_count = sum(1 for indicator in technical_indicators if indicator in content.lower())
        
        if technical_count > 5 or len(content) > 10000:
            return self.TECHNICAL_CLEANING_PROMPT
        else:
            return self.CONTENT_CLEANING_PROMPT
    
    def _validate_content_preservation(self, original: str, cleaned: str) -> float:
        """Validate that important content was preserved."""
        if not original or not cleaned:
            return 0.0
        
        # Check for preservation of critical elements
        critical_elements = [
            "```",  # Code blocks
            "def ", "class ", "function",  # Code definitions  
            "install", "npm", "pip", "curl",  # Installation commands
            "http", "api", "endpoint",  # API references
            "config", "setting", "option"  # Configuration
        ]
        
        original_critical = sum(1 for elem in critical_elements if elem.lower() in original.lower())
        cleaned_critical = sum(1 for elem in critical_elements if elem.lower() in cleaned.lower())
        
        if original_critical == 0:
            # No critical elements to preserve
            return len(cleaned) / len(original)
        
        # Return ratio of preserved critical elements
        critical_preservation = cleaned_critical / original_critical
        
        # Also consider overall length preservation (should not shrink too much)
        length_preservation = len(cleaned) / len(original)
        
        # Combined score favoring critical element preservation
        return 0.7 * critical_preservation + 0.3 * length_preservation
    
    def _combine_cleaned_sections(self, sections: List[str], original_chunks: List[DocumentChunk]) -> str:
        """Combine cleaned sections into final coherent document."""
        if not sections:
            return ""
        
        combined_parts = []
        
        for i, (section, chunk) in enumerate(zip(sections, original_chunks)):
            if section.strip():
                # Add section divider with source info
                if i > 0:
                    combined_parts.append("\n\n---\n\n")
                
                # Add source reference
                if chunk.documents:
                    urls = [doc.url for doc in chunk.documents[:3]]  # Show up to 3 URLs
                    url_list = ", ".join(urls)
                    if len(chunk.documents) > 3:
                        url_list += f" (and {len(chunk.documents) - 3} more)"
                    
                    combined_parts.append(f"<!-- Sources: {url_list} -->\n\n")
                
                combined_parts.append(section)
        
        final_content = "".join(combined_parts)
        
        # Post-processing cleanup
        final_content = self._post_process_combined_content(final_content)
        
        return final_content
    
    def _post_process_combined_content(self, content: str) -> str:
        """Final cleanup of combined content."""
        # Remove excessive whitespace
        lines = content.split('\n')
        cleaned_lines = []
        
        prev_empty = False
        for line in lines:
            line = line.rstrip()
            
            if not line:  # Empty line
                if not prev_empty:  # Only keep one empty line
                    cleaned_lines.append(line)
                prev_empty = True
            else:
                cleaned_lines.append(line)
                prev_empty = False
        
        # Remove more than 2 consecutive empty lines
        result = '\n'.join(cleaned_lines)
        
        # Clean up multiple consecutive separators
        result = result.replace('\n\n\n\n', '\n\n')
        result = result.replace('---\n\n---', '---')
        
        return result.strip()
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        avg_preservation = (sum(self.preservation_scores) / len(self.preservation_scores) 
                          if self.preservation_scores else 0.0)
        
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "total_processing_time": self.total_processing_time,
            "total_tokens_processed": self.total_tokens_processed,
            "average_preservation_ratio": avg_preservation,
            "chunks_per_second": (self.total_chunks_processed / max(1, self.total_processing_time)),
            "tokens_per_second": (self.total_tokens_processed / max(1, self.total_processing_time)),
            "preservation_score_distribution": {
                "min": min(self.preservation_scores) if self.preservation_scores else 0,
                "max": max(self.preservation_scores) if self.preservation_scores else 0,
                "avg": avg_preservation
            }
        }
    
    async def test_content_preservation(self, test_content: str) -> Dict[str, Any]:
        """Test content preservation with sample content."""
        from .intelligent_chunker import DocumentChunk
        from .scraper import ScrapedDocument
        
        # Create test document and chunk
        test_doc = ScrapedDocument(
            url="https://test.example.com",
            title="Test Document",
            markdown=test_content,
            success=True
        )
        
        test_chunk = DocumentChunk(chunk_id="test_chunk")
        test_chunk.add_document(test_doc, len(test_content) // 4)
        
        # Process the test chunk
        result = await self.process_chunks_async([test_chunk])
        
        return {
            "original_length": len(test_content),
            "cleaned_length": len(result.cleaned_content),
            "preservation_ratio": result.preservation_ratio,
            "processing_time": result.processing_time,
            "original_tokens": result.original_token_count,
            "processed_tokens": result.processed_token_count,
            "cleaned_content_preview": result.cleaned_content[:500] + "..." if len(result.cleaned_content) > 500 else result.cleaned_content
        }