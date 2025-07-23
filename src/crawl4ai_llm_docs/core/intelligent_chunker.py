"""
Intelligent content-based chunking for optimal API usage.
"""
import logging
import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..core.scraper import ScrapedDocument
from ..config.models import AppConfig

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """A chunk of documents optimized for processing."""
    documents: List[ScrapedDocument]
    combined_content: str
    token_count: int
    estimated_processing_time: float
    chunk_id: str
    
    def __init__(self, chunk_id: str = ""):
        self.documents = []
        self.combined_content = ""
        self.token_count = 0
        self.estimated_processing_time = 0.0
        self.chunk_id = chunk_id
        
    def add_document(self, document: ScrapedDocument, token_count: int) -> None:
        """Add a document to this chunk."""
        self.documents.append(document)
        
        # Build combined content with clear document separators
        separator = f"\n\n=== DOCUMENT: {document.title} ({document.url}) ===\n\n"
        if self.combined_content:
            self.combined_content += separator + document.markdown
        else:
            self.combined_content = separator + document.markdown
            
        self.token_count += token_count
        # Estimate 1 second per 1000 tokens processing time
        self.estimated_processing_time = self.token_count / 1000.0

class IntelligentChunker:
    """Create optimal chunks based on content analysis and token limits."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.target_tokens = config.chunk_target_tokens
        self.max_tokens = config.chunk_max_tokens
        
        # Initialize token encoder
        try:
            # Try to use the model-specific encoder
            model_encoders = {
                'gpt-4o': 'o200k_base',
                'gpt-4o-mini': 'o200k_base', 
                'gpt-4': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base',
                'gemini-2.5-flash': 'cl100k_base',  # Fallback
            }
            
            model_key = self.config.model.split('/')[-1] if '/' in self.config.model else self.config.model
            encoder_name = model_encoders.get(model_key, 'cl100k_base')
            self.encoder = tiktoken.get_encoding(encoder_name)
            
            logger.info(f"IntelligentChunker initialized with {encoder_name} encoder for model {self.config.model}")
            
        except Exception as e:
            logger.warning(f"Could not initialize tiktoken encoder: {e}. Using character approximation.")
            self.encoder = None
            
        logger.info(f"Target: {self.target_tokens} tokens, Max: {self.max_tokens} tokens per chunk")
        
    def create_chunks(self, documents: List[ScrapedDocument]) -> List[DocumentChunk]:
        """Create optimal chunks that maximize API efficiency."""
        if not documents:
            return []
            
        # Filter successful documents with content
        valid_docs = [doc for doc in documents if doc.success and doc.markdown and doc.markdown.strip()]
        
        if not valid_docs:
            logger.warning("No valid documents found for chunking")
            return []
            
        logger.info(f"Chunking {len(valid_docs)} valid documents")
        
        # Calculate token counts for all documents
        doc_tokens = []
        for doc in valid_docs:
            token_count = self._count_tokens(doc.markdown)
            doc_tokens.append((doc, token_count))
            
        # Sort documents by token count for optimal packing
        doc_tokens.sort(key=lambda x: x[1], reverse=True)
        
        chunks = []
        current_chunk = DocumentChunk(chunk_id=f"chunk_{len(chunks)}")
        
        for doc, token_count in doc_tokens:
            # Check if adding this document would exceed max tokens
            if (current_chunk.token_count + token_count > self.max_tokens and 
                current_chunk.documents):
                # Complete current chunk and start new one
                chunks.append(current_chunk)
                current_chunk = DocumentChunk(chunk_id=f"chunk_{len(chunks)}")
                
            # Add document to current chunk
            current_chunk.add_document(doc, token_count)
            
            # If we've reached target size and have reasonable content, consider completing chunk
            if (current_chunk.token_count >= self.target_tokens and 
                len(current_chunk.documents) >= 1):
                chunks.append(current_chunk)
                current_chunk = DocumentChunk(chunk_id=f"chunk_{len(chunks)}")
                
        # Add remaining documents if any
        if current_chunk.documents:
            chunks.append(current_chunk)
            
        # Optimization: redistribute very small chunks
        chunks = self._optimize_small_chunks(chunks)
        
        logger.info(f"Created {len(chunks)} optimized chunks")
        for i, chunk in enumerate(chunks):
            logger.info(f"  Chunk {i}: {len(chunk.documents)} docs, {chunk.token_count} tokens")
            
        return chunks
        
    def _optimize_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Redistribute very small chunks to improve efficiency."""
        if len(chunks) <= 1:
            return chunks
            
        optimized_chunks = []
        small_chunk_threshold = self.target_tokens // 3  # Less than 1/3 target is "small"
        
        pending_small = None
        
        for chunk in chunks:
            if chunk.token_count < small_chunk_threshold and len(chunks) > 2:
                if pending_small is None:
                    # Hold this small chunk to combine with next
                    pending_small = chunk
                else:
                    # Combine with pending small chunk
                    combined_chunk = DocumentChunk(chunk_id=f"combined_{len(optimized_chunks)}")
                    
                    # Add all documents from both small chunks
                    for doc in pending_small.documents:
                        doc_tokens = self._count_tokens(doc.markdown)
                        combined_chunk.add_document(doc, doc_tokens)
                        
                    for doc in chunk.documents:
                        doc_tokens = self._count_tokens(doc.markdown)
                        # Check if adding would exceed max tokens
                        if combined_chunk.token_count + doc_tokens <= self.max_tokens:
                            combined_chunk.add_document(doc, doc_tokens)
                        else:
                            # Start a new chunk with remaining documents
                            if combined_chunk.documents:
                                optimized_chunks.append(combined_chunk)
                            combined_chunk = DocumentChunk(chunk_id=f"overflow_{len(optimized_chunks)}")
                            combined_chunk.add_document(doc, doc_tokens)
                            
                    optimized_chunks.append(combined_chunk)
                    pending_small = None
            else:
                # Regular-sized chunk or can't combine
                if pending_small is not None:
                    # Add the pending small chunk as-is
                    optimized_chunks.append(pending_small)
                    pending_small = None
                optimized_chunks.append(chunk)
                
        # Don't forget any remaining small chunk
        if pending_small is not None:
            optimized_chunks.append(pending_small)
            
        return optimized_chunks
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the appropriate encoder."""
        if not text:
            return 0
            
        try:
            if self.encoder:
                return len(self.encoder.encode(text))
            else:
                # Fallback: approximate as chars/4
                return len(text) // 4
        except Exception as e:
            logger.warning(f"Token counting failed: {e}. Using character approximation.")
            return len(text) // 4
            
    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking result."""
        if not chunks:
            return {"error": "No chunks provided"}
            
        total_docs = sum(len(chunk.documents) for chunk in chunks)
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens_per_chunk = total_tokens / len(chunks) if chunks else 0
        
        # Efficiency metrics
        target_efficiency = sum(1 for chunk in chunks if chunk.token_count >= self.target_tokens) / len(chunks)
        utilization = avg_tokens_per_chunk / self.target_tokens if self.target_tokens > 0 else 0
        
        # Estimated performance improvement
        naive_chunks = len([doc for chunk in chunks for doc in chunk.documents])  # One doc per chunk
        efficiency_gain = naive_chunks / len(chunks) if chunks else 1
        
        return {
            "total_chunks": len(chunks),
            "total_documents": total_docs,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens_per_chunk,
            "target_efficiency": target_efficiency,  # % of chunks meeting target
            "token_utilization": utilization,        # How well we use target tokens
            "efficiency_gain": efficiency_gain,      # How many fewer API calls vs naive
            "estimated_cost_reduction": (efficiency_gain - 1) / efficiency_gain if efficiency_gain > 1 else 0,
            "chunk_distribution": [
                {
                    "chunk_id": chunk.chunk_id,
                    "documents": len(chunk.documents),
                    "tokens": chunk.token_count,
                    "efficiency": chunk.token_count / self.target_tokens,
                    "estimated_time": chunk.estimated_processing_time
                }
                for chunk in chunks
            ]
        }