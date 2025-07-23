"""Markdown optimization for LLM consumption."""
import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken


logger = logging.getLogger(__name__)


@dataclass
class OptimizationStats:
    """Statistics about markdown optimization."""
    original_length: int
    optimized_length: int
    token_reduction: int
    compression_ratio: float
    optimizations_applied: List[str]


class MarkdownOptimizer:
    """Optimizes markdown content for LLM consumption."""
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize optimizer with model-specific settings.
        
        Args:
            model: Target model for optimization
        """
        self.model = model
        self.encoder = self._get_encoder(model)
        self.stats = OptimizationStats(0, 0, 0, 0.0, [])
    
    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get tiktoken encoder for the model.
        
        Args:
            model: Model name
            
        Returns:
            Tiktoken encoder
        """
        try:
            # Map model names to encodings
            encoding_map = {
                'gpt-4o': 'o200k_base',
                'gpt-4o-mini': 'o200k_base',
                'gpt-4': 'cl100k_base',
                'gpt-3.5-turbo': 'cl100k_base',
                'gemini-2.5-flash': 'cl100k_base',  # Fallback for custom models
            }
            
            encoding_name = encoding_map.get(model, 'cl100k_base')
            return tiktoken.get_encoding(encoding_name)
            
        except Exception as e:
            logger.warning(f"Failed to get encoder for {model}: {e}. Using cl100k_base.")
            return tiktoken.get_encoding('cl100k_base')
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # Fallback: approximate tokens as chars/4
            return len(text) // 4
    
    def normalize_heading_structure(self, content: str) -> str:
        """Normalize heading structure for consistency.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with normalized headings
        """
        lines = content.split('\n')
        normalized_lines = []
        
        # Track heading levels to ensure hierarchy
        heading_stack = []
        
        for line in lines:
            # Check if line is a heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Ensure proper heading hierarchy
                while heading_stack and heading_stack[-1] >= level:
                    heading_stack.pop()
                
                heading_stack.append(level)
                
                # Create normalized heading
                normalized_line = '#' * level + ' ' + title
                normalized_lines.append(normalized_line)
            else:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def convert_to_reference_links(self, content: str) -> str:
        """Convert inline links to reference style for token efficiency.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with reference-style links
        """
        # Find all inline links
        inline_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(inline_link_pattern, content)
        
        if not links:
            return content
        
        # Create reference links
        references = []
        link_map = {}
        ref_counter = 1
        
        for text, url in links:
            if url not in link_map:
                link_map[url] = ref_counter
                references.append(f"[{ref_counter}]: {url}")
                ref_counter += 1
        
        # Replace inline links with reference links
        modified_content = content
        for text, url in links:
            ref_num = link_map[url]
            inline_link = f"[{text}]({url})"
            ref_link = f"[{text}][{ref_num}]"
            modified_content = modified_content.replace(inline_link, ref_link, 1)
        
        # Add references at the end
        if references:
            modified_content += '\n\n' + '\n'.join(references)
            self.stats.optimizations_applied.append("reference_links")
        
        return modified_content
    
    def preserve_code_blocks(self, content: str) -> str:
        """Ensure code blocks are properly formatted and preserved.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with preserved code blocks
        """
        # Find code blocks and ensure they have language specifiers
        code_block_pattern = r'```(\w*)\n(.*?)\n```'
        
        def replace_code_block(match):
            language = match.group(1)
            code = match.group(2)
            
            # If no language specified, try to detect common patterns
            if not language:
                code_lower = code.lower().strip()
                if any(keyword in code_lower for keyword in ['def ', 'import ', 'from ', 'class ']):
                    language = 'python'
                elif any(keyword in code_lower for keyword in ['function', 'var ', 'let ', 'const ']):
                    language = 'javascript'
                elif any(keyword in code_lower for keyword in ['<', '/>', 'html', 'div']):
                    language = 'html'
                elif any(keyword in code_lower for keyword in ['{', '}', 'curl', 'http']):
                    language = 'json' if '{' in code_lower and '}' in code_lower else 'bash'
            
            return f'```{language}\n{code}\n```'
        
        return re.sub(code_block_pattern, replace_code_block, content, flags=re.DOTALL)
    
    def remove_redundant_whitespace(self, content: str) -> str:
        """Remove excessive whitespace while preserving formatting.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with optimized whitespace
        """
        # Remove multiple consecutive blank lines (more than 2)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace from lines
        lines = content.split('\n')
        lines = [line.rstrip() for line in lines]
        
        # Remove leading/trailing whitespace from entire content
        content = '\n'.join(lines).strip()
        
        if len(lines) != len(content.split('\n')):
            self.stats.optimizations_applied.append("whitespace_removal")
        
        return content
    
    def optimize_tables(self, content: str) -> str:
        """Optimize markdown tables for better LLM processing.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with optimized tables
        """
        # Find markdown tables and add unique identifiers
        table_pattern = r'(\|[^\n]+\|\n\|[-:|\s]+\|\n(?:\|[^\n]+\|\n?)*)'
        
        def process_table(match):
            table = match.group(1)
            
            # Add a unique identifier comment before the table
            table_id = f"<!-- Table {hash(table) % 1000} -->\n"
            
            # Ensure consistent spacing in table cells
            lines = table.strip().split('\n')
            processed_lines = []
            
            for line in lines:
                if '|' in line:
                    # Clean up cell spacing
                    cells = line.split('|')
                    cells = [cell.strip() for cell in cells]
                    processed_line = '| ' + ' | '.join(cells[1:-1]) + ' |'
                    processed_lines.append(processed_line)
                else:
                    processed_lines.append(line)
            
            return table_id + '\n'.join(processed_lines)
        
        optimized = re.sub(table_pattern, process_table, content, flags=re.MULTILINE)
        
        if optimized != content:
            self.stats.optimizations_applied.append("table_optimization")
        
        return optimized
    
    def add_reading_order_markers(self, content: str) -> str:
        """Add markers to preserve logical reading order.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with reading order markers
        """
        lines = content.split('\n')
        processed_lines = []
        section_counter = 1
        
        for line in lines:
            # Add section markers for major headings
            if re.match(r'^#{1,2}\s+', line.strip()):
                processed_lines.append(f'<!-- Section {section_counter} -->')
                processed_lines.append(line)
                section_counter += 1
            else:
                processed_lines.append(line)
        
        if section_counter > 1:
            self.stats.optimizations_applied.append("reading_order_markers")
        
        return '\n'.join(processed_lines)
    
    def compress_repetitive_content(self, content: str) -> str:
        """Identify and compress repetitive content patterns.
        
        Args:
            content: Markdown content
            
        Returns:
            Content with compressed repetitive patterns
        """
        # Remove repeated navigation patterns
        nav_patterns = [
            r'(\n\s*[\*\-]\s*(Home|Back|Next|Previous|Contents?|Index)\s*\n){2,}',
            r'(\n\s*\[.*(Home|Back|Next|Previous|Contents?|Index).*\]\([^)]+\)\s*\n){2,}',
        ]
        
        original_length = len(content)
        
        for pattern in nav_patterns:
            content = re.sub(pattern, r'\1', content, flags=re.IGNORECASE | re.MULTILINE)
        
        if len(content) < original_length:
            self.stats.optimizations_applied.append("repetitive_content_removal")
        
        return content
    
    def optimize_for_llm(
        self,
        content: str,
        aggressive: bool = False
    ) -> Tuple[str, OptimizationStats]:
        """Optimize markdown content for LLM consumption.
        
        Args:
            content: Original markdown content
            aggressive: Enable aggressive optimizations
            
        Returns:
            Tuple of (optimized_content, stats)
        """
        if not content:
            return content, OptimizationStats(0, 0, 0, 0.0, [])
        
        # Initialize stats
        original_tokens = self.count_tokens(content)
        self.stats = OptimizationStats(
            original_length=len(content),
            optimized_length=0,
            token_reduction=0,
            compression_ratio=0.0,
            optimizations_applied=[]
        )
        
        logger.info(f"Starting optimization of {len(content)} characters ({original_tokens} tokens)")
        
        # Apply optimizations in order
        optimized = content
        
        # Basic optimizations (always applied)
        optimized = self.normalize_heading_structure(optimized)
        optimized = self.preserve_code_blocks(optimized)
        optimized = self.remove_redundant_whitespace(optimized)
        
        # Advanced optimizations
        optimized = self.convert_to_reference_links(optimized)
        optimized = self.optimize_tables(optimized)
        
        if aggressive:
            optimized = self.add_reading_order_markers(optimized)
            optimized = self.compress_repetitive_content(optimized)
        
        # Final cleanup
        optimized = self.remove_redundant_whitespace(optimized)
        
        # Calculate final stats
        optimized_tokens = self.count_tokens(optimized)
        
        self.stats.optimized_length = len(optimized)
        self.stats.token_reduction = original_tokens - optimized_tokens
        self.stats.compression_ratio = (
            (original_tokens - optimized_tokens) / original_tokens * 100
            if original_tokens > 0 else 0.0
        )
        
        logger.info(
            f"Optimization complete: {len(content)} -> {len(optimized)} chars "
            f"({original_tokens} -> {optimized_tokens} tokens, "
            f"{self.stats.compression_ratio:.1f}% reduction)"
        )
        
        return optimized, self.stats
    
    def get_optimization_report(self, stats: OptimizationStats) -> str:
        """Generate a human-readable optimization report.
        
        Args:
            stats: Optimization statistics
            
        Returns:
            Formatted report string
        """
        report = f"""
Markdown Optimization Report
============================

Original Size: {stats.original_length:,} characters
Optimized Size: {stats.optimized_length:,} characters
Size Reduction: {stats.original_length - stats.optimized_length:,} characters

Token Reduction: {stats.token_reduction:,} tokens
Compression Ratio: {stats.compression_ratio:.1f}%

Optimizations Applied:
"""
        
        for optimization in stats.optimizations_applied:
            report += f"- {optimization.replace('_', ' ').title()}\n"
        
        if not stats.optimizations_applied:
            report += "- None (content was already optimized)\n"
        
        return report.strip()