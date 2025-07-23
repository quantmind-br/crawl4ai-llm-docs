"""
Advanced debugging tools for performance profiling and content preservation analysis.
Helps identify bottlenecks and validate content preservation in the parallel architecture.
"""
import asyncio
import time
import logging
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from ..core.scraper import ScrapedDocument
from ..core.processor import DocumentationProcessor
from ..core.intelligent_chunker import IntelligentChunker, DocumentChunk
from ..core.content_preservation_processor import ContentPreservationProcessor
from ..config.models import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:
    """Detailed performance profiling data."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_usage_mb: float
    cpu_percent: float
    tokens_processed: int = 0
    items_processed: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreservationAnalysis:
    """Content preservation analysis results."""
    original_content: str
    processed_content: str
    preservation_ratio: float
    critical_elements_preserved: int
    critical_elements_total: int
    technical_terms_preserved: int
    technical_terms_total: int
    code_blocks_preserved: int
    code_blocks_total: int
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PerformanceProfiler:
    """Advanced performance profiler for parallel processing architecture."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize performance profiler.
        
        Args:
            console: Rich console for output
        """
        self.console = console or Console()
        self.profiles: List[PerformanceProfile] = []
        self.active_profiles: Dict[str, PerformanceProfile] = {}
        self.system_baseline = self._get_system_baseline()
        
    def _get_system_baseline(self) -> Dict[str, float]:
        """Get system baseline metrics."""
        process = psutil.Process()
        return {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "available_memory_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024
        }
    
    def start_profile(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start profiling an operation.
        
        Args:
            operation: Name of the operation being profiled
            metadata: Additional metadata about the operation
            
        Returns:
            Profile ID for tracking
        """
        profile_id = f"{operation}_{int(time.time() * 1000)}"
        
        process = psutil.Process()
        profile = PerformanceProfile(
            operation=operation,
            start_time=time.time(),
            end_time=0,
            duration=0,
            memory_usage_mb=process.memory_info().rss / 1024 / 1024,
            cpu_percent=process.cpu_percent(),
            metadata=metadata or {}
        )
        
        self.active_profiles[profile_id] = profile
        logger.debug(f"Started profiling: {operation} (ID: {profile_id})")
        
        return profile_id
    
    def end_profile(self, profile_id: str, 
                   tokens_processed: int = 0, 
                   items_processed: int = 0,
                   error_count: int = 0) -> PerformanceProfile:
        """End profiling an operation.
        
        Args:
            profile_id: Profile ID from start_profile
            tokens_processed: Number of tokens processed
            items_processed: Number of items processed
            error_count: Number of errors encountered
            
        Returns:
            Completed performance profile
        """
        if profile_id not in self.active_profiles:
            logger.warning(f"Profile ID not found: {profile_id}")
            return None
        
        profile = self.active_profiles.pop(profile_id)
        
        # Update final metrics
        process = psutil.Process()
        profile.end_time = time.time()
        profile.duration = profile.end_time - profile.start_time
        profile.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        profile.cpu_percent = process.cpu_percent()
        profile.tokens_processed = tokens_processed
        profile.items_processed = items_processed
        profile.error_count = error_count
        
        self.profiles.append(profile)
        
        logger.debug(f"Completed profiling: {profile.operation} in {profile.duration:.2f}s")
        return profile
    
    async def profile_complete_pipeline(self, config: AppConfig, test_urls: List[str]) -> Dict[str, Any]:
        """Profile the complete processing pipeline with test data.
        
        Args:
            config: Application configuration
            test_urls: List of URLs to test with
            
        Returns:
            Comprehensive performance analysis
        """
        self.console.print("🔍 Starting comprehensive pipeline profiling...")
        
        # Initialize processor
        processor = DocumentationProcessor(config, self.console)
        
        overall_profile_id = self.start_profile("complete_pipeline", {
            "test_urls_count": len(test_urls),
            "config": {
                "parallel_enabled": processor.enable_parallel,
                "max_concurrent": config.parallel_processing.max_concurrent_requests,
                "chunk_target_tokens": config.chunk_target_tokens
            }
        })
        
        try:
            # Profile scraping phase
            scraping_profile_id = self.start_profile("scraping_phase")
            
            from ..core.scraper import DocumentationScraper
            scraper = DocumentationScraper(config)
            scraped_docs = await asyncio.get_event_loop().run_in_executor(
                None, scraper.scrape_urls, test_urls
            )
            
            successful_docs = [doc for doc in scraped_docs if doc.success]
            self.end_profile(scraping_profile_id, 
                           items_processed=len(successful_docs),
                           error_count=len(scraped_docs) - len(successful_docs))
            
            if not successful_docs:
                raise ValueError("No documents were successfully scraped")
            
            # Profile chunking phase
            chunking_profile_id = self.start_profile("intelligent_chunking")
            
            chunks = processor.chunker.create_chunks(successful_docs)
            chunking_stats = processor.chunker.get_chunking_stats(chunks)
            
            total_tokens = sum(chunk.token_count for chunk in chunks)
            self.end_profile(chunking_profile_id,
                           tokens_processed=total_tokens,
                           items_processed=len(chunks))
            
            # Profile content processing phase
            processing_profile_id = self.start_profile("content_processing")
            
            consolidated_content = processor.consolidate_documentation(successful_docs)
            
            self.end_profile(processing_profile_id,
                           tokens_processed=len(consolidated_content) // 4,  # Approximate
                           items_processed=len(chunks))
            
            # Complete overall profiling
            self.end_profile(overall_profile_id,
                           tokens_processed=total_tokens,
                           items_processed=len(successful_docs))
            
            # Generate comprehensive analysis
            analysis = self.generate_performance_analysis()
            analysis["chunking_efficiency"] = chunking_stats
            analysis["output_size"] = len(consolidated_content)
            analysis["content_preview"] = consolidated_content[:500] + "..." if len(consolidated_content) > 500 else consolidated_content
            
            return analysis
            
        except Exception as e:
            self.end_profile(overall_profile_id, error_count=1)
            raise e
    
    def generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis from collected profiles.
        
        Returns:
            Detailed performance analysis
        """
        if not self.profiles:
            return {"error": "No performance profiles available"}
        
        # Calculate aggregate statistics
        durations = [p.duration for p in self.profiles]
        memory_usage = [p.memory_usage_mb for p in self.profiles]
        tokens_per_second = [
            p.tokens_processed / p.duration if p.duration > 0 and p.tokens_processed > 0 else 0
            for p in self.profiles
        ]
        
        analysis = {
            "summary": {
                "total_operations": len(self.profiles),
                "total_duration": sum(durations),
                "avg_duration": statistics.mean(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "min_duration": min(durations) if durations else 0,
                "avg_memory_mb": statistics.mean(memory_usage) if memory_usage else 0,
                "peak_memory_mb": max(memory_usage) if memory_usage else 0,
                "avg_tokens_per_second": statistics.mean(tokens_per_second) if tokens_per_second else 0
            },
            "operation_breakdown": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Analyze by operation type
        operations = {}
        for profile in self.profiles:
            if profile.operation not in operations:
                operations[profile.operation] = []
            operations[profile.operation].append(profile)
        
        for operation, profiles in operations.items():
            op_durations = [p.duration for p in profiles]
            op_tokens = [p.tokens_processed for p in profiles]
            
            analysis["operation_breakdown"][operation] = {
                "count": len(profiles),
                "total_duration": sum(op_durations),
                "avg_duration": statistics.mean(op_durations) if op_durations else 0,
                "total_tokens": sum(op_tokens),
                "avg_tokens_per_operation": statistics.mean(op_tokens) if op_tokens else 0,
                "error_rate": sum(p.error_count for p in profiles) / len(profiles) if profiles else 0
            }
        
        # Identify bottlenecks
        sorted_operations = sorted(
            analysis["operation_breakdown"].items(),
            key=lambda x: x[1]["total_duration"],
            reverse=True
        )
        
        for operation, stats in sorted_operations[:3]:  # Top 3 slowest
            if stats["total_duration"] > analysis["summary"]["total_duration"] * 0.2:  # More than 20% of total time
                analysis["bottlenecks"].append({
                    "operation": operation,
                    "duration": stats["total_duration"],
                    "percentage": (stats["total_duration"] / analysis["summary"]["total_duration"]) * 100
                })
        
        # Generate recommendations
        if analysis["summary"]["avg_tokens_per_second"] < 1000:
            analysis["recommendations"].append("Consider increasing parallel processing workers for better throughput")
        
        if analysis["summary"]["peak_memory_mb"] > 1000:
            analysis["recommendations"].append("High memory usage detected - consider reducing chunk sizes")
        
        for operation, stats in analysis["operation_breakdown"].items():
            if stats["error_rate"] > 0.1:
                analysis["recommendations"].append(f"High error rate in {operation} - review error handling")
        
        return analysis
    
    def display_performance_report(self, analysis: Dict[str, Any]) -> None:
        """Display comprehensive performance report.
        
        Args:
            analysis: Performance analysis data
        """
        # Summary table
        summary = analysis.get("summary", {})
        summary_table = Table(title="Performance Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Operations", str(summary.get("total_operations", 0)))
        summary_table.add_row("Total Duration", f"{summary.get('total_duration', 0):.2f}s")
        summary_table.add_row("Average Duration", f"{summary.get('avg_duration', 0):.2f}s")
        summary_table.add_row("Peak Memory", f"{summary.get('peak_memory_mb', 0):.1f} MB")
        summary_table.add_row("Avg Tokens/sec", f"{summary.get('avg_tokens_per_second', 0):.0f}")
        
        self.console.print(summary_table)
        
        # Operation breakdown
        breakdown = analysis.get("operation_breakdown", {})
        if breakdown:
            breakdown_table = Table(title="Operation Breakdown")
            breakdown_table.add_column("Operation", style="cyan")
            breakdown_table.add_column("Count", style="green")
            breakdown_table.add_column("Total Time", style="yellow")
            breakdown_table.add_column("Avg Time", style="yellow")
            breakdown_table.add_column("Error Rate", style="red")
            
            for operation, stats in breakdown.items():
                breakdown_table.add_row(
                    operation,
                    str(stats["count"]),
                    f"{stats['total_duration']:.2f}s",
                    f"{stats['avg_duration']:.2f}s",
                    f"{stats['error_rate']:.2%}"
                )
            
            self.console.print(breakdown_table)
        
        # Bottlenecks
        bottlenecks = analysis.get("bottlenecks", [])
        if bottlenecks:
            bottleneck_panel = Panel(
                "\n".join([
                    f"• {b['operation']}: {b['duration']:.2f}s ({b['percentage']:.1f}% of total)"
                    for b in bottlenecks
                ]),
                title="Performance Bottlenecks",
                border_style="red"
            )
            self.console.print(bottleneck_panel)
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            rec_panel = Panel(
                "\n".join([f"• {rec}" for rec in recommendations]),
                title="Performance Recommendations",
                border_style="blue"
            )
            self.console.print(rec_panel)


class PreservationDebugger:
    """Advanced debugger for content preservation analysis."""
    
    def __init__(self, config: AppConfig, console: Optional[Console] = None):
        """Initialize preservation debugger.
        
        Args:
            config: Application configuration
            console: Rich console for output
        """
        self.config = config
        self.console = console or Console()
        self.content_processor = ContentPreservationProcessor(config)
        self.analyses: List[PreservationAnalysis] = []
        
    def analyze_preservation(self, original_content: str, processed_content: str) -> PreservationAnalysis:
        """Analyze content preservation quality.
        
        Args:
            original_content: Original content before processing
            processed_content: Content after LLM processing
            
        Returns:
            Detailed preservation analysis
        """
        analysis = PreservationAnalysis(
            original_content=original_content,
            processed_content=processed_content,
            preservation_ratio=0,
            critical_elements_preserved=0,
            critical_elements_total=0,
            technical_terms_preserved=0,
            technical_terms_total=0,
            code_blocks_preserved=0,
            code_blocks_total=0
        )
        
        # Define critical elements to check
        critical_elements = [
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',         # Inline code
            r'https?://[^\s]+', # URLs
            r'\w+\.\w+\(',      # Method calls
            r'class\s+\w+',     # Class definitions
            r'def\s+\w+',       # Function definitions
            r'import\s+\w+',    # Import statements
            r'from\s+\w+',      # From imports
            r'npm\s+install',   # npm commands
            r'pip\s+install',   # pip commands
        ]
        
        technical_terms = [
            'api', 'endpoint', 'parameter', 'return', 'function', 'method', 'class',
            'interface', 'implementation', 'configuration', 'installation', 'setup',
            'authentication', 'authorization', 'request', 'response', 'json', 'xml',
            'database', 'query', 'schema', 'table', 'index', 'migration'
        ]
        
        # Count critical elements
        import re
        for pattern in critical_elements:
            original_matches = len(re.findall(pattern, original_content, re.IGNORECASE))
            processed_matches = len(re.findall(pattern, processed_content, re.IGNORECASE))
            
            analysis.critical_elements_total += original_matches
            analysis.critical_elements_preserved += min(processed_matches, original_matches)
            
            if original_matches > processed_matches:
                analysis.issues.append(f"Lost {original_matches - processed_matches} instances of pattern: {pattern}")
        
        # Count technical terms
        for term in technical_terms:
            original_count = len(re.findall(r'\b' + term + r'\b', original_content, re.IGNORECASE))
            processed_count = len(re.findall(r'\b' + term + r'\b', processed_content, re.IGNORECASE))
            
            analysis.technical_terms_total += original_count
            analysis.technical_terms_preserved += min(processed_count, original_count)
            
            if original_count > processed_count:
                analysis.warnings.append(f"Reduced occurrences of '{term}': {original_count} → {processed_count}")
        
        # Count code blocks specifically
        code_block_pattern = r'```[\s\S]*?```'
        original_code_blocks = len(re.findall(code_block_pattern, original_content))
        processed_code_blocks = len(re.findall(code_block_pattern, processed_content))
        
        analysis.code_blocks_total = original_code_blocks
        analysis.code_blocks_preserved = processed_code_blocks
        
        if original_code_blocks > processed_code_blocks:
            analysis.issues.append(f"Lost {original_code_blocks - processed_code_blocks} code blocks")
        
        # Calculate overall preservation ratio
        total_original = len(original_content)
        total_processed = len(processed_content)
        
        if total_original > 0:
            # Weight different factors
            length_ratio = total_processed / total_original
            critical_ratio = (analysis.critical_elements_preserved / max(1, analysis.critical_elements_total))
            technical_ratio = (analysis.technical_terms_preserved / max(1, analysis.technical_terms_total))
            
            # Combined preservation score
            analysis.preservation_ratio = (
                0.3 * length_ratio +
                0.5 * critical_ratio +
                0.2 * technical_ratio
            )
        
        # Add overall warnings
        if analysis.preservation_ratio < 0.8:
            analysis.warnings.append(f"Low preservation ratio: {analysis.preservation_ratio:.2%}")
        
        if total_processed < total_original * 0.5:
            analysis.issues.append(f"Content significantly shortened: {total_processed} vs {total_original} chars")
        
        self.analyses.append(analysis)
        return analysis
    
    async def debug_processing_pipeline(self, test_documents: List[ScrapedDocument]) -> Dict[str, Any]:
        """Debug the complete content processing pipeline.
        
        Args:
            test_documents: Documents to test processing with
            
        Returns:
            Comprehensive debugging results
        """
        self.console.print("🧪 Starting content preservation debugging...")
        
        results = {
            "total_documents": len(test_documents),
            "preservation_analyses": [],
            "overall_stats": {},
            "recommendations": []
        }
        
        # Test chunking
        chunker = IntelligentChunker(self.config)
        chunks = chunker.create_chunks(test_documents)
        
        # Test each chunk processing
        with Progress(
            TextColumn("[bold blue]Analyzing preservation..."),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TextColumn("{task.completed}/{task.total}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Processing chunks", total=len(chunks))
            
            for i, chunk in enumerate(chunks):
                original_content = chunk.combined_content
                
                # Process chunk
                try:
                    processing_result = await self.content_processor.process_chunks_async([chunk])
                    processed_content = processing_result.cleaned_content
                    
                    # Analyze preservation
                    analysis = self.analyze_preservation(original_content, processed_content)
                    
                    results["preservation_analyses"].append({
                        "chunk_id": chunk.chunk_id,
                        "preservation_ratio": analysis.preservation_ratio,
                        "critical_preservation": analysis.critical_elements_preserved / max(1, analysis.critical_elements_total),
                        "issues_count": len(analysis.issues),
                        "warnings_count": len(analysis.warnings),
                        "original_length": len(original_content),
                        "processed_length": len(processed_content)
                    })
                    
                except Exception as e:
                    results["preservation_analyses"].append({
                        "chunk_id": chunk.chunk_id,
                        "error": str(e),
                        "preservation_ratio": 0
                    })
                
                progress.update(task, advance=1)
        
        # Calculate overall statistics
        valid_analyses = [a for a in results["preservation_analyses"] if "error" not in a]
        
        if valid_analyses:
            preservation_ratios = [a["preservation_ratio"] for a in valid_analyses]
            critical_ratios = [a["critical_preservation"] for a in valid_analyses]
            
            results["overall_stats"] = {
                "avg_preservation_ratio": statistics.mean(preservation_ratios),
                "min_preservation_ratio": min(preservation_ratios),
                "max_preservation_ratio": max(preservation_ratios),
                "avg_critical_preservation": statistics.mean(critical_ratios),
                "total_issues": sum(a["issues_count"] for a in valid_analyses),
                "total_warnings": sum(a["warnings_count"] for a in valid_analyses),
                "chunks_with_issues": len([a for a in valid_analyses if a["issues_count"] > 0])
            }
            
            # Generate recommendations
            if results["overall_stats"]["avg_preservation_ratio"] < 0.8:
                results["recommendations"].append("Consider adjusting content preservation prompts to reduce information loss")
            
            if results["overall_stats"]["chunks_with_issues"] > len(valid_analyses) * 0.3:
                results["recommendations"].append("High number of chunks with preservation issues - review chunking strategy")
            
            if results["overall_stats"]["avg_critical_preservation"] < 0.9:
                results["recommendations"].append("Critical technical elements are being lost - strengthen preservation validation")
        
        return results
    
    def display_preservation_report(self, debug_results: Dict[str, Any]) -> None:
        """Display comprehensive preservation debugging report.
        
        Args:
            debug_results: Results from debug_processing_pipeline
        """
        # Overall statistics
        stats = debug_results.get("overall_stats", {})
        stats_table = Table(title="Content Preservation Analysis")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Chunks", str(len(debug_results.get("preservation_analyses", []))))
        stats_table.add_row("Avg Preservation Ratio", f"{stats.get('avg_preservation_ratio', 0):.2%}")
        stats_table.add_row("Min Preservation Ratio", f"{stats.get('min_preservation_ratio', 0):.2%}")
        stats_table.add_row("Critical Elements Preserved", f"{stats.get('avg_critical_preservation', 0):.2%}")
        stats_table.add_row("Chunks with Issues", str(stats.get('chunks_with_issues', 0)))
        stats_table.add_row("Total Issues", str(stats.get('total_issues', 0)))
        stats_table.add_row("Total Warnings", str(stats.get('total_warnings', 0)))
        
        self.console.print(stats_table)
        
        # Problematic chunks
        analyses = debug_results.get("preservation_analyses", [])
        problematic = [a for a in analyses if a.get("preservation_ratio", 1) < 0.8]
        
        if problematic:
            problem_table = Table(title="Problematic Chunks (< 80% preservation)")
            problem_table.add_column("Chunk ID", style="cyan")
            problem_table.add_column("Preservation", style="red")
            problem_table.add_column("Issues", style="yellow")
            problem_table.add_column("Size Change", style="blue")
            
            for chunk in problematic[:10]:  # Show top 10 worst
                size_change = f"{chunk.get('processed_length', 0)}/{chunk.get('original_length', 1)}"
                problem_table.add_row(
                    chunk.get("chunk_id", "unknown"),
                    f"{chunk.get('preservation_ratio', 0):.2%}",
                    str(chunk.get('issues_count', 0)),
                    size_change
                )
            
            self.console.print(problem_table)
        
        # Recommendations
        recommendations = debug_results.get("recommendations", [])
        if recommendations:
            rec_panel = Panel(
                "\n".join([f"• {rec}" for rec in recommendations]),
                title="Preservation Recommendations",
                border_style="blue"
            )
            self.console.print(rec_panel)
    
    def export_analysis_report(self, output_path: Path, debug_results: Dict[str, Any]) -> None:
        """Export detailed analysis report to file.
        
        Args:
            output_path: Path to export the report to
            debug_results: Debug results to export
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": self.config.model,
                "chunk_target_tokens": self.config.chunk_target_tokens,
                "chunk_max_tokens": self.config.chunk_max_tokens
            },
            "results": debug_results,
            "detailed_analyses": [
                {
                    "preservation_ratio": analysis.preservation_ratio,
                    "critical_elements": {
                        "preserved": analysis.critical_elements_preserved,
                        "total": analysis.critical_elements_total
                    },
                    "technical_terms": {
                        "preserved": analysis.technical_terms_preserved,
                        "total": analysis.technical_terms_total
                    },
                    "code_blocks": {
                        "preserved": analysis.code_blocks_preserved,
                        "total": analysis.code_blocks_total
                    },
                    "issues": analysis.issues,
                    "warnings": analysis.warnings
                }
                for analysis in self.analyses
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.console.print(f"[green]Detailed analysis exported to: {output_path}[/green]")