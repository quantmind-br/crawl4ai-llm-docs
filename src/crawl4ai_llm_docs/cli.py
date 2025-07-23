"""Main CLI interface for crawl4ai-llm-docs."""
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

from .config.manager import ConfigManager
from .config.models import AppConfig
from .exceptions import ConfigurationError, FileOperationError
from . import __version__


# Initialize rich console for better output
console = Console()
logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def display_banner() -> None:
    """Display application banner."""
    console.print(Panel(
        f"[bold blue]crawl4ai-llm-docs[/bold blue] v{__version__}\n"
        "[dim]Documentation Scraping and LLM Optimization Tool[/dim]",
        border_style="blue"
    ))


def display_config_info(config_manager: ConfigManager) -> None:
    """Display comprehensive configuration information."""
    info = config_manager.get_config_info()
    config = config_manager.app_config
    
    # Main configuration table
    table = Table(title="Configuration Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Config File", str(info["config_file"]))
    table.add_row("Config Directory", str(info["config_dir"]))
    table.add_row("Base URL", config.base_url)
    table.add_row("Model", config.model)
    table.add_row("Max Workers", str(config.max_workers))
    table.add_row("Debug Mode", str(config.debug))
    
    console.print(table)
    
    # Parallel processing configuration table
    parallel_config = config.parallel_processing
    parallel_table = Table(title="Optimized Parallel Architecture")
    parallel_table.add_column("Component", style="cyan")
    parallel_table.add_column("Setting", style="yellow")
    parallel_table.add_column("Value", style="green")
    
    parallel_table.add_row("Pipeline Coordinator", "Max Concurrent Requests", str(parallel_config.max_concurrent_requests))
    parallel_table.add_row("Rate Limiter", "Adaptive Rate Limiting", str(parallel_config.enable_adaptive_rate_limiting))
    parallel_table.add_row("Rate Limiter", "Buffer Percentage", f"{parallel_config.rate_limit_buffer_percent:.1%}")
    parallel_table.add_row("Rate Limiter", "Circuit Breaker", str(parallel_config.enable_circuit_breaker))
    parallel_table.add_row("Rate Limiter", "Error Threshold", f"{parallel_config.error_threshold_percent:.1%}")
    parallel_table.add_row("Session Manager", "Session Pooling", str(parallel_config.enable_session_pooling))
    parallel_table.add_row("Session Manager", "Max Connections/Host", str(parallel_config.max_connections_per_host))
    parallel_table.add_row("Session Manager", "Request Timeout", f"{parallel_config.request_timeout}s")
    parallel_table.add_row("Progress Tracker", "Update Interval", f"{parallel_config.progress_update_interval}s")
    
    console.print(parallel_table)
    
    # Intelligent chunking configuration
    chunking_table = Table(title="Intelligent Chunking Configuration")
    chunking_table.add_column("Setting", style="cyan")
    chunking_table.add_column("Value", style="green")
    
    chunking_table.add_row("Target Tokens per Chunk", f"{config.chunk_target_tokens:,}")
    chunking_table.add_row("Max Tokens per Chunk", f"{config.chunk_max_tokens:,}")
    chunking_table.add_row("Content-Based Chunking", "Enabled")
    
    console.print(chunking_table)


def interactive_configuration(config_manager: ConfigManager) -> None:
    """Interactive configuration setup."""
    console.print("[bold]Configuration Setup[/bold]")
    console.print("Configure your OpenAI-compatible API settings:\n")
    
    # Get current config or defaults
    try:
        current_config = config_manager.app_config
    except Exception:
        current_config = AppConfig().get_test_config()
    
    # Interactive prompts
    api_key = Prompt.ask(
        "API Key",
        default=current_config.api_key if current_config.api_key else None
    )
    
    base_url = Prompt.ask(
        "Base URL",
        default=current_config.base_url
    )
    
    model = Prompt.ask(
        "Model",
        default=current_config.model
    )
    
    max_workers = int(Prompt.ask(
        "Max Workers",
        default=str(current_config.max_workers)
    ))
    
    debug = Confirm.ask("Enable debug mode?", default=current_config.debug)
    
    # Create new configuration
    try:
        new_config = AppConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_workers=max_workers,
            debug=debug
        )
        
        # Save configuration
        config_manager.save_config(new_config)
        console.print("[green]+ Configuration saved successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]X Configuration error: {e}[/red]")
        raise ConfigurationError(f"Invalid configuration: {e}")


def get_urls_file_interactively() -> Path:
    """Interactively get URLs file from user."""
    console.print("[bold]Select URLs File[/bold]")
    console.print("Enter the path to your text file containing URLs (one per line):\n")
    
    while True:
        file_path = Prompt.ask("URLs file path")
        path = Path(file_path).expanduser()
        
        if path.exists() and path.is_file():
            return path
        else:
            console.print(f"[red]X File not found: {path}[/red]")
            if not Confirm.ask("Try again?"):
                sys.exit(1)


def process_urls_file(
    urls_file: Path,
    config_manager: ConfigManager,
    debug: bool = False,
    test_parallel: bool = False
) -> None:
    """Process URLs file using optimized parallel architecture.
    
    Args:
        urls_file: Path to file containing URLs
        config_manager: Configuration manager instance
        debug: Enable debug logging
        test_parallel: Run parallel processing test
    """
    from .core.scraper import DocumentationScraper
    from .core.processor import DocumentationProcessor
    from .utils.file_handler import FileHandler
    
    setup_logging(debug)
    
    try:
        # Validate configuration
        if not config_manager.is_configured():
            console.print("[red]X Application not configured. Run with --config first.[/red]")
            return
        
        # Load configuration
        config = config_manager.app_config
        
        console.print(f"[cyan]Processing URLs from: {urls_file}[/cyan]")
        
        # Validate parallel processing configuration
        if not config_manager.validate_parallel_processing_config():
            console.print("[red]X Parallel processing configuration is invalid.[/red]")
            return
        
        # Initialize components with optimized architecture
        file_handler = FileHandler()
        scraper = DocumentationScraper(config_manager.crawler_config)
        processor = DocumentationProcessor(config, console)  # Pass console for integrated progress
        
        # Test parallel processing if requested
        if test_parallel:
            console.print("[cyan]Testing optimized parallel architecture...[/cyan]")
            try:
                test_result = asyncio.run(processor.test_parallel_processing(10))
                
                if test_result['status'] == 'completed':
                    console.print(f"[green]+ Parallel architecture test successful![/green]")
                    
                    # Show performance metrics
                    if 'performance_metrics' in test_result:
                        metrics = test_result['performance_metrics']
                        console.print(f"  URLs per second: {metrics.get('urls_per_second', 0):.2f}")
                        console.print(f"  Efficiency ratio: {metrics.get('efficiency_ratio', 0):.2f}x")
                        console.print(f"  Concurrent peak: {metrics.get('concurrent_peak', 0)}")
                    
                    # Show architecture statistics
                    arch_stats = test_result.get('architecture_statistics', {})
                    if arch_stats:
                        rate_limiter = arch_stats.get('rate_limiter', {})
                        console.print(f"  Rate limiter delay: {rate_limiter.get('current_delay', 0):.2f}s")
                        console.print(f"  Circuit breaker: {'Open' if rate_limiter.get('circuit_breaker_open') else 'Closed'}")
                        
                        content_processor = arch_stats.get('content_processor', {})
                        console.print(f"  Content preservation avg: {content_processor.get('average_preservation_ratio', 0):.2%}")
                
                elif test_result['status'] == 'skipped':
                    console.print(f"[yellow]! Parallel processing test skipped: {test_result['reason']}[/yellow]")
                else:
                    console.print(f"[red]X Parallel processing test failed: {test_result.get('error', 'Unknown error')}[/red]")
            except Exception as e:
                console.print(f"[red]X Parallel processing test error: {e}[/red]")
            
            return
        
        # Display architecture health check
        console.print("Performing architecture health check...")
        health_status = processor.get_architecture_health()
        
        if health_status["overall_status"] == "healthy":
            console.print("[green]✓ All architecture components healthy[/green]")
        elif health_status["overall_status"] == "degraded":
            console.print(f"[yellow]⚠ Architecture degraded: {', '.join(health_status.get('degraded_components', []))}[/yellow]")
        else:
            console.print(f"[red]✗ Architecture error: {health_status.get('error', 'Unknown error')}[/red]")
        
        # Display processing configuration
        parallel_config = config.parallel_processing
        is_parallel = parallel_config.max_concurrent_requests > 1
        console.print(f"[dim]Processing mode: {'Optimized Parallel' if is_parallel else 'Sequential'}[/dim]")
        
        if is_parallel:
            console.print(f"[dim]Pipeline coordination: {parallel_config.max_concurrent_requests} concurrent workers[/dim]")
            console.print(f"[dim]Adaptive rate limiting: {parallel_config.enable_adaptive_rate_limiting}[/dim]")
            console.print(f"[dim]Content preservation: Enabled with validation[/dim]")
            console.print(f"[dim]Intelligent chunking: {config.chunk_target_tokens:,} target tokens[/dim]")
        
        # Read and validate URLs
        console.print("Reading URLs...")
        urls = file_handler.read_urls_file(urls_file)
        console.print(f"Found {len(urls)} URLs to process")
        
        # Scrape documentation
        console.print("Scraping documentation...")
        with console.status("[bold green]Scraping websites..."):
            scraped_docs = scraper.scrape_urls(urls)
        
        successful_docs = [doc for doc in scraped_docs if doc.success]
        console.print(f"+ Successfully scraped {len(successful_docs)}/{len(scraped_docs)} documents")
        
        if not successful_docs:
            console.print("[red]X No documents were successfully scraped. Check URLs and network connectivity.[/red]")
            return
        
        # Show comprehensive processing statistics
        try:
            stats = processor.get_processing_stats(successful_docs)
            
            # Display input analysis
            input_analysis = stats.get('input_analysis', {})
            console.print(f"[dim]Input analysis: {input_analysis.get('estimated_tokens', 0):,} tokens, "
                         f"avg {input_analysis.get('average_tokens_per_doc', 0):.0f} tokens/doc[/dim]")
            
            # Display chunking preview
            chunking_stats = stats.get('intelligent_chunking', {})
            if chunking_stats:
                console.print(f"[dim]Chunking strategy: {chunking_stats.get('total_chunks', 0)} chunks, "
                             f"{chunking_stats.get('efficiency_gain', 1):.2f}x more efficient, "
                             f"{chunking_stats.get('estimated_cost_reduction', 0):.1%} cost reduction[/dim]")
            
        except Exception as e:
            logger.debug(f"Could not get processing stats: {e}")
            stats = {}
        
        # Process with LLM using optimized architecture
        console.print("Processing with optimized LLM architecture...")
        
        # The DocumentationProcessor now includes integrated progress tracking
        # Progress will be shown automatically via the ProgressTracker component
        start_time = time.time()
        consolidated_content = processor.consolidate_documentation(successful_docs)
        processing_time = time.time() - start_time
        
        # Generate output filename
        output_file = urls_file.with_suffix('.md')
        
        # Save output
        console.print(f"Saving to: {output_file}")
        file_handler.write_markdown_file(output_file, consolidated_content)
        
        # Display comprehensive completion statistics
        console.print(f"[green]+ Documentation processing complete![/green]")
        console.print(f"[green]Output saved to: {output_file}[/green]")
        console.print(f"[dim]Total processing time: {processing_time:.2f}s[/dim]")
        console.print(f"[dim]Output size: {len(consolidated_content):,} characters[/dim]")
        
        # Show final architecture statistics
        try:
            parallel_stats = processor.get_parallel_processing_stats()
            if parallel_stats:
                # Rate limiter statistics
                rate_stats = parallel_stats.get('rate_limiter', {})
                console.print(f"[dim]Rate limiter: {rate_stats.get('current_delay', 0):.2f}s delay, "
                             f"{rate_stats.get('error_count', 0)} errors[/dim]")
                
                # Content processor statistics
                content_stats = parallel_stats.get('content_processor', {})
                console.print(f"[dim]Content preservation: {content_stats.get('average_preservation_ratio', 0):.2%} avg, "
                             f"{content_stats.get('total_chunks_processed', 0)} chunks processed[/dim]")
                
                # Pipeline coordinator statistics (if available)
                pipeline_stats = parallel_stats.get('pipeline_coordinator', {})
                if pipeline_stats:
                    console.print(f"[dim]Pipeline efficiency: {processing_time / len(successful_docs):.2f}s per document[/dim]")
        except Exception as e:
            logger.debug(f"Could not get final parallel stats: {e}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=debug)
        console.print(f"[red]X Processing failed: {e}[/red]")
        sys.exit(1)


def _apply_cli_overrides(config_manager: ConfigManager, 
                        max_concurrent: Optional[int],
                        disable_adaptive_rate_limiting: bool,
                        progress_interval: Optional[int],
                        disable_session_pooling: bool) -> None:
    """Apply CLI overrides to parallel processing configuration."""
    current_config = config_manager.app_config
    parallel_config = current_config.parallel_processing
    
    # Apply CLI overrides
    overrides = {}
    
    if max_concurrent is not None:
        if 1 <= max_concurrent <= 20:
            overrides['max_concurrent_requests'] = max_concurrent
        else:
            console.print(f"[red]Error: --max-concurrent must be between 1 and 20, got {max_concurrent}[/red]")
            sys.exit(1)
    
    if disable_adaptive_rate_limiting:
        overrides['enable_adaptive_rate_limiting'] = False
    
    if progress_interval is not None:
        if 1 <= progress_interval <= 60:
            overrides['progress_update_interval'] = progress_interval
        else:
            console.print(f"[red]Error: --progress-interval must be between 1 and 60, got {progress_interval}[/red]")
            sys.exit(1)
    
    if disable_session_pooling:
        overrides['enable_session_pooling'] = False
    
    # Apply overrides to parallel processing config
    if overrides:
        from .config.models import ParallelProcessingConfig
        new_parallel_config = ParallelProcessingConfig(
            **{**parallel_config.model_dump(), **overrides}
        )
        
        # Update the app config with new parallel processing config
        new_app_config = current_config.model_copy(
            update={'parallel_processing': new_parallel_config}
        )
        
        # Update the config manager's internal config
        config_manager._app_config = new_app_config
        
        console.print(f"[cyan]Applied CLI overrides: {', '.join(f'{k}={v}' for k, v in overrides.items())}[/cyan]")


@click.command()
@click.argument('urls_file', type=click.Path(path_type=Path), required=False)
@click.option('--config', is_flag=True, help='Configure API settings interactively')
@click.option('--info', is_flag=True, help='Show configuration information')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--max-concurrent', type=int, default=None, 
              help='Maximum number of concurrent LLM requests (1-20)')
@click.option('--disable-adaptive-rate-limiting', is_flag=True, 
              help='Disable adaptive rate limiting based on API headers')
@click.option('--progress-interval', type=int, default=None,
              help='Progress update interval in seconds (1-60)')
@click.option('--disable-session-pooling', is_flag=True,
              help='Disable HTTP session pooling for connection reuse')
@click.option('--test-parallel', is_flag=True,
              help='Test parallel processing functionality and exit')
@click.version_option(version=__version__)
def main(
    urls_file: Optional[Path],
    config: bool,
    info: bool,
    debug: bool,
    max_concurrent: Optional[int],
    disable_adaptive_rate_limiting: bool,
    progress_interval: Optional[int],
    disable_session_pooling: bool,
    test_parallel: bool
) -> None:
    """Process documentation URLs and generate optimized markdown.
    
    URLS_FILE: Text file containing URLs (one per line)
    
    If no URLs file is provided, you'll be prompted interactively.
    """
    setup_logging(debug)
    display_banner()
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Apply CLI overrides to parallel processing configuration
    if any([max_concurrent is not None, disable_adaptive_rate_limiting, 
            progress_interval is not None, disable_session_pooling]):
        _apply_cli_overrides(config_manager, 
                           max_concurrent, disable_adaptive_rate_limiting,
                           progress_interval, disable_session_pooling)
    
    try:
        # Handle configuration mode
        if config:
            interactive_configuration(config_manager)
            return
        
        # Handle info mode
        if info:
            display_config_info(config_manager)
            return
        
        # Handle test parallel mode
        if test_parallel:
            if not config_manager.is_configured():
                console.print("[red]X Application not configured. Run with --config first.[/red]")
                return
            
            # Create a dummy URLs file for testing
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("# Test URLs file for parallel processing test\n")
            
            try:
                process_urls_file(Path(f.name), config_manager, debug, test_parallel=True)
            finally:
                Path(f.name).unlink(missing_ok=True)
            return
        
        # Get URLs file
        if not urls_file:
            urls_file = get_urls_file_interactively()
        
        # Validate URLs file exists
        if not urls_file.exists():
            console.print(f"[red]X URLs file not found: {urls_file}[/red]")
            sys.exit(1)
        
        # Process the documentation
        process_urls_file(urls_file, config_manager, debug, test_parallel)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except ConfigurationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    except FileOperationError as e:
        console.print(f"[red]File operation error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=debug)
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()