"""Main CLI interface for crawl4ai-llm-docs."""
import asyncio
import logging
import sys
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
    """Display configuration information."""
    info = config_manager.get_config_info()
    config = config_manager.app_config
    
    table = Table(title="Configuration Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Config File", str(info["config_file"]))
    table.add_row("Config Directory", str(info["config_dir"]))
    table.add_row("Base URL", config.base_url)
    table.add_row("Model", config.model)
    table.add_row("Max Workers", str(config.max_workers))
    table.add_row("Debug Mode", str(config.debug))
    
    # Add parallel processing configuration
    parallel_config = config.parallel_processing
    table.add_row("--- Parallel Processing ---", "")
    table.add_row("Max Concurrent Requests", str(parallel_config.max_concurrent_requests))
    table.add_row("Adaptive Rate Limiting", str(parallel_config.enable_adaptive_rate_limiting))
    table.add_row("Session Pooling", str(parallel_config.enable_session_pooling))
    table.add_row("Progress Update Interval", f"{parallel_config.progress_update_interval}s")
    table.add_row("Request Timeout", f"{parallel_config.request_timeout}s")
    table.add_row("Error Threshold", f"{parallel_config.error_threshold_percent:.1%}")
    
    console.print(table)


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
    """Process URLs file and generate documentation.
    
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
        
        # Initialize components
        file_handler = FileHandler()
        scraper = DocumentationScraper(config_manager.crawler_config)
        processor = DocumentationProcessor(config)
        
        # Test parallel processing if requested
        if test_parallel:
            console.print("[cyan]Testing parallel processing functionality...[/cyan]")
            try:
                test_result = asyncio.run(processor.test_parallel_processing(10))
                
                if test_result['status'] == 'completed':
                    console.print(f"[green]+ Parallel processing test successful![/green]")
                    console.print(f"  Test items: {test_result['test_items']}")
                    console.print(f"  Successful: {test_result['successful_results']}")
                    console.print(f"  Failed: {test_result['failed_results']}")
                    
                    # Show detailed statistics
                    stats = test_result.get('statistics', {})
                    if stats:
                        processing_stats = stats.get('processing', {})
                        console.print(f"  Success rate: {processing_stats.get('success_rate', 0):.1%}")
                        console.print(f"  Active tasks: {processing_stats.get('active_tasks', 0)}")
                        
                        progress_stats = stats.get('progress', {})
                        if progress_stats:
                            console.print(f"  Requests/sec: {progress_stats.get('requests_per_second', 0):.2f}")
                            console.print(f"  Avg response time: {progress_stats.get('average_response_time', 0):.2f}s")
                
                elif test_result['status'] == 'skipped':
                    console.print(f"[yellow]! Parallel processing test skipped: {test_result['reason']}[/yellow]")
                else:
                    console.print(f"[red]X Parallel processing test failed: {test_result.get('error', 'Unknown error')}[/red]")
            except Exception as e:
                console.print(f"[red]X Parallel processing test error: {e}[/red]")
            
            return
        
        # Display processing configuration
        parallel_config = config.parallel_processing
        console.print(f"[dim]Processing mode: {'Parallel' if parallel_config.max_concurrent_requests > 1 else 'Sequential'}[/dim]")
        if parallel_config.max_concurrent_requests > 1:
            console.print(f"[dim]Max concurrent requests: {parallel_config.max_concurrent_requests}[/dim]")
            console.print(f"[dim]Adaptive rate limiting: {parallel_config.enable_adaptive_rate_limiting}[/dim]")
        
        # Read and validate URLs
        console.print("Reading URLs...")
        urls = file_handler.read_urls_file(urls_file)
        console.print(f"Found {len(urls)} URLs to process")
        
        # Scrape documentation
        console.print("Scraping documentation...")
        with console.status("[bold green]Scraping websites..."):
            scraped_docs = scraper.scrape_urls(urls)
        
        console.print(f"+ Successfully scraped {len(scraped_docs)} documents")
        
        # Show processing statistics
        try:
            stats = processor.get_processing_stats(scraped_docs)
            console.print(f"[dim]Estimated tokens: {stats['estimated_tokens']:,}[/dim]")
            console.print(f"[dim]Processing mode: {'Parallel' if stats['parallel_processing_enabled'] else 'Sequential'}[/dim]")
            if stats['parallel_processing_enabled']:
                console.print(f"[dim]Max concurrent: {stats['max_concurrent_requests']}[/dim]")
        except Exception as e:
            logger.debug(f"Could not get processing stats: {e}")
        
        # Process with LLM
        console.print("Processing with LLM...")
        
        # Show enhanced progress for parallel processing
        if stats and stats.get('parallel_processing_enabled'):
            # For parallel processing, the progress will be handled by the ProgressTracker
            consolidated_content = processor.consolidate_documentation(scraped_docs)
        else:
            # For sequential processing, show simple status
            with console.status("[bold blue]Consolidating documentation..."):
                consolidated_content = processor.consolidate_documentation(scraped_docs)
        
        # Generate output filename
        output_file = urls_file.with_suffix('.md')
        
        # Save output
        console.print(f"Saving to: {output_file}")
        file_handler.write_markdown_file(output_file, consolidated_content)
        
        console.print(f"[green]+ Documentation processing complete![/green]")
        console.print(f"[green]Output saved to: {output_file}[/green]")
        
        # Show final parallel processing statistics if available
        if stats and stats.get('parallel_processing_enabled'):
            try:
                parallel_stats = processor.get_parallel_processing_stats()
                if parallel_stats:
                    processing_stats = parallel_stats.get('processing', {})
                    console.print(f"[dim]Final statistics: {processing_stats.get('completed', 0)} completed, "
                                f"{processing_stats.get('failed', 0)} failed, "
                                f"{processing_stats.get('success_rate', 0):.1%} success rate[/dim]")
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