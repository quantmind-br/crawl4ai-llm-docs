"""Documentation scraper using Crawl4AI with anti-detection features."""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

try:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode, PruningContentFilter
    CRAWL4AI_AVAILABLE = True
except ImportError as e:
    # Handle import error gracefully for development
    logging.warning(f"Crawl4AI not installed: {e}. Please install with: uv pip install crawl4ai>=0.7.0")
    AsyncWebCrawler = None
    CrawlerRunConfig = None
    BrowserConfig = None
    CacheMode = None
    PruningContentFilter = None
    CRAWL4AI_AVAILABLE = False

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.models import CrawlerConfig
from ..exceptions import ScrapingError


logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    """Represents a scraped document with metadata."""
    url: str
    title: str
    content: str
    markdown: str
    success: bool
    error: Optional[str] = None
    scraped_at: Optional[float] = None
    word_count: int = 0


class DocumentationScraper:
    """Documentation scraper with anti-detection capabilities."""
    
    def __init__(self, config: CrawlerConfig):
        """Initialize scraper with configuration.
        
        Args:
            config: Crawler configuration
        """
        self.config = config
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """Validate that Crawl4AI dependencies are available."""
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "Crawl4AI is not installed. Please install with: uv pip install crawl4ai>=0.7.0"
            )
    
    def _create_browser_config(self) -> 'BrowserConfig':
        """Create browser configuration with anti-detection settings.
        
        Returns:
            BrowserConfig instance
        """
        return BrowserConfig(
            user_agent_mode=self.config.user_agent_mode,
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
            headless=self.config.headless,
            use_persistent_context=self.config.use_persistent_context
        )
    
    def _create_crawler_config(self) -> 'CrawlerRunConfig':
        """Create crawler configuration for documentation extraction.
        
        Returns:
            CrawlerRunConfig instance
        """
        return CrawlerRunConfig(
            word_count_threshold=self.config.word_count_threshold,
            magic=self.config.magic_mode,  # Enable anti-detection
            cache_mode=CacheMode.ENABLED if self.config.enable_cache else CacheMode.BYPASS,
            verbose=True,
            wait_for_images=False,
            exclude_external_images=True,
            exclude_all_images=True  # Para documentação, não precisamos de imagens
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _scrape_single_url(
        self,
        crawler: 'AsyncWebCrawler',
        url: str,
        config: 'CrawlerRunConfig'
    ) -> ScrapedDocument:
        """Scrape a single URL with retry logic.
        
        Args:
            crawler: AsyncWebCrawler instance
            url: URL to scrape
            config: Crawler configuration
            
        Returns:
            ScrapedDocument instance
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Scraping URL: {url}")
            
            result = await crawler.arun(url, config=config)
            
            if result.success:
                # Extract markdown content safely
                markdown_content = ''
                try:
                    if hasattr(result, 'markdown') and result.markdown:
                        if hasattr(result.markdown, 'fit_markdown') and result.markdown.fit_markdown:
                            markdown_content = result.markdown.fit_markdown
                        elif isinstance(result.markdown, str):
                            markdown_content = result.markdown
                        else:
                            markdown_content = str(result.markdown)
                    
                    # Fallback to other content if markdown is empty
                    if not markdown_content and hasattr(result, 'cleaned_html') and result.cleaned_html:
                        markdown_content = result.cleaned_html
                    
                    # Last resort: use raw content
                    if not markdown_content and hasattr(result, 'html') and result.html:
                        markdown_content = result.html
                        
                except Exception as e:
                    logger.warning(f"Failed to extract markdown from {url}: {e}")
                    # Try fallback methods
                    if hasattr(result, 'cleaned_html') and result.cleaned_html:
                        markdown_content = result.cleaned_html
                    elif hasattr(result, 'html') and result.html:
                        markdown_content = result.html
                
                # Count words in content
                word_count = len(markdown_content.split()) if markdown_content else 0
                
                scraped_doc = ScrapedDocument(
                    url=url,
                    title=result.metadata.get('title', 'Untitled') if result.metadata else 'Untitled',
                    content=result.cleaned_html or '',
                    markdown=markdown_content,
                    success=True,
                    scraped_at=start_time,
                    word_count=word_count
                )
                
                logger.info(f"Successfully scraped {url} ({word_count} words)")
                return scraped_doc
            else:
                error_msg = result.error_message or "Unknown scraping error"
                logger.warning(f"Failed to scrape {url}: {error_msg}")
                
                return ScrapedDocument(
                    url=url,
                    title="",
                    content="",
                    markdown="",
                    success=False,
                    error=error_msg,
                    scraped_at=start_time
                )
                
        except Exception as e:
            error_msg = f"Exception scraping {url}: {str(e)}"
            logger.error(error_msg)
            
            return ScrapedDocument(
                url=url,
                title="",
                content="",
                markdown="",
                success=False,
                error=error_msg,
                scraped_at=start_time
            )
    
    async def _scrape_urls_async(self, urls: List[str]) -> List[ScrapedDocument]:
        """Scrape multiple URLs asynchronously.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of ScrapedDocument instances
        """
        browser_config = self._create_browser_config()
        crawler_config = self._create_crawler_config()
        
        results = []
        
        async with AsyncWebCrawler() as crawler:
            # Process URLs in batches to manage resources
            max_workers = getattr(self.config, 'max_workers', 4)
            batch_size = min(max_workers, len(urls), 4)  # Max 4 concurrent requests
            
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_urls)} URLs")
                
                # Create tasks for batch
                tasks = [
                    self._scrape_single_url(crawler, url, crawler_config)
                    for url in batch_urls
                ]
                
                # Execute batch with delay between requests
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and handle exceptions
                for url, result in zip(batch_urls, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Exception processing {url}: {result}")
                        results.append(ScrapedDocument(
                            url=url,
                            title="",
                            content="",
                            markdown="",
                            success=False,
                            error=str(result)
                        ))
                    else:
                        results.append(result)
                
                # Add delay between batches to be respectful
                if i + batch_size < len(urls):
                    await asyncio.sleep(1.0)  # Fixed delay of 1 second
        
        return results
    
    def scrape_urls(self, urls: List[str]) -> List[ScrapedDocument]:
        """Scrape multiple URLs and return results.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of ScrapedDocument instances
            
        Raises:
            ScrapingError: If scraping fails completely
        """
        if not urls:
            logger.warning("No URLs provided for scraping")
            return []
        
        logger.info(f"Starting to scrape {len(urls)} URLs")
        
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to handle differently
                import nest_asyncio
                nest_asyncio.apply()
                results = asyncio.run(self._scrape_urls_async(urls))
            except RuntimeError:
                # No running loop, safe to use asyncio.run
                results = asyncio.run(self._scrape_urls_async(urls))
            
            # Analyze results
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            logger.info(f"Scraping completed: {len(successful)} successful, {len(failed)} failed")
            
            # Log failed URLs for debugging
            if failed:
                logger.warning("Failed URLs:")
                for doc in failed:
                    logger.warning(f"  {doc.url}: {doc.error}")
            
            # Check if we have any successful results
            if not successful:
                raise ScrapingError(f"All {len(urls)} URLs failed to scrape")
            
            # Log success statistics
            total_words = sum(doc.word_count for doc in successful)
            logger.info(f"Successfully scraped {total_words} words from {len(successful)} documents")
            
            return results
            
        except Exception as e:
            error_msg = f"Scraping failed: {str(e)}"
            logger.error(error_msg)
            raise ScrapingError(error_msg)

    async def scrape_urls_async(self, urls: List[str]) -> List[ScrapedDocument]:
        """Async version of scrape_urls for use within async contexts.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of ScrapedDocument instances
            
        Raises:
            ScrapingError: If scraping fails completely
        """
        if not urls:
            logger.warning("No URLs provided for scraping")
            return []
        
        logger.info(f"Starting to scrape {len(urls)} URLs")
        
        try:
            # Run async scraping directly
            results = await self._scrape_urls_async(urls)
            
            # Analyze results
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            logger.info(f"Scraping completed: {len(successful)} successful, {len(failed)} failed")
            
            # Log failed URLs for debugging
            if failed:
                logger.warning("Failed URLs:")
                for doc in failed:
                    logger.warning(f"  {doc.url}: {doc.error}")
            
            # Check if we have any successful results
            if not successful:
                raise ScrapingError(f"All {len(urls)} URLs failed to scrape")
            
            # Log success statistics
            total_words = sum(doc.word_count for doc in successful)
            logger.info(f"Successfully scraped {total_words} words from {len(successful)} documents")
            
            return results
            
        except Exception as e:
            error_msg = f"Scraping failed: {str(e)}"
            logger.error(error_msg)
            raise ScrapingError(error_msg)
    
    def get_scraping_stats(self, results: List[ScrapedDocument]) -> Dict[str, Any]:
        """Get statistics about scraping results.
        
        Args:
            results: List of scraping results
            
        Returns:
            Dictionary with statistics
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_words = sum(doc.word_count for doc in successful)
        
        stats = {
            "total_urls": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0,
            "total_words": total_words,
            "average_words": total_words / len(successful) if successful else 0,
            "failed_urls": [{"url": doc.url, "error": doc.error} for doc in failed]
        }
        
        return stats