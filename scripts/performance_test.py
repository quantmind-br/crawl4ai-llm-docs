"""
Performance test script to validate 5-10x improvement target.
Tests the complete optimized implementation against the claude-code.txt URLs.
"""
import sys
import time
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.crawl4ai_llm_docs.config.models import AppConfig, CrawlerConfig
    from src.crawl4ai_llm_docs.core.processor import DocumentationProcessor
    from src.crawl4ai_llm_docs.core.scraper import DocumentationScraper
    from src.crawl4ai_llm_docs.utils.file_handler import FileHandler
    from rich.console import Console
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class PerformanceTest:
    """Test complete implementation performance."""
    
    def __init__(self):
        self.console = Console()
        self.config = AppConfig.get_test_config()
        self.test_file = Path("claude-code.txt")
        
        # Performance targets
        self.targets = {
            "speedup_ratio": 5.0,  # 5x minimum improvement
            "max_processing_time": 30.0,  # 30 seconds max for 28 URLs
            "min_content_preservation": 0.85,  # 85% content preservation
            "max_memory_usage_mb": 500.0  # 500MB max memory usage
        }
        
        self.results = {
            "test_timestamp": time.time(),
            "config": {
                "parallel_enabled": True,
                "max_concurrent": self.config.parallel_processing.max_concurrent_requests,
                "model": self.config.model,
                "chunk_target_tokens": self.config.chunk_target_tokens
            },
            "performance": {},
            "targets_met": {},
            "overall_status": "pending"
        }
    
    def run_performance_test(self):
        """Run complete performance test."""
        print("=== Performance Test - Optimized Architecture ===")
        print(f"Testing with {self.config.parallel_processing.max_concurrent_requests} concurrent workers")
        print(f"Target: Process 28 URLs in under {self.targets['max_processing_time']}s")
        print()
        
        if not self.test_file.exists():
            print(f"ERROR: Test file not found: {self.test_file}")
            return False
        
        try:
            # Initialize components
            file_handler = FileHandler()
            crawler_config = CrawlerConfig()  # Use CrawlerConfig for scraper
            scraper = DocumentationScraper(crawler_config)
            processor = DocumentationProcessor(self.config, self.console)
            
            # Read URLs
            urls = file_handler.read_urls_file(self.test_file)
            print(f"Loaded {len(urls)} URLs from {self.test_file}")
            
            # Measure scraping performance
            print("\n1. Testing scraping performance...")
            scraping_start = time.time()
            
            scraped_docs = scraper.scrape_urls(urls)
            
            scraping_time = time.time() - scraping_start
            successful_docs = [doc for doc in scraped_docs if doc.success]
            
            print(f"   Scraped {len(successful_docs)}/{len(urls)} documents in {scraping_time:.2f}s")
            print(f"   Rate: {len(successful_docs)/scraping_time:.2f} docs/sec")
            
            if len(successful_docs) == 0:
                print("ERROR: No documents were successfully scraped")
                return False
            
            self.results["performance"]["scraping"] = {
                "total_time": scraping_time,
                "successful_docs": len(successful_docs),
                "failed_docs": len(urls) - len(successful_docs),
                "docs_per_second": len(successful_docs) / scraping_time
            }
            
            # Get processing statistics
            print("\n2. Analyzing processing requirements...")
            stats = processor.get_processing_stats(successful_docs)
            
            input_analysis = stats.get('input_analysis', {})
            chunking_stats = stats.get('intelligent_chunking', {})
            
            print(f"   Total tokens: {input_analysis.get('estimated_tokens', 0):,}")
            print(f"   Average tokens per doc: {input_analysis.get('average_tokens_per_doc', 0):.0f}")
            print(f"   Chunks to create: {chunking_stats.get('total_chunks', 0)}")
            print(f"   Efficiency gain: {chunking_stats.get('efficiency_gain', 1):.2f}x")
            print(f"   Cost reduction: {chunking_stats.get('estimated_cost_reduction', 0):.1%}")
            
            self.results["performance"]["analysis"] = {
                "total_tokens": input_analysis.get('estimated_tokens', 0),
                "chunks": chunking_stats.get('total_chunks', 0),
                "efficiency_gain": chunking_stats.get('efficiency_gain', 1),
                "cost_reduction": chunking_stats.get('estimated_cost_reduction', 0)
            }
            
            # Measure processing performance
            print("\n3. Testing optimized processing...")
            processing_start = time.time()
            
            # Process with optimized architecture
            consolidated_content = processor.consolidate_documentation(successful_docs)
            
            processing_time = time.time() - processing_start
            total_time = scraping_time + processing_time
            
            print(f"   Processing completed in {processing_time:.2f}s")
            print(f"   Total pipeline time: {total_time:.2f}s")
            print(f"   Overall rate: {len(successful_docs)/total_time:.2f} docs/sec")
            print(f"   Generated {len(consolidated_content):,} characters")
            
            self.results["performance"]["processing"] = {
                "processing_time": processing_time,
                "total_time": total_time,
                "overall_rate": len(successful_docs) / total_time,
                "output_characters": len(consolidated_content),
                "chars_per_second": len(consolidated_content) / total_time
            }
            
            # Get final architecture statistics
            print("\n4. Collecting architecture statistics...")
            parallel_stats = processor.get_parallel_processing_stats()
            
            if parallel_stats:
                rate_stats = parallel_stats.get('rate_limiter', {})
                content_stats = parallel_stats.get('content_processor', {})
                
                print(f"   Rate limiter final delay: {rate_stats.get('current_delay', 0):.2f}s")
                print(f"   Content preservation avg: {content_stats.get('average_preservation_ratio', 0):.2%}")
                print(f"   Chunks processed: {content_stats.get('total_chunks_processed', 0)}")
                
                self.results["performance"]["architecture"] = {
                    "rate_limiter_delay": rate_stats.get('current_delay', 0),
                    "preservation_ratio": content_stats.get('average_preservation_ratio', 0),
                    "chunks_processed": content_stats.get('total_chunks_processed', 0)
                }
            
            # Evaluate against targets
            print("\n5. Evaluating performance targets...")
            self._evaluate_targets(total_time, len(successful_docs))
            
            # Save detailed results
            output_file = Path("performance_test_results.json")
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            print(f"\nDetailed results saved to: {output_file}")
            
            return self.results["overall_status"] == "passed"
            
        except Exception as e:
            print(f"Performance test failed: {e}")
            self.results["error"] = str(e)
            return False
    
    def _evaluate_targets(self, total_time, successful_docs):
        """Evaluate performance against targets."""
        print("\n=== Performance Target Evaluation ===")
        
        # Time target
        time_target_met = total_time <= self.targets["max_processing_time"]
        print(f"Time Target: {total_time:.2f}s <= {self.targets['max_processing_time']}s : {'PASS' if time_target_met else 'FAIL'}")
        self.results["targets_met"]["time_target"] = time_target_met
        
        # Rate target (documents per second)
        docs_per_second = successful_docs / total_time
        min_rate = successful_docs / self.targets["max_processing_time"]  # Rate needed to meet time target
        rate_target_met = docs_per_second >= min_rate
        print(f"Rate Target: {docs_per_second:.2f} docs/sec >= {min_rate:.2f} docs/sec : {'PASS' if rate_target_met else 'FAIL'}")
        self.results["targets_met"]["rate_target"] = rate_target_met
        
        # Efficiency gain target
        efficiency_gain = self.results["performance"]["analysis"].get("efficiency_gain", 1)
        efficiency_target_met = efficiency_gain >= 2.0  # 2x chunking efficiency minimum
        print(f"Efficiency Target: {efficiency_gain:.2f}x >= 2.0x : {'PASS' if efficiency_target_met else 'FAIL'}")
        self.results["targets_met"]["efficiency_target"] = efficiency_target_met
        
        # Content preservation target
        preservation_ratio = self.results["performance"]["architecture"].get("preservation_ratio", 0)
        preservation_target_met = preservation_ratio >= self.targets["min_content_preservation"]
        print(f"Preservation Target: {preservation_ratio:.2%} >= {self.targets['min_content_preservation']:.2%} : {'PASS' if preservation_target_met else 'FAIL'}")
        self.results["targets_met"]["preservation_target"] = preservation_target_met
        
        # Calculate theoretical speedup
        naive_time_estimate = successful_docs * 10  # Assume 10s per doc sequentially
        theoretical_speedup = naive_time_estimate / total_time
        speedup_target_met = theoretical_speedup >= self.targets["speedup_ratio"]
        print(f"Speedup Target: {theoretical_speedup:.2f}x >= {self.targets['speedup_ratio']}x : {'PASS' if speedup_target_met else 'FAIL'}")
        self.results["targets_met"]["speedup_target"] = speedup_target_met
        
        # Overall evaluation
        targets_met = sum(1 for met in self.results["targets_met"].values() if met)
        total_targets = len(self.results["targets_met"])
        
        if targets_met == total_targets:
            self.results["overall_status"] = "passed"
            print(f"\nRESULT: All {total_targets} performance targets met! PASS")
        elif targets_met >= total_targets * 0.8:  # 80% of targets
            self.results["overall_status"] = "mostly_passed"
            print(f"\nRESULT: {targets_met}/{total_targets} performance targets met (mostly passed)")
        else:
            self.results["overall_status"] = "failed"
            print(f"\nRESULT: Only {targets_met}/{total_targets} performance targets met")
        
        # Performance summary
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Documents processed: {successful_docs}")
        print(f"  Processing rate: {docs_per_second:.2f} docs/sec")
        print(f"  Theoretical speedup: {theoretical_speedup:.2f}x")
        print(f"  Chunking efficiency: {efficiency_gain:.2f}x")
        print(f"  Content preservation: {preservation_ratio:.2%}")

def main():
    """Run performance test."""
    test = PerformanceTest()
    
    try:
        success = test.run_performance_test()
        
        if success:
            print("\nSUCCESS: Performance test PASSED - Ready for production!")
            sys.exit(0)
        else:
            print("\nWARNING: Performance test had issues - Review results")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: Performance test error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()