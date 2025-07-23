"""
Basic validation script for the optimized parallel architecture.
ASCII-only version for Windows compatibility.
"""
import sys
import os
import asyncio
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_validation():
    """Run validation tests."""
    print("Starting validation tests...")
    
    results = {
        "timestamp": time.time(),
        "tests": {},
        "overall_status": "unknown"
    }
    
    # Test imports
    try:
        from src.crawl4ai_llm_docs.config.models import AppConfig
        from src.crawl4ai_llm_docs.core.intelligent_chunker import IntelligentChunker
        from src.crawl4ai_llm_docs.core.adaptive_rate_limiter import AdaptiveRateLimiter
        results["tests"]["imports"] = {"status": "passed", "message": "All imports successful"}
        print("PASS - Import test")
    except Exception as e:
        results["tests"]["imports"] = {"status": "failed", "error": str(e)}
        print(f"FAIL - Import test: {e}")
        return results
    
    # Test configuration
    try:
        config = AppConfig.get_test_config()
        assert config.api_key is not None
        assert config.parallel_processing.max_concurrent_requests >= 1
        results["tests"]["configuration"] = {"status": "passed", "message": "Configuration valid"}
        print("PASS - Configuration test")
    except Exception as e:
        results["tests"]["configuration"] = {"status": "failed", "error": str(e)}
        print(f"FAIL - Configuration test: {e}")
    
    # Test intelligent chunker
    try:
        chunker = IntelligentChunker(config)
        
        # Create dummy document
        from src.crawl4ai_llm_docs.core.scraper import ScrapedDocument
        test_content = "This is a test document with some content that should be chunked properly."
        test_doc = ScrapedDocument(
            url="https://test.example.com",
            title="Test Document",
            content=test_content,
            markdown="# Test\n" + test_content,
            success=True
        )
        
        chunks = chunker.create_chunks([test_doc])
        stats = chunker.get_chunking_stats(chunks)
        
        assert len(chunks) > 0
        assert stats["total_chunks"] > 0
        
        results["tests"]["intelligent_chunker"] = {
            "status": "passed", 
            "chunks_created": len(chunks),
            "efficiency_gain": stats["efficiency_gain"]
        }
        print(f"PASS - Intelligent chunker test - {len(chunks)} chunks created")
    except Exception as e:
        results["tests"]["intelligent_chunker"] = {"status": "failed", "error": str(e)}
        print(f"FAIL - Intelligent chunker test: {e}")
    
    # Test rate limiter
    try:
        async def test_rate_limiter():
            rate_limiter = AdaptiveRateLimiter(config.parallel_processing)
            await rate_limiter.acquire()
            await rate_limiter.acquire()
            
            # Test response handling
            rate_limiter.on_response({}, 0.5)
            status = rate_limiter.get_rate_limit_status()
            
            return {
                "status": "passed",
                "current_delay": status["current_delay"],
                "error_count": status["error_count"]
            }
        
        rate_result = asyncio.run(test_rate_limiter())
        results["tests"]["adaptive_rate_limiter"] = rate_result
        print("PASS - Adaptive rate limiter test")
    except Exception as e:
        results["tests"]["adaptive_rate_limiter"] = {"status": "failed", "error": str(e)}
        print(f"FAIL - Adaptive rate limiter test: {e}")
    
    # Test progress tracker
    try:
        from src.crawl4ai_llm_docs.core.progress_tracker import ProgressTracker
        
        async def test_progress_tracker():
            tracker = ProgressTracker(config.parallel_processing)
            await tracker.start_tracking(5, "Test tracking")
            await tracker.update_progress(completed=3, failed=1)
            final_metrics = await tracker.stop_tracking()
            
            return {
                "status": "passed",
                "completed_items": final_metrics.completed_items,
                "failed_items": final_metrics.failed_items
            }
        
        progress_result = asyncio.run(test_progress_tracker())
        results["tests"]["progress_tracker"] = progress_result
        print("PASS - Progress tracker test")
    except Exception as e:
        results["tests"]["progress_tracker"] = {"status": "failed", "error": str(e)}
        print(f"FAIL - Progress tracker test: {e}")
    
    # Test content preservation processor
    try:
        from src.crawl4ai_llm_docs.core.content_preservation_processor import ContentPreservationProcessor
        
        # Skip API-dependent test for now
        processor = ContentPreservationProcessor(config)
        stats = processor.get_processing_statistics()
        
        results["tests"]["content_preservation"] = {
            "status": "passed",
            "initialized": True,
            "stats_available": True
        }
        print("PASS - Content preservation processor test")
    except Exception as e:
        results["tests"]["content_preservation"] = {"status": "failed", "error": str(e)}
        print(f"FAIL - Content preservation processor test: {e}")
    
    # Calculate overall status
    passed_tests = sum(1 for test in results["tests"].values() if test.get("status") == "passed")
    total_tests = len(results["tests"])
    
    results["summary"] = {
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "pass_rate": passed_tests / total_tests if total_tests > 0 else 0
    }
    
    if passed_tests == total_tests:
        results["overall_status"] = "passed"
        print(f"\nSUCCESS: All {total_tests} tests passed!")
    elif passed_tests >= total_tests * 0.8:  # 80% pass rate
        results["overall_status"] = "mostly_passed"
        print(f"\nWARNING: {passed_tests}/{total_tests} tests passed (mostly passed)")
    else:
        results["overall_status"] = "failed"
        print(f"\nFAILED: Only {passed_tests}/{total_tests} tests passed")
    
    # Save results
    output_file = Path("validation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_validation()
        
        # Exit with appropriate code
        if results["overall_status"] == "passed":
            sys.exit(0)
        elif results["overall_status"] == "mostly_passed":
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        print(f"Validation error: {e}")
        sys.exit(3)