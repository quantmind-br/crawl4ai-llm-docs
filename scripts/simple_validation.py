"""
Simple validation script for the optimized parallel architecture.
Tests core functionality without Unicode issues.
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
        print("✓ Import test passed")
    except Exception as e:
        results["tests"]["imports"] = {"status": "failed", "error": str(e)}
        print(f"✗ Import test failed: {e}")
        return results
    
    # Test configuration
    try:
        config = AppConfig.get_test_config()
        assert config.api_key is not None
        assert config.parallel_processing.max_concurrent_requests >= 1
        results["tests"]["configuration"] = {"status": "passed", "message": "Configuration valid"}
        print("✓ Configuration test passed")
    except Exception as e:
        results["tests"]["configuration"] = {"status": "failed", "error": str(e)}
        print(f"✗ Configuration test failed: {e}")
    
    # Test intelligent chunker
    try:
        chunker = IntelligentChunker(config)
        
        # Create dummy document
        from src.crawl4ai_llm_docs.core.scraper import ScrapedDocument
        test_doc = ScrapedDocument(
            url="https://test.example.com",
            title="Test Document",
            markdown="# Test\nThis is a test document with some content.",
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
        print(f"✓ Intelligent chunker test passed - {len(chunks)} chunks created")
    except Exception as e:
        results["tests"]["intelligent_chunker"] = {"status": "failed", "error": str(e)}
        print(f"✗ Intelligent chunker test failed: {e}")
    
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
        print("✓ Adaptive rate limiter test passed")
    except Exception as e:
        results["tests"]["adaptive_rate_limiter"] = {"status": "failed", "error": str(e)}
        print(f"✗ Adaptive rate limiter test failed: {e}")
    
    # Test content preservation processor
    try:
        from src.crawl4ai_llm_docs.core.content_preservation_processor import ContentPreservationProcessor
        
        async def test_preservation():
            processor = ContentPreservationProcessor(config)
            
            test_content = """
            # API Documentation
            ```javascript
            const api = require('api');
            api.getData();
            ```
            Configuration: apiKey, baseUrl
            """
            
            preservation_test = await processor.test_content_preservation(test_content)
            return {
                "status": "passed",
                "preservation_ratio": preservation_test["preservation_ratio"],
                "processing_time": preservation_test["processing_time"]
            }
        
        preservation_result = asyncio.run(test_preservation())
        results["tests"]["content_preservation"] = preservation_result
        print(f"✓ Content preservation test passed - {preservation_result['preservation_ratio']:.2%} preserved")
    except Exception as e:
        results["tests"]["content_preservation"] = {"status": "failed", "error": str(e)}
        print(f"✗ Content preservation test failed: {e}")
    
    # Calculate overall status
    passed_tests = sum(1 for test in results["tests"].values() if test.get("status") == "passed")
    total_tests = len(results["tests"])
    
    if passed_tests == total_tests:
        results["overall_status"] = "passed"
        print(f"\n✓ All {total_tests} tests passed!")
    elif passed_tests >= total_tests * 0.8:  # 80% pass rate
        results["overall_status"] = "mostly_passed"
        print(f"\n⚠ {passed_tests}/{total_tests} tests passed (mostly passed)")
    else:
        results["overall_status"] = "failed"
        print(f"\n✗ Only {passed_tests}/{total_tests} tests passed")
    
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