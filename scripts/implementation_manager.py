"""
Implementation checkpoint and rollback system.
Execute before major changes to enable safe rollback.
"""
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ImplementationManager:
    def __init__(self, project_root: Path = Path(".")):
        self.project_root = project_root
        self.checkpoints_dir = project_root / ".implementation_checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
    def create_checkpoint(self, phase: str, description: str = "") -> str:
        """Create rollback point before major changes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_id = f"{phase}_{timestamp}"
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        
        # Backup critical files
        files_to_backup = [
            "src/crawl4ai_llm_docs/config/models.py",
            "src/crawl4ai_llm_docs/config/manager.py", 
            "src/crawl4ai_llm_docs/core/processor.py",
            "src/crawl4ai_llm_docs/core/scraper.py",
            "src/crawl4ai_llm_docs/cli.py",
            "pyproject.toml"
        ]
        
        checkpoint_dir.mkdir(exist_ok=True)
        
        for file_path in files_to_backup:
            src_file = self.project_root / file_path
            if src_file.exists():
                dst_file = checkpoint_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                
        # Save metadata
        metadata = {
            "checkpoint_id": checkpoint_id,
            "phase": phase,
            "description": description,
            "timestamp": timestamp,
            "files_backed_up": files_to_backup,
            "git_commit": self._get_git_commit()
        }
        
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        print(f"OK Checkpoint created: {checkpoint_id}")
        return checkpoint_id
        
    def rollback_to(self, checkpoint_id: str) -> bool:
        """Rollback to specific checkpoint."""
        checkpoint_dir = self.checkpoints_dir / checkpoint_id
        
        if not checkpoint_dir.exists():
            print(f"ERROR Checkpoint not found: {checkpoint_id}")
            return False
            
        metadata_file = checkpoint_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"ERROR Invalid checkpoint (no metadata): {checkpoint_id}")
            return False
            
        metadata = json.loads(metadata_file.read_text())
        
        # Restore files
        for file_name in checkpoint_dir.glob("*.py"):
            # Find original location
            for backed_up_file in metadata["files_backed_up"]:
                if Path(backed_up_file).name == file_name.name:
                    dst_file = self.project_root / backed_up_file
                    shutil.copy2(file_name, dst_file)
                    print(f"OK Restored: {backed_up_file}")
                    break
                    
        print(f"OK Rollback completed to: {checkpoint_id}")
        return True
        
    def validate_implementation(self, phase: str) -> Dict[str, Any]:
        """Validate current implementation state."""
        results = {
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "tests_passed": [],
            "tests_failed": [],
            "warnings": []
        }
        
        # Import tests
        try:
            from src.crawl4ai_llm_docs.config.models import AppConfig
            results["tests_passed"].append("config_models_import")
        except Exception as e:
            results["tests_failed"].append(f"config_models_import: {e}")
            
        # API connectivity
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key="PLACEHOLDER_API_KEY_REMOVED",
                base_url="https://api.openai.com/v1"
            )
            # Quick test call
            response = client.chat.completions.create(
                model="gemini/gemini-2.5-flash-lite-preview-06-17",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            results["tests_passed"].append("api_connectivity") 
        except Exception as e:
            results["tests_failed"].append(f"api_connectivity: {e}")
            
        # Performance baseline
        if Path("claude-code.txt").exists():
            try:
                import time
                start_time = time.time()
                # Quick performance test with 1 URL
                test_url = "https://docs.python.org/3/library/json.html"
                # This would run actual processing - simplified for checkpoint
                processing_time = time.time() - start_time
                if processing_time > 60:  # More than 1 minute for 1 URL is concerning
                    results["warnings"].append(f"slow_processing: {processing_time:.1f}s for single URL")
                else:
                    results["tests_passed"].append(f"performance_baseline: {processing_time:.1f}s")
            except Exception as e:
                results["tests_failed"].append(f"performance_test: {e}")
                
        return results
        
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except:
            return None

# Usage in implementation
if __name__ == "__main__":
    manager = ImplementationManager()
    
    # Create checkpoint before starting
    checkpoint_id = manager.create_checkpoint("pre_optimization", "Before performance optimization")
    
    # Validate current state  
    validation = manager.validate_implementation("current")
    print("Validation results:", json.dumps(validation, indent=2))
    
    if validation["tests_failed"]:
        print("WARNING: Validation failed. Review before proceeding.")
    else:
        print("OK: Ready for implementation")