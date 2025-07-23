"""Cross-platform file handling utilities."""
import logging
import re
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

from ..exceptions import FileOperationError, ValidationError


logger = logging.getLogger(__name__)


class FileHandler:
    """Handles file operations with proper error handling and validation."""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Validate if a string is a valid URL.
        
        Args:
            url: URL string to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False
    
    def read_urls_file(self, file_path: Path) -> List[str]:
        """Read and validate URLs from a text file.
        
        Args:
            file_path: Path to the URLs file
            
        Returns:
            List of valid URLs
            
        Raises:
            FileOperationError: If file cannot be read
            ValidationError: If no valid URLs found
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                raise FileOperationError(f"File does not exist: {file_path}")
            
            if not file_path.is_file():
                raise FileOperationError(f"Path is not a file: {file_path}")
            
            # Read file content
            try:
                content = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Try with different encodings
                try:
                    content = file_path.read_text(encoding='latin-1')
                    logger.warning("File read with latin-1 encoding")
                except UnicodeDecodeError:
                    raise FileOperationError("Cannot decode file. Please ensure UTF-8 encoding.")
            
            # Extract URLs from content
            urls = []
            invalid_urls = []
            
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Validate URL
                if self.is_valid_url(line):
                    urls.append(line)
                    logger.debug(f"Valid URL on line {line_num}: {line}")
                else:
                    invalid_urls.append((line_num, line))
                    logger.warning(f"Invalid URL on line {line_num}: {line}")
            
            # Check results
            if invalid_urls:
                error_msg = f"Found {len(invalid_urls)} invalid URLs:\n"
                for line_num, url in invalid_urls[:5]:  # Show first 5 errors
                    error_msg += f"  Line {line_num}: {url}\n"
                if len(invalid_urls) > 5:
                    error_msg += f"  ... and {len(invalid_urls) - 5} more"
                
                logger.error(error_msg)
                
                # If no valid URLs found, raise error
                if not urls:
                    raise ValidationError(f"No valid URLs found in file: {file_path}")
            
            if not urls:
                raise ValidationError(f"No URLs found in file: {file_path}")
            
            logger.info(f"Successfully read {len(urls)} valid URLs from {file_path}")
            return urls
            
        except (IOError, OSError) as e:
            raise FileOperationError(f"Cannot read file {file_path}: {e}")
    
    def write_markdown_file(self, file_path: Path, content: str) -> None:
        """Write markdown content to file with atomic operation.
        
        Args:
            file_path: Path to output file
            content: Markdown content to write
            
        Raises:
            FileOperationError: If file cannot be written
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write operation
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
            
            try:
                temp_path.write_text(content, encoding='utf-8')
                
                # Atomic replace (works on all platforms)
                import os
                if os.name == 'nt' and file_path.exists():  # Windows
                    file_path.unlink()
                    
                temp_path.replace(file_path)
                
                logger.info(f"Successfully wrote {len(content)} characters to {file_path}")
                
            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise e
                
        except PermissionError as e:
            raise FileOperationError(f"No write permission for {file_path}: {e}")
        except OSError as e:
            raise FileOperationError(f"Cannot write file {file_path}: {e}")
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate and normalize file path.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid
        """
        try:
            path = Path(file_path).expanduser().resolve()
            
            # Check if path is absolute (after resolution)
            if not path.is_absolute():
                raise ValidationError(f"Path must be absolute: {file_path}")
            
            return path
            
        except (OSError, ValueError) as e:
            raise ValidationError(f"Invalid file path: {file_path} - {e}")
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for cross-platform compatibility.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace invalid characters
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, '_', filename)
        
        # Trim and handle edge cases
        sanitized = sanitized.strip('. ')
        
        # Ensure not empty
        if not sanitized:
            sanitized = "output"
        
        # Limit length (255 chars is typical filesystem limit)
        if len(sanitized) > 200:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            max_name_len = 200 - len(ext) - 1
            sanitized = name[:max_name_len] + ('.' + ext if ext else '')
        
        return sanitized
    
    def get_file_info(self, file_path: Path) -> dict:
        """Get file information for debugging.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            Dictionary with file information
        """
        try:
            if not file_path.exists():
                return {"exists": False, "error": "File does not exist"}
            
            stat = file_path.stat()
            
            return {
                "exists": True,
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir(),
                "size_bytes": stat.st_size,
                "readable": file_path.is_file() and stat.st_size > 0,
                "absolute_path": str(file_path.absolute()),
                "parent_exists": file_path.parent.exists(),
                "parent_writable": file_path.parent.is_dir()
            }
            
        except Exception as e:
            return {"exists": True, "error": str(e)}