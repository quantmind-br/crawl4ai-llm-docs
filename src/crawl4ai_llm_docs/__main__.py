"""Allow running with python -m crawl4ai_llm_docs."""
import os
import sys

# Fix Unicode encoding issues on Windows before any imports
if os.name == 'nt':  # Windows
    import locale
    import codecs
    
    # Set UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Reconfigure stdout and stderr to use UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        # Fallback for older Python versions
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

from .cli import main

if __name__ == '__main__':
    main()