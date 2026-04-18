import sys
import os
from pathlib import Path

# Add parent directory to Python path to enable backend imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from backend.utils.config import config

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )