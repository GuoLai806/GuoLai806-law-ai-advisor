"""Legacy launch path routed to the shared, evidence-first legal application.

The historical implementation is available in Git history. This compatibility
entry does not initialize the former global agent or its ungrounded RAG path.
"""
from pathlib import Path
import sys

# Supports running from main-system/ before the project is installed.
_repository_root = str(Path(__file__).resolve().parent.parent)
if _repository_root not in sys.path:
    sys.path.insert(0, _repository_root)

from legal_rag.api import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
