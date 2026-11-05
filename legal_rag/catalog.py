"""Compatibility alias; implementation lives in evidence_rag.catalog."""
import sys
from evidence_rag import catalog as _implementation
if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
