"""Compatibility alias; implementation lives in evidence_rag.viking_index."""
import sys
from evidence_rag import viking_index as _implementation
if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
