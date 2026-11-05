"""Compatibility alias; implementation lives in evidence_rag.answer."""
import sys
from evidence_rag import answer as _implementation
if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
