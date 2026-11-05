"""Compatibility alias; implementation lives in evidence_rag.generate_benchmark."""
import sys
from evidence_rag import generate_benchmark as _implementation
if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
