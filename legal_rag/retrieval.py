"""Legal-profile compatibility over the shared evidence retrieval engine."""
from evidence_rag.retrieval import *
from evidence_rag.retrieval import RetrievalService as _RetrievalService
class RetrievalService(_RetrievalService):
    def __init__(self, catalog, dense=None, reranker=None, tokenizer=None, profile='legal'):
        super().__init__(catalog, dense, reranker, tokenizer, profile=profile)
