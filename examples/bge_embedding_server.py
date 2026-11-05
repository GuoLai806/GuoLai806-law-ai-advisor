"""Loopback-only development embedding endpoint used with OpenViking.

Download the pinned public model first. Start explicitly from the repository:
  HF_HUB_OFFLINE=1 uvicorn --app-dir examples bge_embedding_server:app \
      --host 127.0.0.1 --port 19344
No weights, credentials, user inputs or private paths are bundled here.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_ID = 'BAAI/bge-m3'
REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
started = time.monotonic()
requested_device = os.environ.get('EMBEDDING_DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu')
if requested_device not in ('mps', 'cpu'):
    raise ValueError('This development example supports explicit mps or cpu')
if requested_device == 'mps' and not torch.backends.mps.is_available():
    raise RuntimeError('MPS unavailable; do not silently fall back')
model = SentenceTransformer(MODEL_ID, revision=REVISION, local_files_only=True, device=requested_device)
if requested_device == 'mps':
    model.half()
if int(model.get_embedding_dimension()) != 1024:
    raise RuntimeError('Pinned embedding dimension mismatch')
identity = {'model': MODEL_ID, 'revision': REVISION, 'dimension': 1024,
            'device': str(model.device), 'dtype': str(next(model.parameters()).dtype),
            'load_seconds': time.monotonic()-started, 'torch': torch.__version__}
app = FastAPI(title='Pinned BGE development embedding endpoint')
lock = asyncio.Lock()

class EmbeddingRequest(BaseModel):
    model: str = MODEL_ID
    input: str | list[str]
    encoding_format: str = 'float'

@app.get('/health')
async def health():
    return {'ready': True, **identity}

@app.get('/v1/models')
async def models():
    return {'data': [{'id': MODEL_ID, 'object': 'model'}]}

@app.post('/v1/embeddings')
async def embeddings(request: EmbeddingRequest):
    if request.model != MODEL_ID or request.encoding_format != 'float':
        raise HTTPException(422, 'unsupported model/format')
    values = [request.input] if isinstance(request.input, str) else request.input
    if not values or len(values) > 64:
        raise HTTPException(422, 'batch must contain 1..64 inputs')
    counts = [len(model.tokenizer.encode(x, add_special_tokens=True, truncation=False)) for x in values]
    if max(counts) > 8192:
        raise HTTPException(422, 'whole input exceeds pinned model limit')
    started = time.monotonic()
    async with lock:
        vectors = await asyncio.to_thread(model.encode, values, batch_size=8,
                    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, prompt='')
    rows = vectors.astype('float32').tolist()
    # Optional owner-selected local evidence file; never store input plaintext.
    record_path = os.environ.get('EVIDENCE_RAG_EMBEDDING_LOG')
    if record_path:
        record = {'at': time.time(), 'input_hashes': [hashlib.sha256(x.encode()).hexdigest() for x in values],
                  'count': len(rows), 'tokens': sum(counts), 'latency_ms': (time.monotonic()-started)*1000,
                  'model': MODEL_ID, 'revision': REVISION}
        with Path(record_path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
    return {'object': 'list', 'model': MODEL_ID,
            'data': [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(rows)],
            'usage': {'prompt_tokens': sum(counts), 'total_tokens': sum(counts)}}
