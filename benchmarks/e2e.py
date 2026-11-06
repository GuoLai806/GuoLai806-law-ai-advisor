import hashlib
import json
import os
from pathlib import Path
import time

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def source_hashes():
    root = Path(__file__).resolve().parents[1]
    result={str(path.relative_to(root)): file_hash(path)
            for folder in ('evidence_rag', 'benchmarks')
            for path in sorted((root / folder).glob('*.py'))}
    for path in (root/'pyproject.toml',root/'evidence_rag/static/index.html'):
        result[str(path.relative_to(root))]=file_hash(path)
    return result

def save_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())

class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', encoding='utf-8')

    def record(self, data):
        self.stream.write(json.dumps(data, ensure_ascii=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
