"""Explicit loopback-only HTTP contract smoke; inputs are synthetic."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlparse
import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:19344')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    url = urlparse(args.base)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost', '::1') or url.username or url.password:
        raise ValueError('This synthetic probe is restricted to loopback HTTP')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Retain the existing report; use a new output')
    checks = {}
    report = {'kind': 'public BGE development endpoint synthetic HTTP smoke', 'checks': checks,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        with httpx.Client(timeout=90, trust_env=False) as client:
            health = client.get(args.base+'/health');health.raise_for_status()
            report['identity'] = health.json()
            checks['pinned_identity'] = health.json().get('revision') == '5617a9f61b028005a4858fdac845db406aefb181' and health.json().get('dimension') == 1024
            body = {'model': 'BAAI/bge-m3', 'input': ['A synthetic ocean source.', 'A synthetic forest source.']}
            result = client.post(args.base+'/v1/embeddings', json=body);result.raise_for_status()
            rows = result.json()['data']
            checks['two_finite_unit_vectors'] = len(rows)==2 and all(len(x['embedding'])==1024 and all(math.isfinite(v) for v in x['embedding']) and abs(math.sqrt(sum(v*v for v in x['embedding']))-1)<.02 for x in rows)
            checks['wrong_model_422'] = client.post(args.base+'/v1/embeddings', json={'model': 'not-the-model', 'input': 'synthetic'}).status_code==422
            checks['empty_batch_422'] = client.post(args.base+'/v1/embeddings', json={'input': []}).status_code==422
            # Reject before model.encode: the actual BGE tokenizer exceeds its 8192 limit.
            checks['oversized_input_422'] = client.post(args.base+'/v1/embeddings', json={'input': 'hello '*9000}).status_code==422
    except Exception as error:
        report['error'] = type(error).__name__+': '+str(error)
    report['passed'] = len(checks)==5 and all(checks.values())
    report['scope'] = 'synthetic contract/identity/bounds, not retrieval quality or a production capacity test'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'checks': checks}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
