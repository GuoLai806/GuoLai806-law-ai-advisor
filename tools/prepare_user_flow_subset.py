"""Freeze a small stratified SciFact protocol without changing data or gold."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.e2e import file_hash, save_json
from benchmarks.prepare import jsonl, qrels
from benchmarks.verification_metrics import LABELS, gold_label

SEED = '20261107'
SCOPE = 'Public stratified small evaluation; not hidden, unseen, or the full official test; sample membership frozen before model calls.'
RULE = 'Within each gold class take first N SHA256(seed + query_id UTF-8), lexical tie break; order selected IDs by the same key.'


def load_claims(path):
    result = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        identifier = item.get('id')
        if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
            raise ValueError('Invalid claim identifier')
        identifier = str(identifier)
        if not identifier or identifier in result:
            raise ValueError('Duplicate or empty claim identifier')
        if not isinstance(item.get('claim'), str):
            raise ValueError('Missing claim text')
        gold_label(item)
        result[identifier] = item
    return result


def stratified_ids(references, claims, per_label, seed=SEED):
    def key(identifier):
        return hashlib.sha256((seed + identifier).encode('utf-8')).hexdigest(), identifier
    if set(references) - set(claims):
        raise ValueError('Relevance annotations lack matching claims')
    groups = {label: [] for label in LABELS}
    for identifier in references:
        groups[gold_label(claims[identifier])].append(identifier)
    if any(len(identifiers) < per_label for identifiers in groups.values()):
        raise ValueError('Insufficient queries in one or more classes')
    selected = [identifier for label in LABELS
                for identifier in sorted(groups[label], key=key)[:per_label]]
    return sorted(selected, key=key)


def build_manifest(manifest_path, data_dir, claims_train, claims_test, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve the previously frozen subset manifest')
    original = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if original.get('dataset') != 'scifact' or 'short_evaluation' in original:
        raise ValueError('A full SciFact source manifest is required')
    data = Path(data_dir)
    queries = jsonl(data / 'queries.jsonl')
    corpus = jsonl(data / 'corpus.jsonl')
    for name, expected in [('queries.jsonl', original['queries_sha256']),
                           ('corpus.jsonl', original['corpus_sha256'])]:
        if file_hash(data / name) != expected:
            raise ValueError('Source data identity changed')
    if len(corpus) != original['corpus_count'] or len(queries) != original['query_count_total']:
        raise ValueError('Source data count changed')
    claims = {'train': load_claims(claims_train), 'test': load_claims(claims_test)}
    hashes = {'train': file_hash(claims_train), 'test': file_hash(claims_test)}
    references = {}
    for split in ('train', 'test'):
        path = data / 'qrels' / (split + '.tsv')
        references[split] = qrels(path)
        expected = original['qrels'][split]
        if file_hash(path) != expected['sha256'] or len(references[split]) != expected['qids']:
            raise ValueError('Source relevance identity/count changed')
        if set(references[split]) - set(queries) or set(references[split]) - set(claims[split]):
            raise ValueError('Source query/claim membership mismatch')
        if any(claims[split][identifier]['claim'] != queries[identifier]['text']
               for identifier in references[split]):
            raise ValueError('Source claim/query text mismatch')
        if any(not any(value > 0 for value in docs.values()) for docs in references[split].values()):
            raise ValueError('Query has no positive relevance annotation')
        if {identifier for docs in references[split].values() for identifier in docs} - set(corpus):
            raise ValueError('Unknown reference document')
    if set(references['train']) & set(references['test']):
        raise ValueError('Development/formal source splits overlap')
    result = deepcopy(original)
    result['version'] = 2
    result['original_manifest_sha256'] = file_hash(manifest_path)
    for field, split, per_label in [('validation', 'train', 2), ('short_evaluation', 'test', 10)]:
        identifiers = stratified_ids(references[split], claims[split], per_label)
        result[field] = {'source_split': split, 'seed': SEED, 'selection': RULE,
                         'query_ids': identifiers, 'count': len(identifiers),
                         'label_counts': {label: per_label for label in LABELS},
                         'claims_sha256': hashes[split],
                         'full_source_query_count': len(references[split]), 'scope': SCOPE}
    result['short_evaluation_protocol'] = 'scifact-stratified-30-v1'
    result['official_test_policy'] = 'Short evaluation is exactly the frozen 30 IDs, requires config and manifest-bound selection; original full test is preserved separately.'
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'data-dir', 'claims-train', 'claims-test', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = build_manifest(args.manifest, args.data_dir, args.claims_train, args.claims_test, args.output)
    print(json.dumps({'validation_count': result['validation']['count'],
                      'formal_count': result['short_evaluation']['count'],
                      'seed': SEED, 'manifest_sha256': file_hash(args.output), 'scope': SCOPE}))


if __name__ == '__main__':
    main()
