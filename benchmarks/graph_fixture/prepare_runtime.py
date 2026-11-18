"""Stage reviewed development fixture relations and validate their real Viking index."""
import argparse
import asyncio
import json
import os
from pathlib import Path

from evidence_rag.catalog import Catalog
from evidence_rag.graph_index import GraphStore
from evidence_rag.retrieval import OpenViking
from evidence_rag.viking_index import index_one, reconcile_one, validate_index

FIXTURE = Path(__file__).parent
CORPUS = 'express-development'


def lines(name):
    return [json.loads(line) for line in (FIXTURE / name).read_text().splitlines()]


def reviewed_graph(catalog, revision, documents, effective_dates):
    units = {doc: catalog.get(revision, uid) for doc, uid in catalog.db.execute(
        'SELECT doc_id,unit_id FROM documents WHERE revision=?', (revision,))}
    entities, edges = [], []
    for doc in documents:
        key = doc['_id']
        name = doc['metadata']['name']
        unit = units[key]
        start = unit['text'].index(name)
        entities.append({'entity_id': key, 'name': key, 'type': 'package_release',
                         'aliases': [name],
                         'mentions': [{'unit_id': unit['unit_id'], 'char_start': start,
                                       'char_end': start+len(name)}]})
    for edge in lines('edges.jsonl'):
        src, dst = edge['source'], edge['target']
        unit = units[edge['evidence_id']]
        part = unit['parts'][0]
        name = dst.rsplit('@', 1)[0]
        quote = json.dumps(name) + ': ' + json.dumps(edge['spec'])
        start = unit['text'].index(quote)
        # The independent fixture records the dependency's published package
        # manifest. It does not attest installed closure or deployment dates.
        available = max(effective_dates[src]['published_at'][:10],
                        effective_dates[dst]['published_at'][:10])
        edges.append({'src_id': src, 'dst_id': dst, 'predicate': 'depends_on', 'supports': [{
            'unit_id': unit['unit_id'], 'part_no': part['part_no'], 'char_start': start,
            'char_end': start + len(quote), 'quote': quote, 'content_sha256': unit['sha256'],
            'verification_state': 'approved', 'temporal_status': 'interval',
            'valid_from': available, 'valid_to': None,
        }]})
    return {'revision': revision, 'extraction_identity': 'npm-manifest-exact-dependency-subset-v1',
            'coverage_complete': False, 'entities': entities, 'edges': edges}


async def stage(args):
    docs = lines('docs.jsonl')
    effective_dates = json.loads((FIXTURE / 'publication_dates.json').read_text())
    db = Path(args.db)
    if db.exists():
        raise ValueError('use a fresh catalog path; inspect earlier writes before rerun')
    catalog = Catalog(db)
    try:
        catalog.stage_documents(docs, {'corpus_id': CORPUS, 'version_id': 'npm-fixture-v1',
                                'source_kind': 'public_npm_metadata',
                                'verification_state': 'approved'}, args.revision)
        ids = [row[0] for row in catalog.db.execute(
            'SELECT unit_id FROM units WHERE revision=? ORDER BY unit_id', (args.revision,))]
        scope = f'viking://resources/rag/{CORPUS}/{args.revision}'
        key = os.getenv('OPENVIKING_API_KEY')
        async with OpenViking(args.ov_url, scope, headers={'X-API-Key': key} if key else None,
                              timeout=45) as viking:
            for uid in ids:
                await index_one(catalog, viking, args.revision, uid)
                await reconcile_one(catalog, viking, args.revision, uid)
            indexed = await validate_index(catalog, viking, args.revision)
        if not indexed['index_visibility_validated']:
            raise ValueError('Viking scope was written but not validated')
        catalog.activate(args.revision, None)
        store = GraphStore(catalog)
        store.migrate()
        payload = reviewed_graph(catalog, args.revision, docs, effective_dates)
        built = store.stage(args.revision, payload, 'npm-review-fixture-v1')
        graph = store.activate(args.revision, built['graph_build_id'], None)
        return {'scope': scope, 'indexed': indexed, 'graph_build_id': graph['graph_build_id'],
                'graph_source': payload['extraction_identity'], 'coverage_complete': False,
                'entities': len(payload['entities']), 'edges': len(payload['edges'])}
    finally:
        catalog.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--ov-url', required=True)
    parser.add_argument('--report', default=None)
    args = parser.parse_args()
    output = json.dumps(asyncio.run(stage(args)), indent=2)
    if args.report:
        with Path(args.report).open('x') as report:
            report.write(output+'\n')
    print(output)
