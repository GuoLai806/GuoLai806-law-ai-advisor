"""Opt-in dependency traversal wired into the ordinary evidence bundle."""

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import date
from threading import Event
from time import monotonic

from .budget import assemble
from .catalog import Catalog
from .graph_index import GraphStore
from .graph_retrieval import walk_dependencies


def unavailable_graph(as_of, reason):
    return {
        'graph_build_id': None, 'as_of': as_of.isoformat(),
        'traversal_status': 'unavailable', 'evidence_status': 'incomplete',
        'stop_reason': reason, 'graph_ms': 0, 'paths': [], 'paths_truncated': False,
        'visited_count': 0, 'edge_count': 0, 'max_depth_reached': 0,
        'next_path_offset': None, 'seed_id': None,
    }


def _path_evidence(catalog, revision, paths, max_items):
    """Choose the exact source part used by each edge; never a lexical substitute."""
    units = {}
    evidence = {}
    for path in paths:
        uid, part_no = path['unit_id'], path['part_no']
        key = (uid, part_no)
        if key in evidence:
            continue
        if len(evidence) == max_items:
            break
        if uid not in units:
            units[uid] = catalog.get(revision, uid)
        unit = units[uid]
        if not unit or unit['sha256'] != path['content_sha256']:
            raise ValueError('graph source changed')
        doc = unit['document'] or {}
        part = next((p for p in unit['parts'] if p['part_no'] == part_no), None)
        if part_no is not None and part is None:
            raise ValueError('graph source part changed')
        item = {
            'citation_id': f'E{len(evidence) + 1}',
            'unit_id': uid, 'document_id': doc.get('doc_id'),
            'title': unit['title'], 'structure': unit['structure'],
            'text': part['text'] if part else unit['text'],
            'source_url': doc.get('source_url'), 'source_kind': doc.get('source_kind'),
            'source_locator': part['locator'] if part else unit['locator'],
            'content_sha256': part['sha256'] if part else unit['sha256'],
            'verification_state': unit['verification_state'], 'profile': 'general',
            'retrieval_sources': [{'source': 'graph', 'rank': len(evidence) + 1}],
            'reranker_score': None,
        }
        if part:
            item.update(part_no=part_no, document_sha256=unit['sha256'], has_more=len(unit['parts']) > 1)
        evidence[key] = item
    return list(evidence.values())


def _current_source(catalog, revision, expected_manifest):
    row = catalog.revision(revision)
    return bool(row and row['state'] in ('active','retained')
                and (expected_manifest is None or expected_manifest == row['manifest'])
                and json.loads(row['manifest']).get('verification_state') in ('approved','provided'))


def _lookup(db_path, revision, seed, candidates, as_of, budget_ms, max_items, stopped,
            path_offset, expected_build_id, query, expected_manifest):
    started = monotonic()
    try:
        catalog = Catalog(db_path, read_only=True)
        with closing(catalog.db):
            catalog.db.execute('BEGIN')
            if _current_source(catalog, revision, expected_manifest):
                graph = walk_dependencies(catalog, revision, seed, candidates,
                                          as_of=as_of, budget_ms=budget_ms, cancelled=stopped.is_set,
                                          path_offset=path_offset, expected_build_id=expected_build_id,
                                          query=query)
                graph.pop('evidence_unit_ids', None)
                evidence = _path_evidence(catalog, revision, graph['paths'], max_items)
            else:
                graph, evidence = unavailable_graph(as_of, 'source_revoked'), []
            catalog.db.rollback()
            # A graph pointer can advance during a valid pinned read. The
            # sources and previous build still must be authorized at release.
            build = graph['graph_build_id']
            if (not _current_source(catalog, revision, expected_manifest) or
                    (build and not GraphStore(catalog)._available(revision, build))):
                graph = unavailable_graph(as_of, 'source_revoked')
                evidence = []
    except (ValueError, OSError, sqlite3.Error):
        graph = unavailable_graph(as_of, 'graph_error')
        graph['traversal_status'] = 'error'
        evidence = []
    graph['graph_ms'] = round((monotonic() - started) * 1000, 2)
    return graph, evidence


def graph_prompt(query, graph):
    if not graph or graph.get('uses_text_fallback'):
        return query
    return ('请简洁解释随附原文所支持的依赖，引用原文编号。'
            '完整关系清单由页面单独展示，请勿将本段自然语言总结描述成全部依赖。\n' + query)


async def add_dependency_graph(bundle, *, db_path, enabled, seed, as_of: date, budget_ms,
                               max_items, tokenizer, max_evidence_tokens, query, expose_trace,
                               path_offset=0, expected_build_id=None, expected_manifest=None):
    started = monotonic()
    trace = bundle.get('retrieval_trace') or {}
    if not expose_trace:
        bundle.pop('retrieval_trace', None)
    revision = bundle.get('corpus_revision')
    if not enabled or not revision or bundle['query_plan'].get('profile') != 'general':
        bundle['graph'] = unavailable_graph(as_of, 'graph_disabled' if not enabled else 'graph_unavailable')
        bundle['graph']['uses_text_fallback'] = not path_offset
        return bundle
    candidates = tuple(item['unit_id'] for item in trace.get('candidates', ()))
    stopped = Event()
    operation = asyncio.create_task(asyncio.to_thread(
        _lookup, db_path, revision, seed, candidates, as_of, budget_ms, max_items, stopped,
        path_offset, expected_build_id, query, expected_manifest))
    try:
        graph, evidence = await asyncio.shield(operation)
    except asyncio.CancelledError:
        stopped.set()
        # The thread owns its connection and cleans it up even after disconnect.
        raise

    bundle['graph'] = graph
    graph['uses_text_fallback'] = not path_offset and not graph['paths'] and (
        graph['stop_reason'] in ('graph_unavailable', 'graph_error')
    )
    if graph['stop_reason'] == 'source_revoked':
        bundle['evidence'] = []
        bundle.pop('retrieval_trace',None)
        bundle['budget'] = {'state':'not_applicable','used_tokens':None,'omitted':[]}
        bundle['token_budget_status'] = 'not_applicable'
        bundle['status'] = 'insufficient_evidence'
        bundle['degraded'] = True
        bundle['degraded_reasons'].append('graph_source_revoked')
    if evidence:
        selected, budget = assemble(evidence, tokenizer, max_evidence_tokens, graph_prompt(query, graph))
        bundle.update(
            evidence=selected if tokenizer else evidence, budget=budget,
            token_budget_status=budget['state'], status='ready',
        )
        bundle['query_plan']['mode'] = 'graph_dependency'
        ids = {(item['unit_id'], item.get('part_no')): item['citation_id']
               for item in bundle['evidence']}
        for path in graph['paths']:
            citation = ids.get((path['unit_id'], path['part_no']))
            if citation:
                path['citation_id'] = citation
    graph['generation_evidence_complete'] = bool(
        evidence and not path_offset and not graph['paths_truncated'] and
        all('citation_id' in path for path in graph['paths'])
    )
    bundle['retrieval_ms'] = round(bundle['retrieval_ms'] + (monotonic() - started) * 1000, 2)
    return bundle
