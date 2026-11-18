"""Bounded, evidence-checked traversal of a pinned dependency graph."""
from collections import deque
from datetime import date
from hashlib import sha256
from time import monotonic

from .graph_index import GraphStore


def _verified_source(catalog, revision, edge, cache):
    uid = edge['unit_id']
    if uid not in cache:
        try:
            cache[uid] = catalog.get(revision, uid)
        except (ValueError, KeyError, TypeError):
            cache[uid] = None
    unit = cache[uid]
    if not unit or not isinstance(unit.get('text'), str):
        return None
    text = unit['text']
    actual = sha256(text.encode('utf-8')).hexdigest()
    if actual != unit.get('sha256') or actual != edge.get('content_sha256'):
        return None
    start, end = edge.get('char_start'), edge.get('char_end')
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        return None
    if 'quote' in edge and edge['quote'] != text[start:end]:
        return None
    part_no = edge.get('part_no')
    part = None
    if part_no is not None:
        part = next((p for p in unit.get('parts', []) if p.get('part_no') == part_no), None)
        if not part or not isinstance(part.get('text'), str):
            return None
        loc = part.get('locator') or {}
        pstart, pend = loc.get('char_start'), loc.get('char_end')
        if (type(pstart) is not int or type(pend) is not int or
                not 0 <= pstart <= start < end <= pend <= len(text) or
                text[pstart:pend] != part['text'] or
                sha256(part['text'].encode('utf-8')).hexdigest() != part.get('sha256')):
            return None
    return {'unit_id': uid, 'part_no': part_no, 'char_start': start, 'char_end': end,
            'content_sha256': actual, 'source_locator': unit.get('locator') or {},
            'part_locator': part.get('locator') if part else None}


def walk_dependencies(catalog, revision, seed='', candidate_unit_ids=(), *, as_of: date,
                      budget_ms=750, max_states=50000, max_edges=150000,
                      path_limit=100, path_offset=0, expected_build_id=None, cancelled=None,
                      query=None):
    """Walk all reachable validated dependency edges, or report why it stopped.

    Catalog access follows the existing shared-corpus authorization boundary.
    This function does not implement a separate per-user ACL.
    """
    started = monotonic()
    result = {'graph_build_id': None, 'as_of': as_of.isoformat(),
              'traversal_status': 'unavailable', 'evidence_status': 'incomplete',
              'stop_reason': None, 'graph_ms': 0.0, 'max_depth_reached': 0,
              'visited_count': 0, 'edge_count': 0, 'paths': [],
              'paths_truncated': False, 'evidence_unit_ids': [], 'seed_id': None,
              'next_path_offset': None}
    cache = {}
    def stop():
        if cancelled and cancelled():
            return 'cancelled'
        if (monotonic() - started) * 1000 >= budget_ms:
            return 'budget_ms'
        return None
    def finish():
        shown_through = path_offset + len(result['paths'])
        if result['edge_count'] > shown_through and result['paths']:
            result['next_path_offset'] = shown_through
        result['paths_truncated'] = result['edge_count'] > len(result['paths'])
        result['graph_ms'] = round((monotonic() - started) * 1000, 3)
        return result
    def interrupted(reason):
        result['traversal_status'] = 'cancelled' if reason == 'cancelled' else 'partial_budget'
        result['evidence_status'] = 'incomplete'
        result['stop_reason'] = reason
        return finish()
    try:
        reason = stop()
        if reason:
            return interrupted(reason)
        store = GraphStore(catalog)
        build = store.active_build(revision)
        if not build:
            result['stop_reason'] = 'graph_unavailable'
            return finish()
        build_id = build['graph_build_id']
        if expected_build_id is not None and build_id != expected_build_id:
            result['stop_reason'] = 'graph_changed'
            return finish()
        result['graph_build_id'] = build_id
        result['evidence_status'] = 'sufficient' if build.get('coverage_complete') else 'incomplete'
        options = {'query':query} if query else {}
        seeds = store.resolve_seeds(revision, build_id, seed, candidate_unit_ids, **options)
        reason = stop()
        if reason:
            return interrupted(reason)
        if len(seeds) != 1:
            result['evidence_status'] = 'ambiguous_seed' if len(seeds) > 1 else 'incomplete'
            result['stop_reason'] = 'ambiguous_seed' if len(seeds) > 1 else 'seed_not_found'
            return finish()
        if max_states < 1:
            return interrupted('max_states')
        root = seeds[0]['entity_id']
        result['seed_id'] = root
        depths = {root: 0}
        frontier = deque([root])
        result['visited_count'] = 1
        raw_edges = 0
        unit_ids = set()
        seen_edges = set()
        while frontier:
            reason = stop()
            if reason:
                return interrupted(reason)
            # Process a single depth per batch for shortest, deterministic predecessors.
            level = depths[frontier[0]]
            batch = []
            while frontier and depths[frontier[0]] == level and len(batch) < 128:
                batch.append(frontier.popleft())
            batch_ids = set(batch)
            for edge in store.outgoing(revision, build_id, batch, as_of):
                reason = stop()
                if reason:
                    return interrupted(reason)
                if raw_edges >= max_edges:
                    return interrupted('max_edges')
                raw_edges += 1
                src, dst = edge['src_id'], edge['dst_id']
                if src not in batch_ids:
                    result['evidence_status'] = 'incomplete'
                    continue
                source = _verified_source(catalog, revision, edge, cache)
                if source is None:
                    result['evidence_status'] = 'incomplete'
                    continue
                edge_id = edge['edge_id']
                if edge_id in seen_edges:
                    continue
                if dst not in depths and len(depths) >= max_states:
                    return interrupted('max_states')
                seen_edges.add(edge_id)
                depth = depths[src] + 1
                if dst not in depths:
                    depths[dst] = depth
                    frontier.append(dst)
                    result['visited_count'] += 1
                    result['max_depth_reached'] = max(result['max_depth_reached'], depth)
                result['edge_count'] += 1
                uid = source['unit_id']
                if uid not in unit_ids:
                    unit_ids.add(uid)
                    result['evidence_unit_ids'].append(uid)
                path = {'edge_id': edge_id, 'src_id': src, 'src_name': edge['src_name'],
                        'dst_id': dst, 'dst_name': edge['dst_name'],
                        'predicate': edge['predicate'], 'depth': depth, **source}
                if result['edge_count'] > path_offset and len(result['paths']) < max(0, path_limit):
                    result['paths'].append(path)
        result['traversal_status'] = 'exhausted'
        return finish()
    except Exception:
        result['traversal_status'] = 'error'
        result['evidence_status'] = 'incomplete'
        result['stop_reason'] = 'graph_error'
        return finish()
