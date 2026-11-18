from datetime import date
from hashlib import sha256

import pytest

from evidence_rag import graph_retrieval as walk

DAY = date(2026, 1, 1)
TEXT = 'A depends on B.'
HASH = sha256(TEXT.encode()).hexdigest()


class Catalog:
    def get(self, revision, uid):
        assert revision == 'r1'
        if uid == 'missing':
            return None
        return {'text': TEXT, 'sha256': HASH, 'locator': {'kind': 'text'}, 'parts': []}


def edge(n, src, dst, **kwargs):
    return dict(edge_id=str(n), src_id=str(src), src_name=str(src), dst_id=str(dst),
                dst_name=str(dst), predicate='depends_on', unit_id='u', part_no=None,
                char_start=0, char_end=len(TEXT), content_sha256=HASH, **kwargs)


class Store:
    def __init__(self, catalog):
        self.edges = catalog.edges
        self.seeds = getattr(catalog, 'seeds', [{'entity_id': '0', 'name': '0'}])
        self.build = getattr(catalog, 'build', {'graph_build_id': 'b1', 'coverage_complete': True})

    def active_build(self, revision):
        return self.build

    def resolve_seeds(self, revision, build_id, seed, candidate_unit_ids):
        assert build_id == 'b1'
        return self.seeds

    def outgoing(self, revision, build_id, src_ids, as_of):
        assert revision == 'r1' and build_id == 'b1' and as_of == DAY
        for e in self.edges:
            if e['src_id'] in src_ids and (not e.get('valid_from') or e['valid_from'] <= as_of.isoformat()) and (not e.get('valid_to') or as_of.isoformat() < e['valid_to']):
                yield e


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(walk, 'GraphStore', Store)
    c = Catalog()
    c.edges = []
    return c


def run(c, **kw):
    return walk.walk_dependencies(c, 'r1', '0', as_of=DAY, budget_ms=10000, **kw)


def test_long_chain_cycle_and_display_limit(catalog):
    catalog.edges = [edge(i, i, i + 1) for i in range(1001)] + [edge('loop', 1001, 0)]
    result = run(catalog, path_limit=2)
    assert result['traversal_status'] == 'exhausted'
    assert result['visited_count'] == 1002 and result['max_depth_reached'] == 1001
    assert result['edge_count'] == 1002 and result['paths_truncated']
    assert len(result['paths']) == 2 and result['evidence_unit_ids'] == ['u']


def test_wide_duplicate_edges_and_time(catalog):
    catalog.edges = [edge(i, 0, i + 1) for i in range(800)]
    catalog.edges += [edge(0, 0, 1), edge('old', 0, 900, valid_to='2026-01-01'),
                      edge('future', 0, 901, valid_from='2027-01-01')]
    result = run(catalog)
    assert result['visited_count'] == 801 and result['edge_count'] == 800
    assert result['traversal_status'] == 'exhausted'


def test_invalid_source_never_disclosed_or_traversed(catalog):
    invalid = edge('bad', 0, 'secret')
    invalid.update(unit_id='missing', src_name='secret source', dst_name='secret target')
    catalog.edges = [invalid, edge('hidden', 'secret', 2)]
    result = run(catalog)
    assert result['evidence_status'] == 'incomplete' and result['visited_count'] == 1
    assert result['paths'] == [] and result['evidence_unit_ids'] == []
    assert 'secret' not in str(result)


@pytest.mark.parametrize('field,value', [('content_sha256', 'wrong'), ('char_start', -1), ('char_end', 999), ('part_no', 3), ('quote', 'fake')])
def test_invalid_evidence_pointer(catalog, field, value):
    e = edge(1, 0, 1)
    e[field] = value
    catalog.edges = [e]
    result = run(catalog)
    assert result['edge_count'] == 0 and result['evidence_status'] == 'incomplete'


def test_budgets_and_cancellation(catalog):
    catalog.edges = [edge(1, 0, 1), edge(2, 1, 2)]
    for kwargs, reason in [({'max_states': 1}, 'max_states'), ({'max_edges': 1}, 'max_edges')]:
        result = run(catalog, **kwargs)
        assert result['traversal_status'] == 'partial_budget' and result['stop_reason'] == reason
    assert walk.walk_dependencies(catalog, 'r1', '0', as_of=DAY, budget_ms=0)['stop_reason'] == 'budget_ms'
    assert run(catalog, cancelled=lambda: True)['traversal_status'] == 'cancelled'


def test_unavailable_and_ambiguous(catalog):
    catalog.build = None
    assert run(catalog)['traversal_status'] == 'unavailable'
    catalog.build = {'graph_build_id': 'b1', 'coverage_complete': True}
    catalog.seeds = [{'entity_id': '0'}, {'entity_id': '1'}]
    result = run(catalog)
    assert result['evidence_status'] == 'ambiguous_seed' and result['paths'] == []


def test_incomplete_build_and_read_error(catalog):
    catalog.build = {'graph_build_id': 'b1', 'coverage_complete': False}
    assert run(catalog)['evidence_status'] == 'incomplete'
    def fail(*args):
        raise RuntimeError('private detail')
    catalog.get = fail
    catalog.edges = [edge(1, 0, 1)]
    result = run(catalog)
    assert result['traversal_status'] == 'error' and 'private detail' not in str(result)
