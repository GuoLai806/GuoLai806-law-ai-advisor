from datetime import date

from benchmarks.run_legal_temporal import _ranking_metrics, _temporal_rank
from evidence_rag.catalog import Catalog


META = {
    "doc_id": "civil-benchmark",
    "version_id": "v1",
    "title": "中华人民共和国民法典",
    "source_kind": "synthetic",
    "verification_state": "approved",
    "effective_from": "2021-01-01",
}


def test_abstention_utility_requires_empty_ranking_without_gold_article():
    assert _ranking_metrics([], [])["ndcg_at_10"] == 1
    assert _ranking_metrics([], ["x"])["ndcg_at_10"] == 0
    assert _ranking_metrics(["第一条"], ["x", "第一条"])["mrr_at_10"] == 0.5


def test_timely_hard_filter_removes_pre_effective_and_preserves_valid_order(tmp_path):
    catalog = Catalog(tmp_path / "db")
    catalog.stage(
        ["第一条 合同规则甲。", "第二条 合同规则乙。"],
        META,
        "r1",
    )
    catalog.set_temporal_version("r1", valid_from="2021-01-01", published_at="2020-05-28")
    ids = [row[0] for row in catalog.db.execute("SELECT unit_id FROM units ORDER BY label")]

    before, decisions = _temporal_rank(
        catalog,
        "r1",
        ids,
        {"event_date": "2020-12-31"},
    )
    assert before == []
    assert all(value["validity"] == "not_yet_effective" for value in decisions.values())

    valid, decisions = _temporal_rank(
        catalog,
        "r1",
        ids,
        {"event_date": date(2022, 1, 1).isoformat()},
    )
    assert valid == ids
    assert all(value["validity"] == "valid" for value in decisions.values())
