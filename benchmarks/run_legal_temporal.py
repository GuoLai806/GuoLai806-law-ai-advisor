"""Real-catalog retrieval benchmark with deterministic temporal hard filtering.

This is a retrieval-only experiment. It runs against one pinned Catalog
revision and one pinned OpenViking scope, preserves every row, and never uses
test labels for ranking or parameter selection.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import date
import json
import math
import os
from pathlib import Path
import platform
import time

from evidence_rag.catalog import Catalog
from evidence_rag.retrieval import OpenViking, rrf
from evidence_rag.temporal import (
    TemporalConfig,
    TimeContext,
    candidate_time_score,
    temporal_compatibility,
)
from .e2e import Journal, file_hash, now


FORMAT = "legal-temporal-retrieval-v1"
VARIANTS = ("bm25", "openviking_bge_m3", "rrf", "rrf_timely")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_dataset(data, expected_count):
    if data.get("kind") not in {
        "author-constructed-temporal-development-not-independent-human-gold",
        "author-constructed-temporal-formal-not-independent-human-gold",
    }:
        raise ValueError("dataset provenance is not declared")
    rows = data.get("queries")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} queries")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError("query IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate query IDs")
    required = {"query", "event_date", "expected_labels", "time_class", "should_return"}
    for row in rows:
        if not required <= set(row):
            raise ValueError("query contract is incomplete")
        if row["event_date"] is not None:
            date.fromisoformat(row["event_date"])
        labels = row["expected_labels"]
        if not isinstance(labels, list) or len(labels) != len(set(labels)):
            raise ValueError("expected labels must be unique strings")
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("invalid expected label")
        if row["should_return"] != bool(labels):
            raise ValueError("should_return must agree with expected labels")


def _ranking_metrics(expected, ranking):
    """Use abstention utility for no-positive temporal queries.

    Standard nDCG is undefined without a positive gold article. Here no
    returned evidence is the correct outcome, so it scores one; any returned
    evidence scores zero. This does not measure answer correctness.
    """
    expected = set(expected)
    ranking = list(dict.fromkeys(ranking))
    if not expected:
        score = 1.0 if not ranking else 0.0
        return {
            "ndcg_at_10": score,
            "recall_at_20": score,
            "mrr_at_10": score,
            "first_relevant_rank": None,
            "returned_count": len(ranking),
        }
    top10 = ranking[:10]
    top20 = ranking[:20]
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, label in enumerate(top10)
        if label in expected
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(expected), 10)))
    first = next((rank for rank, label in enumerate(top10, 1) if label in expected), None)
    return {
        "ndcg_at_10": dcg / ideal,
        "recall_at_20": len(set(top20) & expected) / len(expected),
        "mrr_at_10": 1.0 / first if first else 0.0,
        "first_relevant_rank": first,
        "returned_count": len(ranking),
    }


def _percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))]


def _temporal_rank(catalog, revision, candidate_ids, row):
    manifest = json.loads(catalog.revision(revision)["manifest"])
    temporal = catalog.temporal_version(revision) or {}
    context = TimeContext(
        query_time=date.today(),
        event_date=date.fromisoformat(row["event_date"]) if row["event_date"] else None,
    )
    decisions = {}
    eligible = []
    for index, unit_id in enumerate(candidate_ids):
        decision = temporal_compatibility(
            context,
            valid_from=temporal.get("valid_from") or manifest.get("effective_from"),
            valid_to=temporal.get("valid_to") or manifest.get("effective_to"),
            available_from=temporal.get("published_at") or manifest.get("published_at"),
            eligible=(temporal.get("verification_state") or manifest.get("verification_state"))
            == "approved",
            config=TemporalConfig(),
        )
        decisions[unit_id] = decision
        if decision["eligible"]:
            semantic = 1.0 - index / max(1, len(candidate_ids) - 1)
            scored = candidate_time_score(semantic, decision, config=TemporalConfig())
            eligible.append((scored["final"], -index, unit_id))
    eligible.sort(reverse=True)
    return [unit_id for _, __, unit_id in eligible], decisions


async def run(args):
    output = Path(args.output)
    journal_path = Path(str(output) + ".journal.jsonl")
    if output.exists() or journal_path.exists():
        raise FileExistsError("preserve earlier benchmark outputs")
    data = _json(args.dataset)
    expected_count = 6 if args.split == "development" else 30
    _validate_dataset(data, expected_count)
    config = {
        "format": FORMAT,
        "variants": list(VARIANTS),
        "candidate_limit": 60,
        "per_source_limit": 40,
        "rrf_k": 60,
        "temporal_config": TemporalConfig().__dict__,
        "gold_used_for_ranking": False,
        "qwen_reranker": "unavailable: local weight files absent; historical SciFact excluded",
        "metrics_scope": "retrieval ranking and temporal abstention only; not answer quality",
    }
    source_files = (
        "benchmarks/run_legal_temporal.py",
        "evidence_rag/catalog.py",
        "evidence_rag/retrieval.py",
        "evidence_rag/temporal.py",
    )
    fingerprint = {path: file_hash(path) for path in source_files}
    catalog = Catalog(args.db)
    adapter = OpenViking(args.url, args.scope, timeout=args.timeout)
    journal = Journal(journal_path)
    started = time.monotonic()
    report = {
        "format": FORMAT,
        "created_utc": now(),
        "split": args.split,
        "dataset": str(args.dataset),
        "dataset_sha256": file_hash(args.dataset),
        "database": str(args.db),
        "database_sha256": file_hash(args.db),
        "revision": args.revision,
        "openviking_scope": args.scope,
        "openviking_health": None,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": config,
        "code_sha256": fingerprint,
        "rows": [],
    }
    journal.record({"type": "header", **report})
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            health = await client.get(args.url.rstrip("/") + "/health")
            health.raise_for_status()
            report["openviking_health"] = health.json()
        revision = catalog.revision(args.revision)
        if not revision:
            raise ValueError("unknown catalog revision")
        if catalog.active() != args.revision:
            raise ValueError("benchmark revision must be active")
        if not catalog.verified_temporal_versions():
            raise ValueError("benchmark revision has no verified temporal metadata")
        unit_by_uri = {
            row["uri"]: row["unit_id"]
            for row in catalog.db.execute(
                "SELECT unit_id,uri FROM units WHERE revision=? AND uri IS NOT NULL",
                (args.revision,),
            )
        }
        if len(unit_by_uri) != 1260:
            raise ValueError("expected 1260 mapped legal units")
        for completed, row in enumerate(data["queries"], 1):
            timings = {}
            stage_started = time.perf_counter()
            lexical_pairs = catalog.lexical_scored(
                args.revision, row["query"], limit=config["per_source_limit"]
            )
            bm25_ids = [unit_id for unit_id, _ in lexical_pairs]
            timings["bm25_ms"] = round((time.perf_counter() - stage_started) * 1000, 3)

            stage_started = time.perf_counter()
            dense_rows = await adapter.find(row["query"])
            dense_ids = []
            for hit in dense_rows:
                unit_id = unit_by_uri.get(hit["uri"])
                if unit_id and unit_id not in dense_ids:
                    dense_ids.append(unit_id)
            dense_ids = dense_ids[: config["per_source_limit"]]
            timings["openviking_bge_m3_ms"] = round(
                (time.perf_counter() - stage_started) * 1000, 3
            )

            stage_started = time.perf_counter()
            fused_ids, sources = rrf(
                {"bm25": bm25_ids, "openviking_bge_m3": dense_ids},
                k=config["rrf_k"],
            )
            fused_ids = fused_ids[: config["candidate_limit"]]
            timely_ids, decisions = _temporal_rank(
                catalog, args.revision, fused_ids, row
            )
            timings["rrf_and_timely_ms"] = round(
                (time.perf_counter() - stage_started) * 1000, 3
            )
            variants = {
                "bm25": bm25_ids[:20],
                "openviking_bge_m3": dense_ids[:20],
                "rrf": fused_ids[:20],
                "rrf_timely": timely_ids[:20],
            }
            labels = {
                unit_id: catalog.get(args.revision, unit_id)["label"]
                for unit_id in set().union(*variants.values())
            }
            result = {
                "id": row["id"],
                "query": row["query"],
                "event_date": row["event_date"],
                "time_class": row["time_class"],
                "expected_labels": row["expected_labels"],
                "should_return": row["should_return"],
                "timings_ms": timings,
                "sources": {
                    "bm25": bm25_ids,
                    "openviking_bge_m3": dense_ids,
                    "rrf_sources": sources,
                },
                "temporal_decisions": decisions,
                "variants": {},
            }
            for variant, unit_ids in variants.items():
                ranked_labels = [labels[unit_id] for unit_id in unit_ids]
                metrics = _ranking_metrics(row["expected_labels"], ranked_labels)
                result["variants"][variant] = {
                    "unit_ids": unit_ids,
                    "labels": ranked_labels,
                    **metrics,
                }
            report["rows"].append(result)
            journal.record({"type": "query", "completed": completed, "row": result})
            print(
                json.dumps(
                    {"event": "progress", "completed": completed, "total": expected_count},
                    separators=(",", ":"),
                ),
                flush=True,
            )

        summary = {}
        for variant in VARIANTS:
            rows = [row["variants"][variant] for row in report["rows"]]
            latencies = [
                sum(
                    value
                    for key, value in row["timings_ms"].items()
                    if (variant == "bm25" and key == "bm25_ms")
                    or (variant == "openviking_bge_m3" and key == "openviking_bge_m3_ms")
                    or (variant in ("rrf", "rrf_timely") and key != "bm25_ms")
                )
                for row in report["rows"]
            ]
            before = [
                item
                for item, source in zip(rows, report["rows"])
                if source["expected_labels"] == []
            ]
            relevant = [
                item
                for item, source in zip(rows, report["rows"])
                if source["expected_labels"] != []
            ]
            summary[variant] = {
                "query_count": len(rows),
                "ndcg_at_10": sum(item["ndcg_at_10"] for item in rows) / len(rows),
                "recall_at_20": sum(item["recall_at_20"] for item in rows) / len(rows),
                "mrr_at_10": sum(item["mrr_at_10"] for item in rows) / len(rows),
                "valid_version_hit_at_10": (
                    sum(float(item["first_relevant_rank"] is not None) for item in relevant)
                    / len(relevant)
                    if relevant
                    else None
                ),
                "stale_evidence_leakage": (
                    sum(bool(item["returned_count"]) for item in before) / len(before)
                    if before
                    else 0.0
                ),
                "latency_ms": {
                    "mean": sum(latencies) / len(latencies),
                    "p50": _percentile(latencies, 0.5),
                    "p95": _percentile(latencies, 0.95),
                },
            }
        report["summary"] = summary
        report["class_counts"] = dict(
            Counter(row["time_class"] for row in report["rows"])
        )
        report["finished_utc"] = now()
        report["elapsed_seconds"] = time.monotonic() - started
        report["complete"] = len(report["rows"]) == expected_count
        _write_exclusive(output, report)
        journal.record(
            {
                "type": "complete",
                "complete": report["complete"],
                "report_sha256": file_hash(output),
            }
        )
        print(json.dumps({"event": "complete", "summary": summary}, ensure_ascii=False))
    finally:
        journal.close()
        await adapter.aclose()
        catalog.db.close()


def _write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("development", "formal"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
