"""Deterministic legal-time filtering and TimelyRAG-inspired reranking.

The catalog/backend establish textual candidates.  This module never treats a
document insertion date as legal validity, and never uses gold labels to tune
the online ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
import re
from typing import Iterable

DATE_RE = re.compile(r'(?<!\d)((?:19|20)\d{2})[-/.年](\d{1,2})(?:[-/.月](\d{1,2})日?)?(?!\d)')
FULL_DATE_RE = re.compile(r'(?<!\d)((?:19|20)\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?(?!\d)')


def as_date(value, *, field='date'):
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f'{field} must be an ISO date') from exc
    raise ValueError(f'{field} must be an ISO date')


def temporal_dates(text: str) -> list[date]:
    """Return explicit, valid ISO-like dates; incomplete dates are not inferred."""
    found = []
    for match in FULL_DATE_RE.finditer(text or ''):
        try:
            found.append(date(int(match.group(1)), int(match.group(2)), int(match.group(3))))
        except ValueError:
            continue
    return found


def interval_contains(day: date, valid_from=None, valid_to=None) -> bool:
    start = as_date(valid_from, field='valid_from')
    end = as_date(valid_to, field='valid_to')
    if start and end and start > end:
        raise ValueError('valid_from must not be after valid_to')
    return (start is None or day >= start) and (end is None or day <= end)


def validity(day: date, valid_from=None, valid_to=None, *, verified=True) -> str:
    """A legal validity decision is separate from a temporal relevance score."""
    if not verified or (valid_from in (None, '') and valid_to in (None, '')):
        return 'unknown'
    start = as_date(valid_from, field='valid_from')
    end = as_date(valid_to, field='valid_to')
    if start and end and start > end:
        return 'invalid_interval'
    day=as_date(day,field='reference_date')
    if start and day < start:
        return 'not_yet_effective'
    if end and day > end:
        return 'expired'
    return 'valid'


@dataclass(frozen=True)
class TemporalConfig:
    retention_days: int = 0
    decay_days: int = 3650
    decay_gamma: float = 2.0
    temporal_weight_max: float = 0.35
    expiry_weight: float = 0.35
    query_time_weight: float = 0.10
    event_time_weight: float = 0.90

    def __post_init__(self):
        if self.retention_days < 0 or self.decay_days <= 0:
            raise ValueError('temporal durations are invalid')
        if not 0 < self.decay_gamma <= 10:
            raise ValueError('decay_gamma must be in (0,10]')
        if not 0 <= self.temporal_weight_max <= 1:
            raise ValueError('temporal_weight_max must be in [0,1]')
        if not 0 <= self.expiry_weight <= 1:
            raise ValueError('expiry_weight must be in [0,1]')
        if not 0 <= self.query_time_weight <= 1 or not 0 <= self.event_time_weight <= 1:
            raise ValueError('time signal weights must be in [0,1]')


@dataclass(frozen=True)
class TimeContext:
    query_time: date
    event_date: date | None
    granularity: str = 'day'

    @property
    def reference(self) -> date:
        return self.event_date or self.query_time


def _distance_days(value: date, start: date | None, end: date | None) -> float:
    if not start and not end:
        return float('inf')
    if (not start or value >= start) and (not end or value <= end):
        return 0.0
    targets = [x for x in (start, end) if x is not None]
    return float(min(abs((value - x).days) for x in targets))


def temporal_compatibility(context: TimeContext, *, valid_from=None, valid_to=None,
                           available_from=None, eligible: bool = True,
                           config: TemporalConfig | None = None) -> dict:
    """Compute an auditable, candidate-pool-relative temporal score.

    Explicit event date is preferred over query time.  A valid selected version
    receives the highest compatibility.  An expired version still receives a
    distance score but is strongly penalized; callers must remove it before
    generation.  Unknown intervals are neutral and never labeled valid.
    """
    cfg = config or TemporalConfig()
    ref = context.reference
    start = as_date(valid_from, field='valid_from')
    end = as_date(valid_to, field='valid_to')
    available = as_date(available_from, field='available_from')
    # Keep all comparisons in dates; public APIs pass ISO strings.
    ref = as_date(ref, field='reference_date')
    dist = _distance_days(ref, start, end)
    state = validity(ref, start, end, verified=eligible)
    if math.isinf(dist):
        affinity = None
    else:
        while dist > cfg.retention_days and dist > cfg.decay_days:
            dist = cfg.decay_days if cfg.decay_days else 1
        affinity = math.exp(-cfg.decay_gamma * min(dist, cfg.decay_days) / cfg.decay_days)
    if eligible and state == 'valid':
        affinity = 1.0
    expiry_risk = 0.0 if not available or context.query_time <= available else min(1.0, (context.query_time - available).days / max(1, cfg.decay_days))
    return {
        'reference_date': ref.isoformat(),
        'reference_source': 'event_date' if context.event_date else 'query_time',
        'validity': state,
        'eligible': bool(eligible and state == 'valid'),
        'distance_days': None if math.isinf(dist) else round(dist, 6),
        'temporal_affinity': affinity,
        'expiry_risk': round(expiry_risk, 6),
        'valid_from': start.isoformat() if start else None,
        'valid_to': end.isoformat() if end else None,
        'available_from': available.isoformat() if available else None,
    }


def candidate_time_score(semantic: float, temporal: dict, *, config: TemporalConfig | None = None) -> dict:
    """Combined online score: semantics + time proximity + validity guard."""
    cfg = config or TemporalConfig()
    if not isinstance(semantic, (int, float)) or isinstance(semantic, bool) or not math.isfinite(semantic):
        raise ValueError('semantic score must be finite')
    affinity = temporal.get('temporal_affinity')
    has_time = affinity is not None
    time_weight = cfg.temporal_weight_max if has_time else 0.0
    temporal_value = float(affinity) if has_time else 0.5
    validity_flag = 1.0 if temporal.get('validity') == 'valid' else 0.0
    # A known invalid version cannot outrank valid evidence at equal semantics.
    final = ((1.0 - time_weight) * float(semantic) + time_weight * temporal_value
             + cfg.expiry_weight * validity_flag)
    if temporal.get('validity') in ('expired', 'not_yet_effective', 'invalid_interval'):
        final -= cfg.expiry_weight
    return {'semantic': float(semantic), 'temporal_weight': time_weight,
            'temporal_score': temporal_value if has_time else None,
            'validity_bonus': cfg.expiry_weight * validity_flag,
            'final': final}


def rank_temporal(candidates: Iterable[dict], context: TimeContext, *, config: TemporalConfig | None=None) -> list[dict]:
    """Risk: candidates need semantic_score and validity fields supplied by caller."""
    cfg = config or TemporalConfig()
    result=[]
    for item in candidates:
        values=temporal_compatibility(context,valid_from=item.get('valid_from'),valid_to=item.get('valid_to'),available_from=item.get('available_from'),eligible=bool(item.get('verified',False)),config=cfg)
        combo=candidate_time_score(float(item.get('semantic_score',0.0)),values,config=cfg)
        result.append({**item,'temporal':values,'temporal_rank_score':combo})
    return sorted(result,key=lambda x:(x['temporal']['eligible'],x['temporal_rank_score']['final'],x.get('unit_id','')),reverse=True)
