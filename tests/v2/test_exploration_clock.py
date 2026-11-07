"""Independent fixtures for preflight clocks; no runner timing helpers are used."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from tools.audit_flow_timing import row_clock_checks

START = 1_794_009_600_000_000_000


def utc(value):
    if value is None:
        return None
    seconds, nanos = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + f'.{nanos:09d}Z'


def sample(milliseconds, monotonic=None, *, valid=True):
    wall = START + int(milliseconds * 1_000_000) if milliseconds is not None else None
    return {'wall_ns': wall, 'utc': utc(wall),
            'monotonic_ms': milliseconds if monotonic is None else monotonic,
            'wall_ms': milliseconds if valid else None, 'observed_wall_ms': milliseconds}


def fixture(*, anomaly=None, no_stream=False, prefix=3):
    snapshots = [sample(100), sample(200), sample(300)]
    anomalies = []
    if anomaly == 'reversed':
        snapshots[1] = sample(50, 200, valid=False)
        snapshots[2] = sample(300, valid=False)
        anomalies = ['wall_clock_reversed']
    elif anomaly == 'missing':
        snapshots[1] = sample(None, 200, valid=False)
        snapshots[2] = sample(300, valid=False)
        anomalies = ['wall_clock_missing']
    elif anomaly == 'gap':
        snapshots[1] = sample(130_200, 200)
        snapshots[2] = sample(130_300, 300)
    valid = not anomalies
    gap = 130_000 if anomaly == 'gap' else 0
    names = ('started', 'preview_received', 'original_received')
    exploration = {name: snapshot for name, snapshot in zip(names[:prefix], snapshots[:prefix])}
    events = []
    if not no_stream:
        for kind, monotonic in [('evidence_ready', 400), ('answer_delta', 500), ('completed', 600)]:
            observed = monotonic + gap
            wall = START + observed * 1_000_000
            events.append({'event': kind, 'data': {}, 'received_ms': monotonic,
                           'received_wall_ns': wall, 'received_utc': utc(wall),
                           'received_wall_ms': observed if valid else None,
                           'observed_received_wall_ms': observed})
    stream_ns = None if no_stream else START + (700 + gap) * 1_000_000
    failure = {'type': 'HTTPStatusError'} if no_stream else None
    raw = {'started_wall_ns': START, 'started_utc': utc(START), 'exploration': exploration,
           'events': events, 'stream_total_ms': None if no_stream else 700,
           'stream_finished_wall_ns': stream_ns, 'stream_finished_utc': utc(stream_ns),
           'stream_total_wall_ms': None if no_stream or not valid else 700 + gap,
           'stream_observed_wall_ms': None if no_stream else 700 + gap,
           'clock_anomalies': anomalies, 'failure': failure}
    row = {'started_wall_ns': START, 'started_utc': utc(START),
           'finished_wall_ns': START + (800 + gap) * 1_000_000,
           'finished_utc': utc(START + (800 + gap) * 1_000_000),
           'user_total_ms': 800, 'observed_user_wall_ms': 800 + gap,
           'user_total_wall_ms': 800 + gap if valid else None,
           'wall_minus_monotonic_ms': gap, 'clock_anomalies': anomalies,
           'wall_clock_valid': valid, 'wall_monotonic_divergence': gap > 1000,
           'first_answer_ms': None if no_stream else 500,
           'first_answer_wall_ms': None if no_stream or not valid else 500 + gap,
           'stream_total_wall_ms': None if no_stream or not valid else 700 + gap,
           'failure': failure}
    if not no_stream:
        row['stream_total_ms'] = 700
        row['exploration'] = {'preview_ms': 200, 'original_ms': 300}
    return row, raw


@pytest.mark.parametrize('anomaly', [None, 'reversed', 'missing', 'gap'])
def test_preflight_clock_samples_recompute_before_stream(anomaly):
    row, raw = fixture(anomaly=anomaly)
    assert all(row_clock_checks(row, raw).values())


@pytest.mark.parametrize('prefix', [1, 2, 3])
def test_failed_preflight_has_no_stream_clock_or_first_answer(prefix):
    row, raw = fixture(no_stream=True, prefix=prefix)
    assert 'stream_total_ms' not in row  # Older failure records need not have this key.
    assert all(row_clock_checks(row, raw).values())
    assert row_clock_checks(row, raw)['clock_flags_match'] is True


@pytest.mark.parametrize('anomaly', ['reversed', 'missing'])
def test_failed_preflight_still_audits_observed_clock_anomalies(anomaly):
    row, raw = fixture(no_stream=True, anomaly=anomaly, prefix=2)
    assert all(row_clock_checks(row, raw).values())
    forged = deepcopy(row)
    forged.update(clock_anomalies=[], wall_clock_valid=True,
                  user_total_wall_ms=forged['observed_user_wall_ms'])
    assert row_clock_checks(forged, raw)['clock_flags_match'] is False


@pytest.mark.parametrize('name', ['started', 'preview_received', 'original_received'])
@pytest.mark.parametrize('field', ['utc', 'observed_wall_ms', 'wall_ms', 'monotonic_ms'])
def test_forged_preflight_snapshot_is_detected(name, field):
    row, raw = fixture()
    snapshot = raw['exploration'][name]
    if field == 'utc':
        snapshot[field] = utc(snapshot['wall_ns'] + 1)
        key = 'utc_ns_matches'
    elif field == 'monotonic_ms':
        snapshot[field] = -1
        key = 'monotonic_order'
    else:
        snapshot[field] += 1
        key = 'exploration_clock_matches'
    assert row_clock_checks(row, raw)[key] is False


def test_stream_events_cannot_precede_last_preflight_monotonic_snapshot():
    row, raw = fixture()
    raw['events'][0]['received_ms'] = 250
    assert row_clock_checks(row, raw)['monotonic_order'] is False


def test_event_wall_validity_inherits_preflight_anomaly():
    row, raw = fixture(anomaly='reversed')
    raw['events'][0]['received_wall_ms'] = raw['events'][0]['observed_received_wall_ms']
    assert row_clock_checks(row, raw)['event_wall_matches'] is False


@pytest.mark.parametrize('missing', ['started', 'preview_received'])
def test_incomplete_preflight_prefix_cannot_hide_observed_phase(missing):
    row, raw = fixture()
    del raw['exploration'][missing]
    assert row_clock_checks(row, raw)['exploration_order_matches'] is False


def test_missing_snapshot_field_is_an_audit_failure():
    row, raw = fixture()
    del raw['exploration']['preview_received']['wall_ns']
    assert row_clock_checks(row, raw)['exploration_clock_matches'] is False


@pytest.mark.parametrize('forged', ['stream_end', 'stream_elapsed', 'event', 'first_answer', 'success'])
def test_absent_stream_does_not_accept_fabricated_timing(forged):
    row, raw = fixture(no_stream=True, prefix=2)
    if forged == 'stream_end':
        raw['stream_finished_wall_ns'] = START + 500_000_000
        raw['stream_finished_utc'] = utc(raw['stream_finished_wall_ns'])
    elif forged == 'stream_elapsed':
        row['stream_total_ms'] = 0
    elif forged == 'event':
        raw['events'] = [{'event': 'answer_delta', 'data': {}, 'received_ms': 500,
                          'received_wall_ns': START + 500_000_000,
                          'received_utc': utc(START + 500_000_000), 'received_wall_ms': 500,
                          'observed_received_wall_ms': 500}]
    elif forged == 'first_answer':
        row['first_answer_ms'] = 0
        row['first_answer_wall_ms'] = 0
    else:
        row['failure'] = raw['failure'] = None
    checks = row_clock_checks(row, raw)
    assert not all(checks.values())


def test_row_preview_timing_must_match_raw_snapshot():
    row, raw = fixture()
    row['exploration']['preview_ms'] -= 1
    assert row_clock_checks(row, raw)['exploration_clock_matches'] is False
