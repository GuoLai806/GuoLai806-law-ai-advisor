import json

import pytest

from benchmarks import run_user_flow
from test_short_user_flow import full_scifact_fixture, run_fixture_protocol


class ControlledClock:
    """The third sample advances wall by 130s and monotonic by 10s."""
    def __init__(self, reverse=False, missing=False):
        self.calls = 0
        self.monotonic = 100.0
        self.wall = 1794009600_000000000
        self.reverse = reverse
        self.missing = missing

    def __call__(self):
        self.calls += 1
        if self.calls == 3:
            self.monotonic += 10
            self.wall += -500_000_000 if self.reverse else 130_000_000_000
        return self.monotonic, None if self.missing else self.wall


@pytest.mark.asyncio
async def test_wall_and_monotonic_divergence_is_preserved(full_scifact_fixture, monkeypatch):
    f = full_scifact_fixture
    monkeypatch.setattr(run_user_flow, 'clock_sample', ControlledClock(), raising=False)
    _, report, _ = await run_fixture_protocol(f, monkeypatch, f.short)
    first = report['rows'][0]
    assert first['user_total_ms'] == 10_000
    assert first['user_total_wall_ms'] == 130_000
    assert first['stream_total_wall_ms'] == 130_000
    assert first['first_answer_wall_ms'] == 130_000
    assert first['wall_minus_monotonic_ms'] == 120_000
    assert first['wall_clock_valid'] is True
    assert first['wall_monotonic_divergence'] is True
    assert report['timing_health']['wall_monotonic_divergence_rows'] == 1
    assert report['run_wall_ms'] == 130_000
    raw = json.loads(next((f.tmp / 'raw').glob('*-' + first['query_id'] + '.json')).read_text())
    assert raw['started_utc'] == first['started_utc']
    assert raw['events'][0]['received_wall_ms'] == 130_000
    assert raw['events'][0]['received_ms'] == 10_000
    assert raw['events'][0]['received_utc']
    assert report['clock_identity']['monotonic']['implementation']
    assert report['wall_latency_ms']['user_total_wall_ms']['n'] == 30


@pytest.mark.asyncio
async def test_failed_requests_still_keep_calendar_times(full_scifact_fixture, monkeypatch):
    f = full_scifact_fixture
    monkeypatch.setattr(run_user_flow, 'clock_sample', ControlledClock(), raising=False)
    _, report, calls = await run_fixture_protocol(f, monkeypatch, f.short, fail_stream=True)
    assert len(calls) == report['failures'] == 30
    first = report['rows'][0]
    assert first['started_utc'] and first['finished_utc']
    assert first['user_total_wall_ms'] == first['stream_total_wall_ms'] == 130_000
    assert first['first_answer_wall_ms'] is None
    assert first['first_answer_ms'] is None
    assert first['failure']['type'] == 'HTTPStatusError'
    assert first['predicted_label'] is None
    assert report['wall_latency_ms']['first_answer_wall_ms'] == {'n': 0, 'p50': None, 'p95': None}


@pytest.mark.asyncio
async def test_wall_clock_reversal_is_not_clamped_or_hidden(full_scifact_fixture, monkeypatch):
    f = full_scifact_fixture
    monkeypatch.setattr(run_user_flow, 'clock_sample', ControlledClock(reverse=True), raising=False)
    _, report, _ = await run_fixture_protocol(f, monkeypatch, f.short)
    first = report['rows'][0]
    assert first['wall_clock_valid'] is False
    assert first['user_total_wall_ms'] is None
    assert first['observed_user_wall_ms'] == -500
    assert first['wall_minus_monotonic_ms'] == -10_500
    assert 'wall_clock_reversed' in first['clock_anomalies']
    assert report['timing_health']['clock_anomaly_rows'] == 1
    assert report['wall_latency_ms']['user_total_wall_ms']['n'] == 29
    raw = json.loads(next((f.tmp / 'raw').glob('*-' + first['query_id'] + '.json')).read_text())
    assert raw['events'][0]['received_wall_ms'] is None
    assert raw['events'][0]['observed_received_wall_ms'] == -500
    assert len(report['rows']) == 30  # clock anomaly cannot filter a quality row


@pytest.mark.asyncio
async def test_missing_wall_clock_is_unknown_not_zero(full_scifact_fixture, monkeypatch):
    f = full_scifact_fixture
    monkeypatch.setattr(run_user_flow, 'clock_sample', ControlledClock(missing=True), raising=False)
    _, report, _ = await run_fixture_protocol(f, monkeypatch, f.short)
    assert all(row['user_total_wall_ms'] is None and row['started_utc'] is None for row in report['rows'])
    assert report['run_wall_ms'] is None
    assert report['wall_latency_ms']['user_total_wall_ms'] == {'n': 0, 'p50': None, 'p95': None}
    assert report['timing_health']['all_wall_clocks_valid'] is False


@pytest.mark.asyncio
async def test_run_observes_clock_reversal_between_queries(full_scifact_fixture, monkeypatch):
    class GapClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            # Report start, row start, four events, stream end, row end: 8 samples.
            return 100 + self.calls, 1794009600_000000000 + (0 if self.calls < 9 else -500_000_000)

    f = full_scifact_fixture
    monkeypatch.setattr(run_user_flow, 'clock_sample', GapClock(), raising=False)
    _, report, _ = await run_fixture_protocol(f, monkeypatch, f.short)
    assert 'wall_clock_reversed' in report['run_clock_anomalies']
    assert report['run_wall_ms'] is None
    assert report['run_observed_wall_ms'] == -500
    assert report['timing_health']['all_wall_clocks_valid'] is False
    assert len(report['rows']) == 30


@pytest.mark.asyncio
async def test_single_missing_event_clock_invalidates_wall_summary(full_scifact_fixture, monkeypatch):
    class MissingOnceClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return 100 + self.calls, None if self.calls == 3 else 1794009600_000000000

    f = full_scifact_fixture
    monkeypatch.setattr(run_user_flow, 'clock_sample', MissingOnceClock(), raising=False)
    _, report, _ = await run_fixture_protocol(f, monkeypatch, f.short)
    first = report['rows'][0]
    assert first['started_utc'] and first['finished_utc']
    assert first['user_total_wall_ms'] is None
    assert first['first_answer_wall_ms'] is None
    assert first['clock_anomalies'] == ['wall_clock_missing']
    assert report['wall_latency_ms']['user_total_wall_ms']['n'] == 29
    assert 'wall_clock_missing' in report['run_clock_anomalies']
