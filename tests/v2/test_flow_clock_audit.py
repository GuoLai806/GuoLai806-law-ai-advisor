import json
import pytest
from test_short_user_flow import full_scifact_fixture,run_fixture_protocol
from test_wall_clock_flow import ControlledClock
from benchmarks import run_user_flow
from tools.audit_flow_timing import row_clock_checks,report_clock_checks


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['normal','gap','reversed','missing','failure'])
async def test_observed_clock_records_recompute_including_anomalies(full_scifact_fixture,monkeypatch,kind):
    f=full_scifact_fixture
    if kind!='normal':monkeypatch.setattr(run_user_flow,'clock_sample',ControlledClock(reverse=kind=='reversed',missing=kind=='missing'))
    _,report,_=await run_fixture_protocol(f,monkeypatch,f.short,fail_stream=kind=='failure')
    assert all(report_clock_checks(report).values())
    for row in report['rows']:
        raw=json.loads((f.tmp/'raw'/('report-short-'+row['query_id']+'.json')).read_text())
        assert all(row_clock_checks(row,raw).values())
        if kind=='reversed' and row['clock_anomalies']:
            from copy import deepcopy
            forged=deepcopy(row);forged.update(clock_anomalies=[],wall_clock_valid=True,
                                               user_total_wall_ms=row['observed_user_wall_ms'])
            assert not row_clock_checks(forged,raw)['clock_flags_match']
    report['wall_latency_ms']['user_total_wall_ms']['p95']=-1_000_000
    assert not report_clock_checks(report)['user_total_wall_ms_quantiles']
