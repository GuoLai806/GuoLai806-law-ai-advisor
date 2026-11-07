"""Recompute observable timing from UTC/ns records without using runner helpers."""
from datetime import datetime,timezone
from math import isclose, isfinite
import re


def ns(utc):
    if utc is None:return None
    match=re.fullmatch(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d{9})Z',utc)
    if not match:raise ValueError('Expected nanosecond UTC snapshot')
    return int(datetime.strptime(match[1],'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp())*1_000_000_000+int(match[2])


def same(a,b):
    return a is None and b is None or isinstance(a,(int,float)) and isinstance(b,(int,float)) and isclose(a,b,rel_tol=0,abs_tol=1e-6)


def elapsed(start,end):return (end-start)/1_000_000 if start is not None and end is not None else None


def quantile(values,q):
    values=sorted(values)
    return values[min(len(values)-1,max(0,int((len(values)-1)*q)))] if values else None


class WallSequence:
    def __init__(self,start):
        self.last=start
        self.anomalies=[] if start is not None else ['wall_clock_missing']
    def observe(self,value):
        reason='wall_clock_missing' if value is None else 'wall_clock_reversed' if self.last is not None and value<self.last else None
        if reason and reason not in self.anomalies:self.anomalies.append(reason)
        if value is not None:self.last=value


def row_clock_checks(row,raw):
    start=row['started_wall_ns'];end=row['finished_wall_ns'];observed=elapsed(start,end)
    sequence=WallSequence(start)

    def numeric(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and isfinite(value)

    def utc_matches(utc,value):
        try:return ns(utc)==value
        except (ValueError,TypeError,OverflowError):return False

    checks={'utc_ns_matches':utc_matches(row['started_utc'],start) and utc_matches(row['finished_utc'],end),
            'raw_start_matches':raw['started_wall_ns']==start and raw['started_utc']==row['started_utc'],
            'monotonic_order':numeric(row['user_total_ms']) and row['user_total_ms']>=0,
            'exploration_clock_matches':True,'exploration_order_matches':True,
            'event_wall_matches':True}
    previous=0
    exploration=raw.get('exploration')
    sample_names=('started','preview_received','original_received')
    if exploration is not None:
        if not isinstance(exploration,dict):
            checks['exploration_clock_matches']=False
        else:
            present=[name for name in sample_names if name in exploration]
            # Failed previews/deep reads retain only the completed prefix. An
            # original snapshot without a preview/start snapshot is not valid.
            checks['exploration_order_matches']=present==list(sample_names[:len(present)])
            for name in present:
                sample=exploration[name]
                required={'wall_ns','utc','monotonic_ms','wall_ms','observed_wall_ms'}
                if not isinstance(sample,dict) or not required.issubset(sample):
                    checks['exploration_clock_matches']=False
                    continue
                sequence.observe(sample['wall_ns'])
                sample_observed=elapsed(start,sample['wall_ns'])
                checks['utc_ns_matches'] &= utc_matches(sample['utc'],sample['wall_ns'])
                checks['exploration_clock_matches'] &= same(sample['observed_wall_ms'],sample_observed)
                checks['exploration_clock_matches'] &= same(sample['wall_ms'],sample_observed if not sequence.anomalies else None)
                ordered=numeric(sample['monotonic_ms']) and previous<=sample['monotonic_ms']<=row['user_total_ms']
                checks['monotonic_order'] &= ordered
                if numeric(sample['monotonic_ms']):previous=sample['monotonic_ms']
            if row.get('exploration'):
                summary=row['exploration']
                checks['exploration_clock_matches'] &= (
                    same(summary.get('preview_ms'),(exploration.get('preview_received') or {}).get('monotonic_ms'))
                    and same(summary.get('original_ms'),(exploration.get('original_received') or {}).get('monotonic_ms')))

    for event in raw['events']:
        sequence.observe(event['received_wall_ns'])
        event_observed=elapsed(start,event['received_wall_ns'])
        checks['utc_ns_matches'] &= utc_matches(event['received_utc'],event['received_wall_ns'])
        checks['monotonic_order'] &= numeric(event['received_ms']) and previous<=event['received_ms']<=row['user_total_ms']
        if numeric(event['received_ms']):previous=event['received_ms']
        checks['event_wall_matches'] &= same(event['observed_received_wall_ms'],event_observed)
        checks['event_wall_matches'] &= same(event['received_wall_ms'],event_observed if not sequence.anomalies else None)

    stream_total=raw.get('stream_total_ms')
    if stream_total is not None:
        sequence.observe(raw['stream_finished_wall_ns'])
        checks['monotonic_order'] &= numeric(stream_total) and previous<=stream_total<=row['user_total_ms']
    raw_anomalies=list(sequence.anomalies)
    sequence.observe(end)
    valid=not sequence.anomalies
    difference=observed-row['user_total_ms'] if observed is not None else None
    checks.update(
        wall_total_matches=same(row['observed_user_wall_ms'],observed) and same(row['user_total_wall_ms'],observed if valid else None),
        difference_matches=same(row['wall_minus_monotonic_ms'],difference),
        clock_flags_match=row['clock_anomalies']==sequence.anomalies and raw['clock_anomalies']==raw_anomalies
            and row['wall_clock_valid']==valid and row['wall_monotonic_divergence']==(abs(difference)>1000 if difference is not None else None))
    if stream_total is None:
        # A preflight failure did not sample a stream end. Missing *observed*
        # clocks remain anomalies; a phase that never started is simply absent.
        checks['stream_matches']=(row.get('stream_total_ms') is None
            and all(raw.get(field) is None for field in ('stream_finished_utc','stream_finished_wall_ns',
                                                       'stream_total_wall_ms','stream_observed_wall_ms'))
            and row['stream_total_wall_ms'] is None and not raw['events']
            and bool(row.get('failure')) and bool(raw.get('failure')))
    else:
        stream_observed=elapsed(start,raw['stream_finished_wall_ns'])
        checks['stream_matches']=(same(row.get('stream_total_ms'),stream_total)
            and utc_matches(raw['stream_finished_utc'],raw['stream_finished_wall_ns'])
            and same(raw['stream_observed_wall_ms'],stream_observed)
            and same(raw['stream_total_wall_ms'],stream_observed if not raw_anomalies else None)
            and same(row['stream_total_wall_ms'],raw['stream_total_wall_ms'] if valid else None))
    first=next((e for e in raw['events'] if e['event']=='answer_delta'),None)
    checks['first_answer_matches']=same(row['first_answer_ms'],first['received_ms'] if first else None) and same(
        row['first_answer_wall_ms'],first['received_wall_ms'] if first and valid else None)
    return checks


def report_clock_checks(report):
    rows=report['rows']; checks={}
    for field in ('stream_total_wall_ms','user_total_wall_ms','first_answer_wall_ms'):
        values=[row[field] for row in rows if row.get('wall_clock_valid') and row.get(field) is not None]
        stored=report['wall_latency_ms'][field]
        checks[field+'_quantiles']=stored['n']==len(values) and same(stored['p50'],quantile(values,.5)) and same(stored['p95'],quantile(values,.95))
    health=report['timing_health']
    checks['timing_health']=health['wall_monotonic_divergence_rows']==sum(r['wall_monotonic_divergence'] is True for r in rows)
    checks['timing_health'] &= health['clock_anomaly_rows']==sum(bool(r['clock_anomalies']) for r in rows)
    checks['timing_health'] &= health['all_wall_clocks_valid']==(not report['run_clock_anomalies'] and all(r['wall_clock_valid'] for r in rows))
    observed=elapsed(ns(report['started_utc']),ns(report['finished_utc']))
    checks['run_clock_snapshot']=same(report['run_observed_wall_ms'],observed) and same(report['run_wall_ms'],observed if not report['run_clock_anomalies'] else None)
    difference=observed-report['run_monotonic_ms'] if observed is not None else None
    checks['run_clock_snapshot'] &= same(report['run_wall_minus_monotonic_ms'],difference)
    sequence=WallSequence(ns(report['started_utc']))
    for row in rows:
        sequence.observe(row['started_wall_ns'])
        for anomaly in row['clock_anomalies']:
            if anomaly not in sequence.anomalies:sequence.anomalies.append(anomaly)
        sequence.observe(row['finished_wall_ns'])
    sequence.observe(ns(report['finished_utc']))
    checks['run_anomalies_reconstructed']=report['run_clock_anomalies']==sequence.anomalies
    return checks
