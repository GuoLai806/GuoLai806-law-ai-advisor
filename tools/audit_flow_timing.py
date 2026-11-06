"""Recompute observable timing from UTC/ns records without using runner helpers."""
from datetime import datetime,timezone
from math import isclose
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
    for event in raw['events']:sequence.observe(event['received_wall_ns'])
    sequence.observe(raw['stream_finished_wall_ns'])
    raw_anomalies=list(sequence.anomalies)
    sequence.observe(end)
    valid=not sequence.anomalies;difference=observed-row['user_total_ms'] if observed is not None else None
    checks={'utc_ns_matches':ns(row['started_utc'])==start and ns(row['finished_utc'])==end,
            'raw_start_matches':raw['started_wall_ns']==start and raw['started_utc']==row['started_utc'],
            'wall_total_matches':same(row['observed_user_wall_ms'],observed) and same(row['user_total_wall_ms'],observed if valid else None),
            'difference_matches':same(row['wall_minus_monotonic_ms'],difference),
            'clock_flags_match':row['clock_anomalies']==sequence.anomalies and raw['clock_anomalies']==raw_anomalies
                 and row['wall_clock_valid']==valid and row['wall_monotonic_divergence']==(abs(difference)>1000 if difference is not None else None),
            'stream_matches':same(row['stream_total_ms'],raw['stream_total_ms'])
               and ns(raw['stream_finished_utc'])==raw['stream_finished_wall_ns']
               and same(raw['stream_observed_wall_ms'],elapsed(start,raw['stream_finished_wall_ns']))
               and same(raw['stream_total_wall_ms'],elapsed(start,raw['stream_finished_wall_ns']) if not raw_anomalies else None)
               and same(row['stream_total_wall_ms'],raw['stream_total_wall_ms'] if valid else None),
            'monotonic_order':row['user_total_ms']>=raw['stream_total_ms']>=0}
    previous=0
    event_sequence=WallSequence(start)
    for event in raw['events']:
        event_sequence.observe(event['received_wall_ns'])
        event_observed=elapsed(start,event['received_wall_ns'])
        checks['utc_ns_matches'] &= ns(event['received_utc'])==event['received_wall_ns']
        checks['monotonic_order'] &= event['received_ms']>=previous and raw['stream_total_ms']>=event['received_ms']
        previous=event['received_ms']
        checks['event_wall_matches']=checks.get('event_wall_matches',True) and same(event['observed_received_wall_ms'],event_observed)
        checks['event_wall_matches'] &= same(event['received_wall_ms'],event_observed if not event_sequence.anomalies else None)
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
