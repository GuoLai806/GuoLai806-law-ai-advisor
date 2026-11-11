from datetime import date
import pytest
from evidence_rag.temporal import TemporalConfig, TimeContext, candidate_time_score, interval_contains, rank_temporal, temporal_compatibility, temporal_dates, validity

def test_validity_is_interval_based_and_unknown_without_dates():
    day=date(2022,1,1)
    assert validity(day,'2021-01-01',None)=='valid'
    assert validity(day,'2022-01-02',None)=='not_yet_effective'
    assert validity(day,None,'2021-12-31')=='expired'
    assert validity(day,None,None)=='unknown'
    assert validity(day,None,None,verified=False)=='unknown'
    assert interval_contains(day,'2021-01-01','2022-01-01')
    with pytest.raises(ValueError): interval_contains(day,'2022-01-02','2022-01-01')

def test_event_date_wins_over_query_date_and_valid_beats_expired():
    context=TimeContext(date(2030,1,1),date(2021,6,1))
    old=temporal_compatibility(context,valid_from='2020-01-01',valid_to='2020-12-31',eligible=True)
    new=temporal_compatibility(context,valid_from='2021-01-01',valid_to=None,eligible=True)
    assert old['reference_source']=='event_date' and old['validity']=='expired'
    assert new['validity']=='valid' and new['temporal_affinity']==1
    assert candidate_time_score(.8,new)['final']>candidate_time_score(.8,old)['final']

def test_time_distance_and_missing_interval_are_explicit():
    context=TimeContext(date(2023,1,1),None)
    near=temporal_compatibility(context,valid_from='2022-12-30',valid_to='2022-12-31',eligible=True)
    far=temporal_compatibility(context,valid_from='2010-01-01',valid_to='2010-12-31',eligible=True)
    missing=temporal_compatibility(context,valid_from=None,valid_to=None,eligible=True)
    assert near['distance_days']==1 and far['distance_days']>near['distance_days']
    assert near['temporal_affinity']>far['temporal_affinity']
    assert missing['temporal_affinity'] is None and missing['validity']=='unknown'

def test_date_extraction_does_not_invent_partial_dates():
    assert temporal_dates('2020年5月28日，2021-01-01，2022/03') == [date(2020,5,28),date(2021,1,1)]

def test_rank_keeps_invalid_for_audit_but_never_eligible():
    rows=[{'unit_id':'old','semantic_score':.9,'valid_from':'2020-01-01','valid_to':'2020-12-31','verified':True},{'unit_id':'new','semantic_score':.8,'valid_from':'2021-01-01','valid_to':None,'verified':True}]
    out=rank_temporal(rows,TimeContext(date(2022,1,1),date(2021,1,1)))
    assert out[0]['unit_id']=='new' and out[0]['temporal']['eligible']
    assert not out[1]['temporal']['eligible'] and out[1]['temporal']['validity']=='expired'
