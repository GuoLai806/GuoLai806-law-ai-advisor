from evidence_rag.legal_versions import select_version

def rows():
 return [
 {'revision':'old','valid_from':'2010-01-01','valid_to':'2020-12-31','published_at':'2009-12-01','verification_state':'approved','source_kind':'law'},
 {'revision':'new','valid_from':'2021-01-01','valid_to':None,'published_at':'2020-05-28','verification_state':'approved','source_kind':'law'},
 {'revision':'unknown','valid_from':None,'valid_to':None,'published_at':None,'verification_state':'approved','source_kind':'law'},
 {'revision':'pending','valid_from':'2019-01-01','valid_to':None,'published_at':'2018-01-01','verification_state':'pending','source_kind':'law'},
]

def test_selects_event_date_version_not_latest_upload():
 assert select_version(rows(),'2015-06-01')['selected']=='old'
 assert select_version(rows(),'2021-06-01')['selected']=='new'

def test_event_before_any_interval_does_not_fall_forward():
 r=select_version(rows(),'2000-01-01')
 assert r['selected'] is None and r['reason']=='no_verified_revision'

def test_current_question_uses_query_date_and_never_unknown():
 r=select_version(rows(),query_time='2022-01-01')
 assert r['selected']=='new' and all(x['revision']!='unknown' for x in r['candidates'])
