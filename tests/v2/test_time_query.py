from datetime import date
from evidence_rag.time_query import extract_event_date

def test_absolute_chinese_and_iso_dates_are_extracted():
 assert extract_event_date('我在2021年6月1日签的合同').value==date(2021,6,1)
 assert extract_event_date('事件发生于2020-12-31').value==date(2020,12,31)

def test_partial_or_absent_dates_are_not_invented():
 assert extract_event_date('2021年发生的事情').ambiguity=='year_without_month_day'
 assert extract_event_date('2021年6月发生').ambiguity=='month_without_day'
 assert extract_event_date('合同纠纷怎么办').value is None

def test_relative_dates_are_explicitly_relative():
 out=extract_event_date('三年前的买卖合同',query_time=date(2026,1,1))
 assert out.value==date(2023,1,2) and out.kind=='relative_approximate'
