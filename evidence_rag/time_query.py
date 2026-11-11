"""Extract explicit or relative Chinese legal-event time from a user query.

The extractor is deliberately conservative: when no unambiguous reference is
available it returns unknown, allowing the API to ask for the event date. It
never silently substitutes the most recent legal version for a historical event.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
import unicodedata

from .temporal import FULL_DATE_RE

CN_MONTH = re.compile(r'(?P<year>(?:19|20)\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日')
CN_MONTH_ONLY = re.compile(r'(?P<year>(?:19|20)\d{2})年(?P<month>\d{1,2})月')
YEAR = re.compile(r'(?P<year>(?:19|20)\d{2})年')
RELATIVE_DAYS = re.compile(r'(?P<n>[一二两三四五六七八九十百\d]{1,4})天(?:前|以前)')
RELATIVE_MONTHS = re.compile(r'(?P<n>[一二两三四五六七八九十百\d]{1,4})个?月(?:前|以前)')
RELATIVE_YEARS = re.compile(r'(?P<n>[一二两三四五六七八九十百\d]{1,4})年(?:前|以前)')


@dataclass(frozen=True)
class ExtractedTime:
    value: date | None
    kind: str
    ambiguity: str | None = None


def _cn_number(text):
    text=unicodedata.normalize('NFKC',text)
    if text.isdigit():return int(text)
    digits={'零':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    units={'十':10,'百':100}
    total=0;current=0
    for ch in text:
        if ch in digits:current=digits[ch]
        elif ch in units:
            n=units[ch];total+=(current or 1)*n;current=0
        else:raise ValueError('invalid number')
    return total+current


def _subtract_months(day, months):
    total=day.year*12+day.month-1-months
    year,month=divmod(total,12);month+=1
    # Clamp to the same day or final valid day of the target month.
    for candidate in range(day.day,27,-1):
        try:return date(year,month,candidate)
        except ValueError:continue
    return date(year,month,1)


def extract_event_date(query, query_time=None):
    text=unicodedata.normalize('NFKC',query or '')
    now=query_time or date.today()
    if isinstance(now,datetime):now=now.date()
    if isinstance(now,str):now=date.fromisoformat(now)
    exact=FULL_DATE_RE.search(text) or CN_MONTH.search(text)
    if exact:
        try:
            if hasattr(exact,'groupdict'):
                year=int(exact.group(1));month=int(exact.group(2));day=int(exact.group(3))
            else:year=int(re.search(r'(?:19|20)\d{2}',exact.group()).group());month=int(exact.group('month'));day=int(exact.group('day'))
            return ExtractedTime(date(year,month,day),'absolute')
        except ValueError:return ExtractedTime(None,'invalid','invalid_date')
    month=CN_MONTH_ONLY.search(text)
    if month:return ExtractedTime(None,'partial','month_without_day')
    year=YEAR.search(text)
    if year:return ExtractedTime(None,'partial','year_without_month_day')
    match=RELATIVE_DAYS.search(text)
    if match:
        try:return ExtractedTime(now-timedelta(days=_cn_number(match.group('n'))),'relative')
        except ValueError:return ExtractedTime(None,'invalid','relative_number')
    match=RELATIVE_MONTHS.search(text)
    if match:
        try:return ExtractedTime(_subtract_months(now,_cn_number(match.group('n'))),'relative')
        except ValueError:return ExtractedTime(None,'invalid','relative_number')
    match=RELATIVE_YEARS.search(text)
    if match:
        try:return ExtractedTime(now-timedelta(days=365*_cn_number(match.group('n'))),'relative_approximate')
        except ValueError:return ExtractedTime(None,'invalid','relative_number')
    return ExtractedTime(None,'unknown')
