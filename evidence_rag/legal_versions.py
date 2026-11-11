"""Choose the applicable legal revision after source/date verification.

A revision with an explicit verified interval can become active for a query
date. Unknown dates never win. Revision selection is auditable and kept separate
from article relevance.
"""
from dataclasses import dataclass
from datetime import date
from .temporal import as_date, validity


@dataclass(frozen=True)
class VersionRecord:
    revision: str
    valid_from: date | None
    valid_to: date | None
    published_at: date | None
    verification_state: str
    source_kind: str | None = None
    revision_note: str | None = None


def version_records(rows):
    out=[]
    for row in rows:
        def parse(value):
            return as_date(value) if value not in (None,'') else None
        out.append(VersionRecord(str(row['revision']),parse(row.get('valid_from')),parse(row.get('valid_to')),parse(row.get('published_at')),row.get('verification_state') or 'unknown',row.get('source_kind'),row.get('revision_note')))
    return out


def select_version(rows, event_date=None, query_time=None):
    """Return the latest eligible revision valid at the event date.

    event_date is the default legal reference. query_time is used only when no
    event date exists and only for sources whose metadata explicitly describes
    current validity. Synthetic fixtures bypass this selector.
    """
    # A general question with no event date should use current validity, not
    # a caller-supplied historical query clock when available.
    reference=as_date(event_date) if event_date else as_date(query_time or date.today())
    candidates=[]
    for record in version_records(rows):
        if record.source_kind=='official-npc-metadata' and record.verification_state!='approved':
            continue
        if record.verification_state!='approved':continue
        if record.valid_from is None and record.valid_to is None:continue
        state=validity(reference,record.valid_from,record.valid_to,verified=True)
        if state=='valid':
            candidates.append((record.valid_from or date.min, record.revision, state, record))
    if not candidates:return {'selected':None,'reference_date':reference.isoformat(),'reason':'no_verified_revision','candidates':[]}
    _,revision,state,record=max(candidates,key=lambda x:(x[0],x[1]))
    return {'selected':revision,'reference_date':reference.isoformat(),'reason':'latest_valid_interval','valid_from':record.valid_from.isoformat() if record.valid_from else None,'valid_to':record.valid_to.isoformat() if record.valid_to else None,'candidates':[{'revision':x[1],'valid_from':x[3].valid_from.isoformat() if x[3].valid_from else None,'valid_to':x[3].valid_to.isoformat() if x[3].valid_to else None} for x in sorted(candidates)]}
