"""Safe, auditable acquisition of public Chinese legal source files.

Network calls are explicit; no redirect to a different host, credentials, or
unknown response body is accepted.  Raw bytes and source metadata are returned
to a caller that owns persistence.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from urllib.parse import urlparse
import httpx
from html.parser import HTMLParser
from urllib.parse import urljoin,urlparse

MAX_DOWNLOAD_BYTES = 50_000_000


class SourceAcquisitionError(RuntimeError):
    def __init__(self, reason, *, host=None):
        self.reason=reason;self.host=host
        super().__init__(reason)
ALLOWED_SCHEMES = {'https'}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    authority: str
    published_at: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    revision_note: str | None = None

    def validate(self):
        parsed=urlparse(self.url)
        if parsed.scheme not in ALLOWED_SCHEMES or not parsed.hostname:
            raise ValueError('source URL must use HTTPS and include a host')
        if not self.name or not self.authority:
            raise ValueError('source name and authority are required')
        for field in ('published_at','effective_from','effective_to'):
            value=getattr(self,field)
            if value:
                datetime.fromisoformat(value)
        return self


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _same_or_subdomain(actual, allowed):
    return actual == allowed or actual.endswith('.'+allowed)


async def acquire_source(spec: SourceSpec, *, transport=None, max_redirects=3) -> dict:
    """Fetch one bounded public source and reject redirects outside the source host."""
    spec.validate()
    parsed=urlparse(spec.url)
    allowed_host=parsed.hostname.lower()
    async with httpx.AsyncClient(timeout=30,transport=transport,trust_env=False,follow_redirects=False) as client:
        current=spec.url
        history=[]
        for _ in range(max_redirects+1):
            response=await client.get(current,headers={'Accept':'application/pdf,text/html,text/plain'})
            if response.status_code in (301,302,303,307,308):
                target=response.headers.get('location')
                if not target: raise ValueError('redirect without location')
                target=httpx.URL(current).join(target)
                if target.scheme!='https' or not target.host or not _same_or_subdomain(target.host.lower(),allowed_host):
                    raise ValueError('redirect left the authorized source host')
                history.append(current);current=str(target);continue
            response.raise_for_status()
            if len(response.content)>MAX_DOWNLOAD_BYTES: raise ValueError('source exceeds download limit')
            content_type=response.headers.get('content-type','').split(';',1)[0].strip().lower()
            if not content_type.startswith(('application/pdf','text/html','text/plain')):
                raise ValueError('unsupported source content type')
            fetched_at=datetime.now(timezone.utc).isoformat()
            return {'spec':spec.__dict__,'requested_url':spec.url,'final_url':current,'redirect_history':history,'status':response.status_code,'content_type':content_type,'bytes':response.content,'byte_length':len(response.content),'sha256':sha256_bytes(response.content),'fetched_at':fetched_at}
        raise ValueError('redirect limit exceeded')


def discover_links(acquisition: dict, base_url=None) -> list[str]:
    """Extract same-host HTTPS links from a bounded HTML response.

    Discovery does not imply that a PDF is authoritative. The caller must fetch
    and verify that candidate separately.
    """
    if acquisition.get('content_type')!='text/html':
        return []
    html=acquisition['bytes'].decode('utf-8','replace')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.values=[]
        def handle_starttag(self,tag,attrs):
            if tag.lower()!='a':return
            href=dict(attrs).get('href')
            if href:self.values.append(href)
    parser=Links();parser.feed(html)
    host=urlparse(base_url or acquisition['final_url']).hostname.lower()
    result=[]
    for href in parser.values:
        target=urljoin(base_url or acquisition['final_url'],href)
        parsed=urlparse(target)
        if parsed.scheme=='https' and parsed.hostname and _same_or_subdomain(parsed.hostname.lower(),host):result.append(str(target))
    return list(dict.fromkeys(result))


def parse_npc_search_result(payload: dict, expected_title: str) -> dict:
    """Validate one official-law search result from flk.npc.gov.cn."""
    if payload.get('code')!=200 or not isinstance(payload.get('rows'),list):
        raise ValueError('invalid NPC search envelope')
    title=''.join(jieba.cut_for_search(expected_title,HMM=False)) if False else expected_title
    import re
    matches=[]
    for row in payload['rows']:
        if not isinstance(row,dict) or not row.get('bbbs') or not row.get('title'):continue
        clean=re.sub(r'<[^>]+>','',str(row['title'])).strip()
        if clean==expected_title:matches.append(row)
    if len(matches)!=1:raise ValueError('expected exactly one exact NPC title match')
    row=matches[0]
    for key in ('gbrq','sxrq','zdjgName','sxx'):
        if not row.get(key):raise ValueError(f'NPC result missing {key}')
    return {'bbbs':row['bbbs'],'title':row['title'],'published_at':row['gbrq'],'effective_from':row['sxrq'],'authority':row['zdjgName'],'validity_code':row['sxx'],'category':row.get('flxz')}


def npc_content_articles(content_root: dict) -> list[dict]:
    """Traverse the official detail tree into article labels with structure."""
    if not isinstance(content_root,dict):raise ValueError('NPC content tree missing')
    output=[]
    def walk(node,path):
        if not isinstance(node,dict):raise ValueError('invalid NPC content node')
        title=str(node.get('title') or '').replace('\u200b','').strip()
        next_path=path
        if title.startswith('第') and title.endswith('条'):
            output.append({'label':title,'structure':list(path),'external_id':node.get('id'),'index':node.get('index')})
            next_path=path+[title]
        elif title.startswith('第') and any(marker in title for marker in ('编','章','节')) or title.startswith('附则'):
            next_path=path+[title]
        for child in node.get('children') or []:walk(child,next_path)
    walk(content_root,[])
    labels=[x['label'] for x in output]
    if len(labels)!=1260:raise ValueError(f'NPC official tree has {len(labels)} articles, expected 1260')
    if labels[0]!='第一条' or labels[-1]!='第一千二百六十条':raise ValueError('NPC article boundaries mismatch')
    return output


def parse_npc_detail(payload: dict, bbbs: str, expected_title: str) -> dict:
    if payload.get('code')!=200 or not isinstance(payload.get('data'),dict):raise ValueError('invalid NPC detail envelope')
    data=payload['data']
    if data.get('bbbs')!=bbbs or data.get('title')!=expected_title:raise ValueError('NPC detail identity mismatch')
    oss=data.get('ossFile') or {}
    if not isinstance(oss,dict):raise ValueError('NPC detail missing file metadata')
    return {'bbbs':bbbs,'title':data['title'],'published_at':data.get('gbrq'),'effective_from':data.get('sxrq'),'authority':data.get('zdjgName'),'validity_code':data.get('sxx'),'pdf_path':oss.get('ossPdfPath'),'docx_path':oss.get('ossWordPath'),'ofd_path':oss.get('ossPdfOfdPath'),'content_root':data.get('content')}


async def resolve_official_record(title: str, bbbs: str | None=None, *, transport=None) -> dict:
    """Search by exact title or use a known stable official ID."""
    if bbbs:return await fetch_npc_document(bbbs,title,transport=transport)
    return await search_npc_document(title,transport=transport)


async def search_npc_document(title: str, *, transport=None) -> dict:
    """Search the official database, then fetch exact metadata/detail."""
    if not title:raise ValueError('title required')
    async with httpx.AsyncClient(timeout=30,transport=transport,trust_env=False,follow_redirects=False) as client:
        response=await client.post('https://flk.npc.gov.cn/law-search/search/list',json={'searchContent':title,'searchType':1,'searchRange':1,'pageNum':1,'pageSize':20,'orderByParam':{'order':'-1','sort':''}})
        response.raise_for_status();found=parse_npc_search_result(response.json(),title)
    return await fetch_npc_document(found['bbbs'],title,transport=transport)


def derive_version_intervals(records: list[dict]) -> list[dict]:
    """Close an earlier official version when a later version has a valid date.

    The interval is marked inferred_pending because a change in publication
    dates alone does not prove every clause was superseded. It can drive
    development checks but cannot silently mark an article authoritative.
    """
    ordered=sorted(records,key=lambda x:(x.get('effective_from') or '',x.get('published_at') or '',x.get('bbbs','')))
    out=[]
    for index,row in enumerate(ordered):
        item={**row}
        if index+1<len(ordered):
            next_start=ordered[index+1].get('effective_from')
            if next_start:
                from datetime import date,timedelta
                d=date.fromisoformat(next_start)-timedelta(days=1)
                item['effective_to']=d.isoformat();item['interval_state']='inferred_pending'
            else:item['effective_to']=None;item['interval_state']='unknown'
        else:item['effective_to']=None;item['interval_state']='open_last_known'
        out.append(item)
    return out


def verify_official_source_signature(body: bytes, content_type: str, expected_title: str) -> str:
    """Distinguish a PDF/DOCX body from a web shell before acquisition succeeds."""
    ctype=(content_type or '').split(';',1)[0].lower()
    if body.startswith(b'%PDF-'):return 'pdf'
    if body.startswith(b'PK\x03\x04'):
        import io,zipfile
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            names=set(z.namelist())
            if 'word/document.xml' in names:return 'docx'
    if ctype in ('text/html','application/xhtml+xml') or body.lstrip().startswith(b'<!DOCTYPE html') or body.lstrip().startswith(b'<html'):
        raise ValueError('source response is a web shell, not a legal body')
    raise ValueError(f'unsupported source body ({ctype or "unknown"})')


def official_pdf_body_path(acquisition: dict) -> str | None:
    """Return only a body identifier backed by acquired bytes and PDF signature."""
    if acquisition.get('body') and acquisition.get('body_kind')=='pdf':return acquisition.get('metadata',{}).get('pdf_path')
    return None


def official_tree_articles(content_root: dict) -> list[dict]:
    """Return official article order, parents and stable external IDs."""
    output=[]
    def walk(node,path):
        if not isinstance(node,dict):raise ValueError('invalid official content node')
        title=str(node.get('title') or '').replace('\u200b','').strip()
        if title.startswith('第') and title.endswith('条'):
            output.append({'label':title,'structure':list(path),'external_id':node.get('id'),'index':node.get('index')})
        next_path=path
        if title.startswith('第') and any(marker in title for marker in ('编','章','节')) or title=='附则':
            next_path=path+[title]
        for child in node.get('children') or []:walk(child,next_path)
    walk(content_root,[])
    if len(output)!=1260 or output[0]['label']!='第一条' or output[-1]['label']!='第一千二百六十条':
        raise ValueError('official article tree is incomplete')
    return output

def official_snapshot_manifest(acquisition: dict) -> dict:
    """Canonical manifest fields for ingesting acquired official metadata.

    The current NPC endpoint provides dates and a structured article tree even
    when the downloadable body is unavailable. The source review state remains
    pending so it cannot be activated as authoritative content by accident.
    """
    meta=acquisition['metadata']
    from datetime import date
    today=date.today().isoformat()
    # The official database's status code is a point-in-time catalog signal,
    # not by itself proof that no future amendment exists. Record the observation
    # date and leave valid_to open unless a later official version closes it.
    return {'doc_id':'civil-code-npc','version_id':meta['bbbs'],'title':meta['title'], 'issuer':meta['authority'],'validity_observed_at':today,'source_kind':'official-npc-metadata','source_url':'https://flk.npc.gov.cn/law-search/search/flfgDetails?bbbs='+meta['bbbs'],'published_at':meta.get('published_at'),'effective_from':meta.get('effective_from'),'effective_to':None,'verification_state':'pending','revision_note':'metadata and official tree acquired; source body/validity requires separate review','npc_validity_code':meta.get('validity_code'),'source_body_sha256':acquisition.get('body_sha256')}


async def discover_npc_history(title: str, *, transport=None) -> list[dict]:
    """Return same-title records from the official database sorted by effective date."""
    async with httpx.AsyncClient(timeout=30,transport=transport,trust_env=False,follow_redirects=False) as client:
        response=await client.post('https://flk.npc.gov.cn/law-search/search/list',json={'searchContent':title,'searchType':1,'searchRange':1,'pageNum':1,'pageSize':100,'orderByParam':{'order':'-1','sort':''}})
        response.raise_for_status();payload=response.json()
    if payload.get('code')!=200 or not isinstance(payload.get('rows'),list):raise ValueError('invalid NPC history envelope')
    records=[]
    import re
    for row in payload['rows']:
        if not isinstance(row,dict):continue
        clean=re.sub(r'<[^>]+>','',str(row.get('title') or '')).strip()
        if clean!=title:continue
        records.append({'bbbs':row['bbbs'],'title':clean,'published_at':row.get('gbrq'),'effective_from':row.get('sxrq'),'effective_to':None,'authority':row.get('zdjgName'),'validity_code':row.get('sxx'),'category':row.get('flxz')})
    return sorted(records,key=lambda x:(x.get('effective_from') or '',x.get('published_at') or '',x['bbbs']))


async def fetch_npc_document(bbbs: str, expected_title: str, *, transport=None) -> dict:
    """Fetch official metadata/detail from the NPC law database.

    The database currently exposes public metadata but may return an internal
    download host. In that case we return verification_state=metadata_only and
    never pretend a body was acquired.
    """
    if not bbbs or not expected_title:raise ValueError('bbbs and title required')
    async with httpx.AsyncClient(timeout=30,transport=transport,trust_env=False,follow_redirects=True) as client:
        q=await client.get('https://flk.npc.gov.cn/law-search/search/flfgDetails',params={'bbbs':bbbs})
        q.raise_for_status();detail=parse_npc_detail(q.json(),bbbs,expected_title)
        result={'metadata':detail,'body':None,'verification_state':'metadata_only','download_url_present':False,'download_url_host':None,'download_failure':None}
        path=detail.get('pdf_path')
        if not path:return result
        link=await client.get('https://flk.npc.gov.cn/law-search/amazonFile/previewLink',params={'filePath':path});link.raise_for_status();payload=link.json()
        if payload.get('code')!=200 or not isinstance(payload.get('data'),dict):raise ValueError('invalid NPC preview envelope')
        url=payload['data'].get('url') or ''
        result['download_url_present']=bool(url)
        parsed=urlparse(url)
        result['download_url_host']=parsed.hostname
        candidates=[url]
        if url.startswith('http://'):
            candidates.append('https'+url[4:])
        if '-internal.' in url:
            candidates.append(url.replace('-internal.','.',1))
        failure='official download URL did not return a usable PDF'
        for candidate in dict.fromkeys(candidates):
            try:
                raw=await client.get(candidate);raw.raise_for_status()
                if len(raw.content)>MAX_DOWNLOAD_BYTES:raise ValueError('source exceeds download limit')
                content_type=raw.headers.get('content-type','').split(';',1)[0].lower()
                kind=verify_official_source_signature(raw.content,content_type,expected_title)
                result.update(body=raw.content,body_sha256=sha256_bytes(raw.content),body_content_type=content_type,body_kind=kind,verification_state='body_acquired',download_url_host=urlparse(candidate).hostname)
                return result
            except Exception as exc:
                failure=f'{type(exc).__name__}: {str(exc)[:160]}'
        result['download_failure']=failure
        return result


def public_state(acquisition: dict) -> dict:
    """Return publishable provenance without raw source content."""
    return {k:v for k,v in acquisition.items() if k!='bytes'}
