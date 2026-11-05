"""Neutral input adapters. Each document retains its ID, source and character locators."""
import hashlib, json, os, stat
from pathlib import Path

def load_documents(path, fmt='auto', doc_id=None, title=None, source_url=None):
    path=Path(path)
    if path.stat().st_size>100_000_000: raise ValueError('input exceeds 100 MB')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as source:
        info=os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode): raise ValueError('input must be a regular file')
        if info.st_size>100_000_000: raise ValueError('input exceeds 100 MB')
        raw=source.read(100_000_001)
    if len(raw)>100_000_000: raise ValueError('input exceeds 100 MB')
    if fmt=='auto':fmt={'.jsonl':'jsonl','.txt':'text','.md':'text','.pdf':'pdf'}.get(path.suffix.lower(),'')
    if fmt=='jsonl':
        documents=[]
        for line_no,line in enumerate(raw.decode('utf-8-sig').splitlines(),1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f'invalid JSONL line {line_no}') from exc
            if not isinstance(row,dict): raise ValueError(f'JSONL line {line_no} must be an object')
            documents.append(row)
    elif fmt=='text':
        documents=[{'_id':doc_id or path.stem,'title':title or path.name,'text':raw.decode('utf-8-sig'),'source_url':source_url}]
    elif fmt=='pdf':
        import pymupdf
        with pymupdf.open(stream=raw,filetype='pdf') as pdf: pages=[page.get_text() for page in pdf]
        text='';ranges=[]
        for index,page in enumerate(pages,1):
            if text:text+='\n\n'
            start=len(text);text+=page;ranges.append({'page':index,'start':start,'end':len(text)})
        documents=[{'_id':doc_id or path.stem,'title':title or path.name,'text':text,'source_url':source_url,'page_ranges':ranges}]
    else: raise ValueError('format must be jsonl, text, or pdf')
    return documents,{'source_sha256':hashlib.sha256(raw).hexdigest(),'input_format':fmt}
