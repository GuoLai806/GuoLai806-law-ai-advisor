import hashlib, json, re, sqlite3, time, uuid, unicodedata
from pathlib import Path

PARSER_VERSION = 'article-v4-section-crosscheck'
SECTION = re.compile(r'(?m)^[ \t\u3000]*(?:第[一二三四五六七八九十百千万零〇两\d]+(?:编|分编|章|节)[ \t\u3000]+[^\n]*|附[ \t\u3000]*则[ \t\u3000]*)$')
PARAGRAPH = re.compile(r'(?m)^[\u3000]{2}(?!第[一二三四五六七八九十百千万零〇两\d]+(?:条|编|分编|章|节))')
ITEM = re.compile(r'^[ \t\u3000]*（[一二三四五六七八九十百\d]+）')
ARTICLE = re.compile(r'(?m)^[ \t\u3000]*(第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?)[ \t\u3000]+')

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def _number(value):
    value=unicodedata.normalize('NFKC',value)
    if value.isdigit(): return int(value)
    digits={'零':0,'〇':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    units={'十':10,'百':100,'千':1000,'万':10000}; total=0; section=0; current=0
    for char in value:
        if char in digits: current=digits[char]
        elif char in units:
            n=units[char]
            if n==10000: total+=(section+current)*n; section=current=0
            else: section+=(current or 1)*n; current=0
        else: raise ValueError('invalid article number')
    return total+section+current

def canonical_label(label):
    match=re.fullmatch(r'第([一二三四五六七八九十百千万零〇两\d]+)条(?:之([一二三四五六七八九十\d]+))?',unicodedata.normalize('NFKC',label))
    if not match: raise ValueError('invalid article label')
    return (_number(match.group(1)),_number(match.group(2)) if match.group(2) else 0)

def tokens(text):
    import jieba
    return ' '.join(t for t in jieba.cut_for_search(text, HMM=False) if re.search(r'[\w\u4e00-\u9fff]', t))

def parse_pages(pages):
    """Parse document globally; locator offsets refer to concatenated extracted text."""
    starts, full = [], ''
    for page in pages:
        starts.append(len(full)); full += page + '\n'
    hits = list(ARTICLE.finditer(full))
    # Table of contents occurs before the body. Remove duplicated headings preceding the last
    # monotonic article sequence by using the last consecutive restart at article one.
    ones = [i for i, h in enumerate(hits) if h.group(1) in ('第一条', '第1条')]
    if len(ones) > 1: hits = hits[ones[-1]:]
    result = []
    for i, hit in enumerate(hits):
        end = hits[i+1].start() if i+1 < len(hits) else len(full)
        section = SECTION.search(full, hit.end(), end)
        if section: end = section.start()
        text = full[hit.start():end].strip()
        if not text: continue
        included = [n+1 for n, start in enumerate(starts) if start < end and (starts[n+1] if n+1 < len(starts) else len(full)) > hit.start()]
        paragraph_starts = [hit.start()] + [p.start() for p in PARAGRAPH.finditer(full, hit.end(), end)]
        paragraphs=[]
        for j, start in enumerate(paragraph_starts):
            stop=paragraph_starts[j+1] if j+1<len(paragraph_starts) else end
            content=full[start:stop].strip()
            if not content: continue
            pg=[n+1 for n, pos in enumerate(starts) if pos < stop and (starts[n+1] if n+1<len(starts) else len(full)) > start]
            paragraphs.append({'kind':'item' if ITEM.match(content) else 'paragraph','text':content,'locator':{'pages':pg,'char_start':start,'char_end':stop}})
        result.append({'label': hit.group(1), 'text': text, 'locator': {'pages': included, 'char_start': hit.start(), 'char_end': end}, 'paragraphs':paragraphs})
    return result

class Catalog:
    def __init__(self, path):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, timeout=10, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS revisions(revision TEXT PRIMARY KEY, manifest TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS units(revision TEXT NOT NULL, unit_id TEXT NOT NULL, label TEXT NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL, locator TEXT NOT NULL, sha256 TEXT NOT NULL, uri TEXT, PRIMARY KEY(revision,unit_id));
        CREATE TABLE IF NOT EXISTS parts(revision TEXT NOT NULL, unit_id TEXT NOT NULL, part_no INTEGER NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, locator TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(revision,unit_id,part_no), FOREIGN KEY(revision,unit_id) REFERENCES units(revision,unit_id));
        CREATE TABLE IF NOT EXISTS relations(revision TEXT NOT NULL, from_id TEXT NOT NULL, to_id TEXT NOT NULL, relation TEXT NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, verified INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(revision,from_id,to_id,char_start));
        CREATE VIRTUAL TABLE IF NOT EXISTS unit_fts USING fts5(revision UNINDEXED, unit_id UNINDEXED, text, tokenize='unicode61');
        CREATE VIRTUAL TABLE IF NOT EXISTS part_fts USING fts5(revision UNINDEXED, unit_id UNINDEXED, part_no UNINDEXED, text, tokenize='unicode61');
        CREATE TABLE IF NOT EXISTS jobs(job_id TEXT PRIMARY KEY, idem TEXT UNIQUE NOT NULL, revision TEXT NOT NULL, state TEXT NOT NULL, error TEXT);
        CREATE TABLE IF NOT EXISTS documents(revision TEXT NOT NULL, doc_id TEXT NOT NULL, unit_id TEXT NOT NULL, title TEXT NOT NULL, source_url TEXT, source_kind TEXT NOT NULL, content_sha256 TEXT NOT NULL, metadata TEXT NOT NULL, PRIMARY KEY(revision,doc_id), UNIQUE(revision,unit_id));
        CREATE TABLE IF NOT EXISTS source_reviews(revision TEXT PRIMARY KEY, reviewer TEXT NOT NULL, evidence TEXT NOT NULL, created REAL NOT NULL);
        ''')
    def active(self):
        r = self.db.execute("SELECT value FROM settings WHERE key='active'").fetchone(); return r[0] if r else None
    def revision(self, rev):
        r = self.db.execute('SELECT * FROM revisions WHERE revision=?', (rev,)).fetchone(); return dict(r) if r else None
    def stage(self, pages, manifest, revision=None, idem=None):
        if any(not isinstance(manifest.get(k),str) or not manifest[k].strip() for k in ('doc_id','version_id','title','source_kind','verification_state')): raise ValueError('manifest identity fields required')
        if manifest.get('source_kind')=='synthetic' and manifest.get('source_sha256'): raise ValueError('synthetic fixture must not declare a source file hash')
        manifest = {**manifest, 'extracted_text_sha256': digest('\n'.join(pages))}
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        parsed = parse_pages(pages)
        if not parsed: raise ValueError('no article headings found')
        rev = revision or 'law_'+uuid.uuid4().hex
        idem = idem or digest(encoded + digest('\n'.join(pages)) + PARSER_VERSION)
        previous = self.db.execute('SELECT * FROM jobs WHERE idem=?', (idem,)).fetchone()
        if previous:
            existing=self.revision(previous['revision'])
            if not existing: raise ValueError('idempotency key has no revision')
            old=json.loads(existing['manifest']); new=json.loads(encoded)
            old.pop('verification_state',None); new.pop('verification_state',None)
            if old != new: raise ValueError('idempotency key reused for different input')
            return dict(previous)
        labels = [x['label'] for x in parsed]
        if len(set(labels)) != len(labels): raise ValueError('duplicate article labels')
        jid = 'job_'+uuid.uuid4().hex
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,NULL)',(jid,idem,rev,'queued'))
            self.db.execute('INSERT INTO revisions VALUES(?,?,?,?)',(rev,encoded,'staged',time.time()))
            for a in parsed:
                uid = digest(manifest['doc_id']+'/'+manifest['version_id']+'/'+a['label'])[:32]
                self.db.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?,?)',(rev,uid,a['label'],manifest['title'],a['text'],json.dumps(a['locator']),digest(a['text']),None))
                for n, part in enumerate(a['paragraphs'],1):
                    self.db.execute('INSERT INTO parts VALUES(?,?,?,?,?,?,?)',(rev,uid,n,part['kind'],part['text'],json.dumps(part['locator']),digest(part['text'])))
                self.db.execute('INSERT INTO unit_fts VALUES(?,?,?)',(rev,uid,tokens(a['label']+' '+a['text'])))
            rows=self.db.execute('SELECT unit_id,label,text FROM units WHERE revision=?',(rev,)).fetchall()
            lookup={canonical_label(row['label']):row['unit_id'] for row in rows}
            for row in rows:
                for match in re.finditer(r'第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?',row['text']):
                    target=lookup.get(canonical_label(match.group()))
                    if target and target!=row['unit_id']:
                        self.db.execute('INSERT OR IGNORE INTO relations VALUES(?,?,?,?,?,?,0)',(rev,row['unit_id'],target,'cites',match.start(),match.end()))
            self.db.execute("UPDATE jobs SET state='indexed' WHERE job_id=?",(jid,))
            self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK'); raise
        return {'job_id':jid,'idem':idem,'revision':rev,'state':'indexed','error':None}
    def stage_documents(self, documents, manifest, revision=None, idem=None):
        """Stage a generic document revision; each full document is a retrievable unit."""
        if not isinstance(documents,list) or not documents: raise ValueError('at least one document required')
        for key in ('corpus_id','version_id','source_kind','verification_state'):
            if not isinstance(manifest.get(key),str) or not manifest[key].strip(): raise ValueError('manifest identity fields required')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}',manifest['corpus_id']): raise ValueError('invalid corpus ID')
        if revision and not re.fullmatch(r'[A-Za-z0-9_-]{1,120}',revision): raise ValueError('invalid revision')
        manifest={**manifest,'profile':'general','parser_version':'generic-documents-v2','embedding_serialization':'title-newline-text-v1'}
        prepared=[]; ids=set()
        for doc in documents:
            if not isinstance(doc,dict): raise ValueError('document must be an object')
            doc_id=doc.get('_id',doc.get('id'));title=doc.get('title','');content=doc.get('text')
            if not isinstance(doc_id,str) or not doc_id.strip() or len(doc_id)>512: raise ValueError('document _id must be a nonempty string')
            if doc_id in ids: raise ValueError('duplicate document _id')
            if not isinstance(title,str) or not isinstance(content,str) or not content.strip(): raise ValueError('document title/text invalid')
            ids.add(doc_id);prepared.append({'_id':doc_id,'title':title,'text':content,'source_url':doc.get('source_url',manifest.get('source_url')),'metadata':doc.get('metadata',{}),'page_ranges':doc.get('page_ranges')})
        corpus_content=json.dumps(prepared,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        manifest['extracted_text_sha256']=digest(corpus_content)
        encoded=json.dumps(manifest,ensure_ascii=False,sort_keys=True)
        rev=revision or 'corpus_'+uuid.uuid4().hex
        idem=idem or digest(encoded+manifest['extracted_text_sha256'])
        old=self.db.execute('SELECT * FROM jobs WHERE idem=?',(idem,)).fetchone()
        if old:
            old_rev=self.revision(old['revision'])
            if not old_rev or old_rev['manifest']!=encoded: raise ValueError('idempotency key reused for different input')
            return dict(old)
        jid='job_'+uuid.uuid4().hex
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('INSERT INTO jobs VALUES(?,?,?,?,NULL)',(jid,idem,rev,'queued'))
            self.db.execute('INSERT INTO revisions VALUES(?,?,?,?)',(rev,encoded,'staged',time.time()))
            for doc in prepared:
                uid=digest(manifest['corpus_id']+'/'+manifest['version_id']+'/'+doc['_id'])[:32]
                content=doc['text']; page_ranges=doc.get('page_ranges') or []
                pages=[x['page'] for x in page_ranges] if page_ranges else []
                locator={'kind':'pdf' if pages else 'text','pages':pages,'char_start':0,'char_end':len(content)}
                self.db.execute('INSERT INTO units VALUES(?,?,?,?,?,?,?,?)',(rev,uid,doc['title'],doc['title'],content,json.dumps(locator),digest(content),None))
                self.db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',(rev,doc['_id'],uid,doc['title'],doc.get('source_url'),manifest['source_kind'],digest(content),json.dumps(doc.get('metadata') or {},ensure_ascii=False)))
                boundaries=[0]+[m.end() for m in re.finditer(r'\n[ \t]*\n+',content)]+[len(content)]
                for n,(start,end) in enumerate(zip(boundaries,boundaries[1:]),1):
                    chunk=content[start:end]
                    if not chunk.strip(): continue
                    part_pages=[x['page'] for x in page_ranges if x['start']<end and x['end']>start]
                    loc={'kind':'pdf' if pages else 'text','pages':part_pages,'char_start':start,'char_end':end}
                    self.db.execute('INSERT INTO parts VALUES(?,?,?,?,?,?,?)',(rev,uid,n,'paragraph',chunk,json.dumps(loc),digest(chunk)))
                    self.db.execute('INSERT INTO part_fts VALUES(?,?,?,?)',(rev,uid,n,tokens(chunk)))
                self.db.execute('INSERT INTO unit_fts VALUES(?,?,?)',(rev,uid,tokens(doc['title']+' '+content)))
            self.db.execute("UPDATE jobs SET state='indexed' WHERE job_id=?",(jid,));self.db.execute('COMMIT')
        except Exception:self.db.execute('ROLLBACK');raise
        return {'job_id':jid,'idem':idem,'revision':rev,'state':'indexed','error':None}

    def best_part(self, rev, uid, query):
        unit=self.get(rev,uid)
        if not unit:return None
        terms=list(dict.fromkeys(tokens(query).split()))[:30]
        if terms:
            expression=' OR '.join('"'+x.replace('"','""')+'"' for x in terms)
            row=self.db.execute('SELECT part_no FROM part_fts WHERE part_fts MATCH ? AND revision=? AND unit_id=? ORDER BY bm25(part_fts), CAST(part_no AS INTEGER) LIMIT 1',(expression,rev,uid)).fetchone()
            if row:
                part=next((p for p in unit['parts'] if p['part_no']==int(row[0])),None)
                if part:return part
        return unit['parts'][0] if unit['parts'] else None

    def exact_documents(self, rev, query, doc_id=None):
        if doc_id is not None:
            rows=self.db.execute('SELECT unit_id FROM documents WHERE revision=? AND doc_id=?',(rev,doc_id))
        else:
            rows=self.db.execute('SELECT unit_id FROM documents WHERE revision=? AND title<>? AND lower(title)=lower(?)',(rev,'',query.strip()))
        return [r[0] for r in rows][:10]

    def approve_source(self, rev, reviewer, evidence):
        if not reviewer.strip() or len(evidence.strip())<20: raise ValueError('reviewer and detailed evidence required')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row=self.revision(rev)
            if not row or row['state']=='active': raise ValueError('staged revision required')
            meta=json.loads(row['manifest'])
            if not meta.get('source_sha256') or not meta.get('source_url'): raise ValueError('source URL and source hash required')
            meta['verification_state']='approved'
            self.db.execute('UPDATE revisions SET manifest=? WHERE revision=?',(json.dumps(meta,ensure_ascii=False,sort_keys=True),rev))
            self.db.execute('INSERT INTO source_reviews VALUES(?,?,?,?)',(rev,reviewer,evidence,time.time()))
            self.db.execute("UPDATE jobs SET state='verified' WHERE revision=?",(rev,))
            self.db.execute('COMMIT')
        except Exception: self.db.execute('ROLLBACK'); raise
    def activate(self, rev, expected):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.active() != expected: raise ValueError('active revision changed')
            row = self.revision(rev)
            if not row or not self.db.execute('SELECT COUNT(*) FROM units WHERE revision=?',(rev,)).fetchone()[0]: raise ValueError('missing or empty revision')
            meta=json.loads(row['manifest'])
            if meta.get('verification_state') not in (('approved','provided') if meta.get('profile')=='general' else ('approved',)): raise ValueError('source verification is not approved')
            if meta.get('profile','legal') == 'legal' and meta.get('source_kind') != 'synthetic' and not self.db.execute('SELECT 1 FROM source_reviews WHERE revision=?',(rev,)).fetchone(): raise ValueError('source review record required')
            if meta.get('profile','legal') == 'legal' and meta.get('source_kind') != 'synthetic' and not self.db.execute('SELECT 1 FROM settings WHERE key=? AND value=?',('dense_validated:'+rev,'yes')).fetchone(): raise ValueError('dense index not validated')
            counts=self.db.execute('SELECT (SELECT COUNT(*) FROM units WHERE revision=?),(SELECT COUNT(*) FROM unit_fts WHERE revision=?)',(rev,rev)).fetchone()
            if counts[0] != counts[1]: raise ValueError('full text index count mismatch')
            unit_ids={r[0] for r in self.db.execute('SELECT unit_id FROM units WHERE revision=?',(rev,))}
            fts_ids={r[0] for r in self.db.execute('SELECT unit_id FROM unit_fts WHERE revision=?',(rev,))}
            if unit_ids != fts_ids or len(fts_ids)!=counts[1]: raise ValueError('full text index mapping mismatch')
            indexed={r['unit_id']:r['text'] for r in self.db.execute('SELECT unit_id,text FROM unit_fts WHERE revision=?',(rev,))}
            for row in self.db.execute('SELECT unit_id,label,text FROM units WHERE revision=?',(rev,)):
                if indexed[row['unit_id']] != tokens(row['label']+' '+row['text']): raise ValueError('full text index content mismatch')
            if meta.get('profile')=='general':
                part_index={(r['unit_id'],int(r['part_no'])):r['text'] for r in self.db.execute('SELECT unit_id,part_no,text FROM part_fts WHERE revision=?',(rev,))}
                parts=list(self.db.execute('SELECT unit_id,part_no,text,sha256 FROM parts WHERE revision=?',(rev,)))
                if len(part_index)!=len(parts) or any(part_index.get((r['unit_id'],r['part_no']))!=tokens(r['text']) or digest(r['text'])!=r['sha256'] for r in parts): raise ValueError('paragraph index mismatch')
            for unit in self.db.execute('SELECT text,sha256 FROM units WHERE revision=?',(rev,)):
                if digest(unit['text'])!=unit['sha256']: raise ValueError('unit hash mismatch')
            self.db.execute("INSERT INTO settings VALUES('active',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(rev,))
            self.db.execute("UPDATE revisions SET state='retained' WHERE state='active' AND revision<>?",(rev,))
            self.db.execute("UPDATE revisions SET state='active' WHERE revision=?",(rev,))
            self.db.execute("UPDATE jobs SET state='active' WHERE revision=?",(rev,)); self.db.execute('COMMIT')
        except Exception: self.db.execute('ROLLBACK'); raise
    def get(self, rev, uid):
        r = self.db.execute('SELECT * FROM units WHERE revision=? AND unit_id=?',(rev,uid)).fetchone()
        if not r: return None
        v = dict(r)
        if digest(v['text'])!=v['sha256']:raise ValueError('catalog unit hash mismatch')
        v['locator']=json.loads(v['locator'])
        doc=self.db.execute('SELECT doc_id,source_url,source_kind,metadata,content_sha256 FROM documents WHERE revision=? AND unit_id=?',(rev,uid)).fetchone()
        v['document']=dict(doc) if doc else None
        if doc and doc['content_sha256']!=v['sha256']:raise ValueError('catalog document hash mismatch')
        v['parts']=[{**dict(x),'locator':json.loads(x['locator'])} for x in self.db.execute('SELECT part_no,kind,text,locator,sha256 FROM parts WHERE revision=? AND unit_id=? ORDER BY part_no',(rev,uid))]
        for part in v['parts']:
            if digest(part['text'])!=part['sha256']:raise ValueError('catalog paragraph hash mismatch')
            if doc:
                loc=part['locator'];start=loc.get('char_start');end=loc.get('char_end')
                if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<=end<=len(v['text']) or v['text'][start:end]!=part['text']:raise ValueError('catalog paragraph locator mismatch')
        v['relations']=[dict(x) for x in self.db.execute('SELECT to_id,relation,char_start,char_end,verified FROM relations WHERE revision=? AND from_id=?',(rev,uid))]
        return v
    def lexical_scored(self, rev, query, limit=40):
        terms = list(dict.fromkeys(tokens(query).split()))[:30]
        if not terms: return []
        expression = ' OR '.join('"'+x.replace('"','""')+'"' for x in terms)
        return [(r[0],r[1]) for r in self.db.execute('SELECT unit_id,bm25(unit_fts) AS score FROM unit_fts WHERE unit_fts MATCH ? AND revision=? ORDER BY score LIMIT ?', (expression,rev,limit))]
    def lexical(self, rev, query, limit=40):
        return [uid for uid,score in self.lexical_scored(rev,query,limit)]
    def exact(self, rev, query):
        query=unicodedata.normalize('NFKC',query)
        labels=re.findall(r'第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?',query)
        wanted={canonical_label(label) for label in labels}
        if not wanted: return []
        rows=self.db.execute('SELECT unit_id,label FROM units WHERE revision=?',(rev,))
        return [r['unit_id'] for r in rows if canonical_label(r['label']) in wanted][:10]
