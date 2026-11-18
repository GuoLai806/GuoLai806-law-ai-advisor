"""Versioned, evidence-backed dependency graph. Migration is explicit.

Reviewed input to ``build --input graph.json`` (offsets are unit.text [start,end))::

  {"revision":"r1", "extraction_identity":"manual-v1", "coverage_complete":false,
   "entities":[{"entity_id":"a","name":"A","type":"module","aliases":[],
      "mentions":[{"unit_id":"...","char_start":0,"char_end":1}]}],
   "edges":[{"src_id":"a","dst_id":"b","predicate":"depends_on",
      "supports":[{"unit_id":"...","part_no":1,"char_start":0,"char_end":11,
        "content_sha256":"<unit sha256>","quote":"A needs B.",
        "verification_state":"approved","temporal_status":"unbounded",
        "valid_from":null,"valid_to":null}]}]}

Use compile-candidates on a local OpenViking v0.5.3 export to produce a
reproducible review template. Source URIs are candidates, never approval.
Dates are ISO dates and intervals are half-open. Coverage is an explicit
reviewer attestation, not inferred from a successful import.
"""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sqlite3
import time
import uuid

from .catalog import Catalog


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _date(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('date must be ISO YYYY-MM-DD')
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('date must be ISO YYYY-MM-DD') from exc
    if parsed.isoformat() != value:
        raise ValueError('date must be ISO YYYY-MM-DD')
    return value


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + ' must be nonempty')
    return value


class GraphStore:
    def __init__(self, catalog):
        self.catalog = catalog
        self.db = catalog.db

    def migrate(self):
        """Only graph administration invokes schema migration."""
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS graph_builds(
          graph_build_id TEXT PRIMARY KEY, revision TEXT NOT NULL, corpus_id TEXT NOT NULL,
          source_manifest_sha256 TEXT NOT NULL, extraction_identity TEXT NOT NULL,
          schema_version INTEGER NOT NULL, state TEXT NOT NULL, job_key TEXT NOT NULL UNIQUE,
          payload_sha256 TEXT NOT NULL, payload TEXT NOT NULL, coverage_complete INTEGER NOT NULL,
          created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS graph_active(revision TEXT PRIMARY KEY, graph_build_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS graph_entities(graph_build_id TEXT NOT NULL, entity_id TEXT NOT NULL,
          name TEXT NOT NULL, entity_type TEXT NOT NULL, PRIMARY KEY(graph_build_id,entity_id));
        CREATE TABLE IF NOT EXISTS graph_aliases(graph_build_id TEXT NOT NULL, entity_id TEXT NOT NULL,
          alias TEXT NOT NULL, PRIMARY KEY(graph_build_id,entity_id,alias));
        CREATE TABLE IF NOT EXISTS graph_mentions(graph_build_id TEXT NOT NULL, entity_id TEXT NOT NULL,
          unit_id TEXT NOT NULL, char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
          content_sha256 TEXT NOT NULL, PRIMARY KEY(graph_build_id,entity_id,unit_id,char_start,char_end));
        CREATE TABLE IF NOT EXISTS graph_edges(graph_build_id TEXT NOT NULL, edge_id TEXT NOT NULL,
          src_id TEXT NOT NULL, dst_id TEXT NOT NULL, predicate TEXT NOT NULL,
          PRIMARY KEY(graph_build_id,edge_id), UNIQUE(graph_build_id,src_id,predicate,dst_id));
        CREATE TABLE IF NOT EXISTS graph_supports(graph_build_id TEXT NOT NULL, edge_id TEXT NOT NULL,
          support_id TEXT NOT NULL, unit_id TEXT NOT NULL, part_no INTEGER, char_start INTEGER NOT NULL,
          char_end INTEGER NOT NULL, content_sha256 TEXT NOT NULL, quote TEXT NOT NULL,
          temporal_status TEXT NOT NULL, valid_from TEXT, valid_to TEXT,
          PRIMARY KEY(graph_build_id,edge_id,support_id));
        CREATE INDEX IF NOT EXISTS graph_edges_frontier ON graph_edges(graph_build_id,src_id,predicate);
        CREATE INDEX IF NOT EXISTS graph_mention_units ON graph_mentions(graph_build_id,unit_id);
        ''')

    def _manifest(self, revision, require_approved=True):
        row = self.catalog.revision(revision)
        if not row:
            raise ValueError('unknown revision')
        manifest = json.loads(row['manifest'])
        if manifest.get('profile') != 'general':
            raise ValueError('graph supports general profile only')
        if require_approved and (manifest.get('verification_state') != 'approved' or row['state'] not in ('active', 'retained')):
            raise ValueError('approved active or retained source required')
        return manifest

    def _available(self, revision, build_id):
        try:
            manifest = self._manifest(revision)
            row = self.db.execute("SELECT source_manifest_sha256 FROM graph_builds WHERE revision=? AND graph_build_id=? AND state IN ('active','retained')", (revision,build_id)).fetchone()
            return bool(row and row[0] == _sha(_json(manifest)))
        except (ValueError, sqlite3.OperationalError):
            return False

    def _span(self, revision, item, *, mention_names=None):
        uid = _text(item.get('unit_id'), 'unit_id')
        unit = self.catalog.get(revision, uid)
        if not unit:
            raise ValueError('support unit missing in revision')
        start, end = item.get('char_start'), item.get('char_end')
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(unit['text']):
            raise ValueError('invalid source span')
        quote = unit['text'][start:end]
        if mention_names is not None:
            if quote.casefold() not in {x.casefold() for x in mention_names}:
                raise ValueError('entity mention does not match name or alias')
        else:
            if item.get('content_sha256') != unit['sha256'] or item.get('quote') != quote:
                raise ValueError('support hash or quote mismatch')
            part_no = item.get('part_no')
            if part_no is not None:
                if type(part_no) is not int:
                    raise ValueError('invalid part_no')
                part = next((p for p in unit['parts'] if p['part_no'] == part_no), None)
                if not part or not (part['locator']['char_start'] <= start < end <= part['locator']['char_end']):
                    raise ValueError('support outside part')
        return unit, quote

    def _prepare(self, revision, payload):
        manifest = self._manifest(revision)
        if not isinstance(payload, dict) or payload.get('revision') != revision:
            raise ValueError('payload revision mismatch')
        _text(payload.get('extraction_identity'), 'extraction_identity')
        if type(payload.get('coverage_complete', False)) is not bool:
            raise ValueError('coverage_complete must be boolean')
        entities, edges = payload.get('entities'), payload.get('edges')
        if not isinstance(entities,list) or not isinstance(edges,list):
            raise ValueError('entities and edges must be lists')
        names, mentions, aliases, edge_rows, supports = {}, [], [], {}, []
        pending = False
        for ent in entities:
            eid = _text(ent.get('entity_id'), 'entity_id')
            name = _text(ent.get('name'), 'name')
            if eid in names:
                raise ValueError('duplicate entity ID')
            typ = _text(ent.get('type','unknown'), 'type')
            al = ent.get('aliases',[])
            if not isinstance(al,list): raise ValueError('aliases must be list')
            al = [_text(a,'alias') for a in al]
            names[eid] = (name,typ)
            aliases.extend((eid,a) for a in set(al))
            ms = ent.get('mentions',[])
            if not isinstance(ms,list): raise ValueError('mentions must be list')
            for m in ms:
                unit,_ = self._span(revision,m,mention_names=[name,*al])
                mentions.append((eid,m['unit_id'],m['char_start'],m['char_end'],unit['sha256']))
        for edge in edges:
            src, dst, pred = edge.get('src_id'), edge.get('dst_id'), edge.get('predicate')
            if src not in names or dst not in names or pred != 'depends_on':
                raise ValueError('invalid edge endpoints or predicate')
            edge_id = _sha(_json([src,pred,dst]))
            edge_rows[edge_id] = (src,dst,pred)
            ss = edge.get('supports',[])
            if not isinstance(ss,list): raise ValueError('supports must be list')
            if not ss: pending = True
            for support in ss:
                if support.get('verification_state') != 'approved':
                    pending = True
                    continue
                unit, quote = self._span(revision,support)
                status = support.get('temporal_status')
                start, end = _date(support.get('valid_from')), _date(support.get('valid_to'))
                if status == 'unknown':
                    pending = True
                    continue
                if status not in ('unbounded','interval') or (status == 'unbounded' and (start or end)) or (status == 'interval' and not (start or end)) or (start and end and start >= end):
                    raise ValueError('invalid temporal interval')
                # Do not silently extend evidence outside source validity.
                source_start, source_end = _date(manifest.get('effective_from')), _date(manifest.get('effective_to'))
                if source_start and (not start or start < source_start): start = source_start
                if source_end and (not end or end > source_end): end = source_end
                if start and end and start >= end: raise ValueError('support outside source interval')
                data = (support['unit_id'],support.get('part_no'),support['char_start'],support['char_end'],unit['sha256'],quote,status,start,end)
                supports.append((edge_id,_sha(_json(data)),*data))
        if pending and payload.get('coverage_complete',False):
            raise ValueError('incomplete candidates cannot attest complete coverage')
        return manifest, names, aliases, mentions, edge_rows, supports

    def stage(self, revision, payload, job_key):
        _text(job_key, 'job_key')
        prepared = self._prepare(revision,payload)
        manifest,names,aliases,mentions,edges,supports = prepared
        encoded = _json(payload); ph = _sha(encoded)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old = self.db.execute('SELECT * FROM graph_builds WHERE job_key=?',(job_key,)).fetchone()
            if old:
                if old['revision'] != revision or old['payload_sha256'] != ph or old['source_manifest_sha256'] != _sha(_json(manifest)):
                    raise ValueError('job key reused for different input')
                self.db.execute('COMMIT'); return dict(old)
            bid = 'graph_' + uuid.uuid4().hex
            self.db.execute('INSERT INTO graph_builds VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (bid,revision,manifest['corpus_id'],_sha(_json(manifest)),payload['extraction_identity'],1,'staged',job_key,ph,encoded,int(payload.get('coverage_complete',False)),time.time()))
            self.db.executemany('INSERT INTO graph_entities VALUES(?,?,?,?)',[(bid,e,*v) for e,v in names.items()])
            self.db.executemany('INSERT OR IGNORE INTO graph_aliases VALUES(?,?,?)',[(bid,*x) for x in aliases])
            self.db.executemany('INSERT OR IGNORE INTO graph_mentions VALUES(?,?,?,?,?,?)',[(bid,*x) for x in mentions])
            self.db.executemany('INSERT INTO graph_edges VALUES(?,?,?,?,?)',[(bid,e,*v) for e,v in edges.items()])
            self.db.executemany('INSERT OR IGNORE INTO graph_supports VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',[(bid,*x) for x in supports])
            self.db.execute('COMMIT')
            return dict(self.db.execute('SELECT * FROM graph_builds WHERE graph_build_id=?',(bid,)).fetchone())
        except Exception:
            self.db.execute('ROLLBACK'); raise

    def validate(self, revision, build_id):
        row = self.db.execute('SELECT * FROM graph_builds WHERE revision=? AND graph_build_id=?',(revision,build_id)).fetchone()
        if not row: raise ValueError('unknown build for revision')
        manifest = self._manifest(revision)
        if _sha(_json(manifest)) != row['source_manifest_sha256'] or _sha(row['payload']) != row['payload_sha256']:
            raise ValueError('source or build changed')
        _, names, aliases, mentions, edges, supports = self._prepare(revision,json.loads(row['payload']))
        expected = {
            'graph_entities': {(eid,*value) for eid,value in names.items()},
            'graph_aliases': set(aliases),
            'graph_mentions': set(mentions),
            'graph_edges': {(eid,*value) for eid,value in edges.items()},
            'graph_supports': set(supports),
        }
        columns = {
            'graph_entities':'entity_id,name,entity_type',
            'graph_aliases':'entity_id,alias',
            'graph_mentions':'entity_id,unit_id,char_start,char_end,content_sha256',
            'graph_edges':'edge_id,src_id,dst_id,predicate',
            'graph_supports':'edge_id,support_id,unit_id,part_no,char_start,char_end,content_sha256,quote,temporal_status,valid_from,valid_to',
        }
        for table, wanted in expected.items():
            actual = {tuple(r) for r in self.db.execute('SELECT '+columns[table]+' FROM '+table+' WHERE graph_build_id=?',(build_id,))}
            if actual != wanted: raise ValueError('graph derived data changed: '+table)
        return {'graph_build_id':build_id,'revision':revision,'valid':True,'coverage_complete':bool(row['coverage_complete'])}

    def activate(self, revision, build_id, expected_build_id):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            current = self.db.execute('SELECT graph_build_id FROM graph_active WHERE revision=?',(revision,)).fetchone()
            if (current[0] if current else None) != expected_build_id: raise ValueError('active graph changed')
            self.validate(revision,build_id)
            self.db.execute("UPDATE graph_builds SET state='retained' WHERE revision=? AND state='active'",(revision,))
            self.db.execute("UPDATE graph_builds SET state='active' WHERE revision=? AND graph_build_id=?",(revision,build_id))
            self.db.execute('INSERT INTO graph_active VALUES(?,?) ON CONFLICT(revision) DO UPDATE SET graph_build_id=excluded.graph_build_id',(revision,build_id))
            self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK'); raise
        return self.active_build(revision)

    def active_build(self, revision):
        try:
            self._manifest(revision)
            row = self.db.execute("SELECT b.* FROM graph_active a JOIN graph_builds b ON b.graph_build_id=a.graph_build_id AND b.revision=a.revision WHERE a.revision=? AND b.state='active'",(revision,)).fetchone()
            if not row: return None
            if _sha(_json(self._manifest(revision))) != row['source_manifest_sha256']: return None
            result = dict(row); result['coverage_complete'] = bool(result['coverage_complete'])
            result.pop('payload',None)
            return result
        except (ValueError, sqlite3.OperationalError): return None

    def resolve_seeds(self, revision, build_id, seed, candidate_unit_ids=(), *, query=None):
        if not self._available(revision,build_id): raise ValueError('graph build unavailable')
        if seed:
            exact = self.db.execute(
                'SELECT entity_id,name FROM graph_entities WHERE graph_build_id=? AND entity_id=?',
                (build_id,seed.strip())).fetchone()
            if exact is not None:
                return [dict(exact)]
        rows = self.db.execute('SELECT entity_id,name FROM graph_entities WHERE graph_build_id=? ORDER BY entity_id',(build_id,)).fetchall()
        if seed:
            needle = seed.casefold().strip()
            aliases = {}
            for row in self.db.execute('SELECT entity_id,alias FROM graph_aliases WHERE graph_build_id=?',(build_id,)):
                aliases.setdefault(row['entity_id'],set()).add(row['alias'].casefold())
            return [dict(r) for r in rows if needle in (r['entity_id'].casefold(),r['name'].casefold()) or needle in aliases.get(r['entity_id'],set())]
        candidates = sorted(set(candidate_unit_ids))
        if not candidates:
            return []
        matches = set()
        placeholders = ','.join('?' for _ in candidates)
        for row in self.db.execute(f'SELECT * FROM graph_mentions WHERE graph_build_id=? AND unit_id IN ({placeholders})',
                                   (build_id,*candidates)):
            try:
                unit = self.catalog.get(revision,row['unit_id'])
                if unit and unit['sha256']==row['content_sha256'] and 0 <= row['char_start'] < row['char_end'] <= len(unit['text']): matches.add(row['entity_id'])
            except ValueError: continue
        linked = [dict(r) for r in rows if r['entity_id'] in matches]
        if query:
            folded = query.casefold()
            mentioned = [r for r in linked if len(r['name']) >= 3 and
                         re.search(r'(?<![\w@.-])'+re.escape(r['name'].casefold())+r'(?![\w@.-])', folded)]
            if mentioned:
                return mentioned
        return linked

    def outgoing(self, revision, build_id, src_ids, as_of):
        if not self._available(revision,build_id): raise ValueError('graph build unavailable')
        day = as_of.isoformat() if isinstance(as_of,dt.date) else _date(as_of)
        if day is None: raise ValueError('as_of is required')
        ids = list(dict.fromkeys(src_ids))
        def generate():
            for offset in range(0,len(ids),400):
                chunk = ids[offset:offset+400]
                sql = '''SELECT e.edge_id,e.src_id,a.name AS src_name,e.dst_id,b.name AS dst_name,e.predicate,
                         s.unit_id,s.part_no,s.char_start,s.char_end,s.content_sha256,s.valid_from,s.valid_to,s.quote
                         FROM graph_edges e JOIN graph_supports s USING(graph_build_id,edge_id)
                         JOIN graph_entities a ON a.graph_build_id=e.graph_build_id AND a.entity_id=e.src_id
                         JOIN graph_entities b ON b.graph_build_id=e.graph_build_id AND b.entity_id=e.dst_id
                         WHERE e.graph_build_id=? AND e.src_id IN (%s) AND e.predicate='depends_on'
                         AND (s.valid_from IS NULL OR s.valid_from<=?) AND (s.valid_to IS NULL OR ?<s.valid_to)
                         ORDER BY e.edge_id,s.support_id''' % ','.join('?' for _ in chunk)
                for row in self.db.execute(sql,(build_id,*chunk,day,day)): yield dict(row)
        return generate()


def compile_candidates(directory, revision, extraction_identity='openviking-v0.5.3-compile'):
    """Convert an exported KG tree to an explicitly unreviewed import template.

    PyYAML is needed only for this adapter, to parse YAML safely and correctly.
    The raw evidence references are preserved for review; no URI is approval.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('compile-candidates requires PyYAML to parse entity frontmatter') from exc
    root = pathlib.Path(directory)
    entities = []
    for path in sorted((root/'entities').glob('*.md')):
        content = path.read_text(encoding='utf-8')
        pieces = content.split('---',2)
        if len(pieces) != 3 or pieces[0].strip(): raise ValueError('missing frontmatter: ' + str(path))
        data = yaml.safe_load(pieces[1])
        if not isinstance(data,dict) or data.get('type') != 'entity' or data.get('id') != path.stem:
            raise ValueError('invalid entity frontmatter: ' + str(path))
        entities.append({'entity_id':_text(data.get('id'),'id'), 'name':_text(data.get('title'),'title'),
                         'type':data.get('entity_type') or 'unknown', 'aliases':data.get('aliases') or [],
                         'mentions':[], 'candidate_sources':data.get('sources') or []})
    ids = {e['entity_id'] for e in entities}
    if len(ids) != len(entities): raise ValueError('duplicate entity ID')
    edges, skipped = [], []
    for line_no,line in enumerate((root/'relations.jsonl').read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        relation = json.loads(line)
        if relation.get('from') not in ids or relation.get('to') not in ids:
            raise ValueError('unknown endpoint on line ' + str(line_no))
        if relation.get('relation') != 'depends_on':
            skipped.append({'line':line_no,'relation':relation})
            continue
        evidence = relation.get('evidence')
        if not isinstance(evidence,list) or not evidence: raise ValueError('missing evidence on line ' + str(line_no))
        edges.append({'src_id':relation['from'], 'dst_id':relation['to'], 'predicate':'depends_on',
                      'supports':[{'verification_state':'pending','source_reference':ref} for ref in evidence]})
    return {'revision':revision,'extraction_identity':extraction_identity,'coverage_complete':False,
            'entities':entities,'edges':edges,'skipped_relations':skipped}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command',required=True)
    c = sub.add_parser('compile-candidates',help='convert local OpenViking export to unreviewed JSON')
    c.add_argument('--directory',required=True); c.add_argument('--revision',required=True); c.add_argument('--output',required=True)
    b = sub.add_parser('build',help='explicitly migrate and stage reviewed JSON')
    b.add_argument('--db',required=True); b.add_argument('--input',required=True); b.add_argument('--job-key',required=True)
    v = sub.add_parser('validate',help='revalidate staged or retained build')
    v.add_argument('--db',required=True); v.add_argument('--revision',required=True); v.add_argument('--build-id',required=True)
    a = sub.add_parser('activate',help='atomically switch build with expected current pointer')
    a.add_argument('--db',required=True); a.add_argument('--revision',required=True); a.add_argument('--build-id',required=True)
    a.add_argument('--expected-build-id',help='omit only when no current build')
    args=parser.parse_args(argv)
    if args.command == 'compile-candidates':
        result=compile_candidates(args.directory,args.revision)
        pathlib.Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(_json({'output':args.output,'entities':len(result['entities']),'candidate_edges':len(result['edges'])}))
        return
    catalog=Catalog(args.db); store=GraphStore(catalog); store.migrate()
    try:
        if args.command=='build':
            payload=json.loads(pathlib.Path(args.input).read_text(encoding='utf-8'))
            result=store.stage(payload['revision'],payload,args.job_key)
            result.pop('payload',None)
        elif args.command=='validate': result=store.validate(args.revision,args.build_id)
        else: result=store.activate(args.revision,args.build_id,args.expected_build_id)
        print(_json(result))
    finally: catalog.db.close()


if __name__=='__main__':
    main()
