"""Adapter to the product's generic catalogue; document IDs used only for evaluation."""
import hashlib,json,time
from pathlib import Path
from evidence_rag.catalog import Catalog

REVISION='beir_validation_v1'

class ProductIndex:
 def __init__(self,corpus,manifest):
  start=time.perf_counter();self.cat=Catalog(':memory:')
  clean=[{'_id':key,'title':doc.get('title') or '', 'text':doc['text']} for key,doc in sorted(corpus.items())]
  source={'corpus_id':'beir_'+manifest['dataset'],'version_id':manifest['archive_sha256'], 'source_kind':'benchmark',
          'verification_state':'pending','source_url':manifest['url'],'source_sha256':manifest['corpus_sha256']}
  self.cat.stage_documents(clean,source,revision=REVISION)
  records=self.cat.db.execute('SELECT doc_id,unit_id FROM documents WHERE revision=?',(REVISION,)).fetchall()
  self.doc_to_unit={r['doc_id']:r['unit_id'] for r in records};self.unit_to_doc={v:k for k,v in self.doc_to_unit.items()}
  if set(self.doc_to_unit)!=set(corpus) or len(self.unit_to_doc)!=len(corpus):raise ValueError('Product corpus mapping not one-to-one')
  self.stage_ms=round((time.perf_counter()-start)*1000,2)
 def lexical(self,query):return self.cat.lexical_scored(REVISION,query,limit=40)
 def exact(self,query):return self.cat.exact_documents(REVISION,query)
 def unit(self,doc_id):
  unit=self.cat.get(REVISION,self.doc_to_unit[doc_id])
  if not unit:raise ValueError('Product unit missing')
  return unit
 def close(self):self.cat.db.close()
