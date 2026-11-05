import hashlib,json,math,zipfile
from pathlib import Path
import numpy as np
import pytest
from benchmarks.core import BM25, fuse, input_text, load_dataset
from benchmarks.metrics import evaluate
from benchmarks.prepare import extract_safe, filehash, inspect
from benchmarks.run import dense_cache


def test_linear_ndcg_recall_mrr_and_no_duplicate_credit():
    r=evaluate('q',{'a':2,'b':1,'c':0},['x','b','b','a'])
    expected=(1/math.log2(3)+2/math.log2(4))/(2+1/math.log2(3))
    assert r['ndcg_at_10']==pytest.approx(expected)
    assert r['recall_at_20']==1 and r['mrr_at_10']==0.5
    assert evaluate('q',{'a':1},[])['recall_at_20']==0


def test_identical_query_id_is_ignored_as_in_beir():
    r=evaluate('q',{'q':1,'a':1},['q','a'])
    assert r['recall_at_20']==0.5 and r['mrr_at_10']==1


def test_bm25_uses_corpus_only_and_fusion_records_raw_scores():
    corpus={'one':{'title':'Blue sky','text':'rain clouds'},'two':{'title':'Animal','text':'yellow dog'}}
    index=BM25(corpus)
    try:
        hits=index.search('blue AND "rain"'); assert hits and hits[0][0]=='one'
        assert input_text(corpus['one'])=='Blue sky\nrain clouds'
        ids,scores,sources=fuse({'bm25':hits,'dense':[('two',0.9),('one',0.5)]})
        assert set(ids)=={'one','two'} and sources['one'][0]['source']=='bm25' and scores['one']>scores['two']
    finally:index.close()


def test_malformed_or_traversal_zip_rejected(tmp_path):
    z=tmp_path/'bad.zip'
    with zipfile.ZipFile(z,'w') as f:f.writestr('scifact/../../escape','bad')
    with pytest.raises(ValueError):extract_safe(z,tmp_path/'out','scifact')
    assert not (tmp_path/'escape').exists()


def test_safe_zip_rejects_symlink(tmp_path):
    z=tmp_path/'bad.zip';info=zipfile.ZipInfo('scifact/link');info.create_system=3;info.external_attr=(0o120777<<16)
    with zipfile.ZipFile(z,'w') as f:f.writestr(info,'../target')
    with pytest.raises(ValueError):extract_safe(z,tmp_path/'out','scifact')


def test_fusion_rejects_nonfinite_scores():
    with pytest.raises(ValueError):fuse({'dense':[('a',float('nan'))]})


def test_cache_is_hash_checked_and_no_pickle(tmp_path):
    class Fake:
        calls=0
        def encode(self,texts,**kwargs):self.calls+=1;return np.array([[1.,0.] if i%2==0 else [0.,1.] for i in range(len(texts))])
    m={'corpus_sha256':'a','input_format':'title newline text','model':'fixture','revision':'r','dtype':'float16','normalized':True,'dimension':2,'query_hash':'b'};model=Fake()
    _,_,first=dense_cache(tmp_path,m,model,['doc1','doc2'],['query'],['d1','d2'],['q1'],2)
    _,_,second=dense_cache(tmp_path,m,model,['doc1','doc2'],['query'],['d1','d2'],['q1'],2)
    assert not first['hit'] and second['hit'] and model.calls==2
    cache=tmp_path/first['cache_key']/'corpus.npy';cache.write_bytes(cache.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='hash mismatch'):dense_cache(tmp_path,m,model,['doc1','doc2'],['query'],['d1','d2'],['q1'],2)


def test_model_cache_key_changes_with_query_and_dtype(tmp_path):
    class Fake:
        def encode(self,texts,**kwargs):return np.tile(np.array([[1.,0.]]),(len(texts),1))
    meta={'dimension':2,'query_hash':'A','dtype':'float16'}
    _,_,a=dense_cache(tmp_path,meta,Fake(),['doc'],['q'],['d'],['qid'],2)
    _,_,b=dense_cache(tmp_path,{**meta,'query_hash':'B'},Fake(),['doc'],['q'],['d'],['qid'],2)
    _,_,c=dense_cache(tmp_path,{**meta,'dtype':'float32'},Fake(),['doc'],['q'],['d'],['qid'],2)
    assert len({a['cache_key'],b['cache_key'],c['cache_key']})==3

def make_tiny_dataset(tmp_path):
    from benchmarks.prepare import filehash
    root=tmp_path/'data'/'fixture';(root/'qrels').mkdir(parents=True)
    docs=[{'_id':'doc-secret-1','title':'Blue','text':'sky'},{'_id':'doc-secret-2','title':'Yellow','text':'dog'}]
    queries=[{'_id':'query-1','text':'blue sky'},{'_id':'query-2','text':'yellow dog'}]
    for name,rows in [('corpus.jsonl',docs),('queries.jsonl',queries)]:
        (root/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (root/'qrels/train.tsv').write_text('query-id\tcorpus-id\tscore\nquery-1\tdoc-secret-1\t1\n')
    (root/'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\nquery-2\tdoc-secret-2\t1\n')
    manifest={'dataset':'fixture','corpus_count':2,'corpus_sha256':filehash(root/'corpus.jsonl'),'queries_sha256':filehash(root/'queries.jsonl'),'qrels':{'train':{'sha256':filehash(root/'qrels/train.tsv')},'test':{'sha256':filehash(root/'qrels/test.tsv')}},'validation':{'source_split':'train','query_ids':['query-1'],'count':1},'official_test_count':1}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest));return root.parent,path


def test_runner_hybrid_fake_models_exercises_full_pipeline(tmp_path,monkeypatch):
    import sys,types
    from benchmarks import run
    root,manifest=make_tiny_dataset(tmp_path);events=[];inputs=[]
    class Param:
        dtype='torch.float16'
        def is_floating_point(self):return True
    class Embedding:
        device='mps';max_seq_length=100
        def __init__(self,*a,**kw):events.append('embedding_load')
        def parameters(self):return [Param()]
        def tokenizer(self,text,**kw):return {'input_ids':[1]*len(text.split())}
        def get_sentence_embedding_dimension(self):return 2
        def encode(self,texts,**kw):inputs.extend(texts);return np.array([[1.,0.] for _ in texts])
        def __del__(self):events.append('embedding_released')
    class Cross:
        max_length=4096;prompts={};default_prompt_name=None
        def __init__(self,*a,**kw):events.append('reranker_load');self.model=types.SimpleNamespace(device='mps')
        def parameters(self):return [Param()]
        def preprocess(self,pairs,**kw):return {'attention_mask':np.array([[1,1]])}
        def predict(self,pairs,**kw):return np.array([.8 for _ in pairs])
    fake_torch=types.SimpleNamespace(float16='float16',backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda:True)),mps=types.SimpleNamespace(empty_cache=lambda:events.append('empty_cache')))
    monkeypatch.setitem(sys.modules,'torch',fake_torch);monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(SentenceTransformer=Embedding,CrossEncoder=Cross))
    monkeypatch.setattr(run.importlib.metadata,'version',lambda _: 'fixture')
    out=tmp_path/'hybrid.json';monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(root),'--manifest',str(manifest),'--mode','hybrid','--embedding','bge','--reranker','bge','--cache-dir',str(tmp_path/'cache'),'--output',str(out)])
    assert run.main()==0
    report=json.loads(out.read_text());assert report['query_count_recorded']==1 and report['failures']==0 and report['model_runtime']['models_loaded_sequentially']
    assert events.index('embedding_released')<events.index('reranker_load')
    assert all('doc-secret' not in text and 'query-1' not in text for text in inputs)
    assert report['results'][0]['source_candidates']['dense'] and report['results'][0]['ranking'][0]['sources']


def test_official_test_requires_frozen_selection(tmp_path,monkeypatch):
    import sys
    from benchmarks import run
    root,manifest=make_tiny_dataset(tmp_path)
    monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(root),'--manifest',str(manifest),'--split','test','--mode','bm25','--output',str(tmp_path/'result.json')])
    with pytest.raises(ValueError,match='frozen validation'):run.main()
    assert not (tmp_path/'result.json').exists()

def test_failure_stays_in_denominator_and_report(tmp_path,monkeypatch):
    import sys
    from benchmarks import run
    root,manifest=make_tiny_dataset(tmp_path)
    monkeypatch.setattr(run.BM25,'search',lambda *_args,**_kw: (_ for _ in ()).throw(RuntimeError('synthetic failure')))
    out=tmp_path/'failed.json';monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(root),'--manifest',str(manifest),'--mode','bm25','--output',str(out)])
    assert run.main()==1
    report=json.loads(out.read_text());assert report['failures']==1 and report['query_count_recorded']==1
    assert report['metrics']['ndcg_at_10']==0 and report['eligibility']=='failed_run_not_comparable'
    assert report['results'][0]['failure']['message']=='synthetic failure'


def test_synthetic_test_requires_matching_complete_validation(tmp_path,monkeypatch):
    import sys
    from benchmarks import run,select
    root,manifest=make_tiny_dataset(tmp_path);validation=tmp_path/'validation.json';selection=tmp_path/'selection.json';test=tmp_path/'test.json'
    monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(root),'--manifest',str(manifest),'--mode','bm25','--output',str(validation)])
    assert run.main()==0
    monkeypatch.setattr(sys,'argv',['select','--manifest',str(manifest),'--validation-report',str(validation),'--output',str(selection)])
    select.main()
    monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(root),'--manifest',str(manifest),'--mode','bm25','--split','test','--selection',str(selection),'--validation-report',str(validation),'--output',str(test)])
    assert run.main()==0 and json.loads(test.read_text())['split']=='test'
    # A changed report cannot be substituted after configuration freeze.
    validation.write_text(validation.read_text()+' ')
    monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(root),'--manifest',str(manifest),'--mode','bm25','--split','test','--selection',str(selection),'--validation-report',str(validation),'--output',str(tmp_path/'invalid.json')])
    with pytest.raises(ValueError,match='does not match'):run.main()


def test_product_catalog_and_shared_ranker_paths():
    from benchmarks.product import ProductIndex,REVISION
    from evidence_rag.retrieval import rank_candidates,document_embedding_text,reranker_document_text
    corpus={'doc-secret-1':{'title':'Blue sky','text':'The cloud is blue.'},'doc-secret-2':{'title':'Bright sun','text':'Solar rays and light.'}}
    m={'dataset':'fixture','archive_sha256':'a'*64,'url':'https://example.invalid','corpus_sha256':'b'*64}
    index=ProductIndex(corpus,m)
    try:
        unit=index.unit('doc-secret-1');assert unit['document']['doc_id']=='doc-secret-1'
        assert document_embedding_text(unit)=='Blue sky\nThe cloud is blue.'
        assert reranker_document_text(unit,unit['parts'][0])=='Blue sky\nThe cloud is blue.'
        rank=rank_candidates(index.cat,REVISION,'cloud',index.lexical('cloud'),[])
        assert rank['candidates'][0]['document']['doc_id']=='doc-secret-1'
        assert rank['sources'][unit['unit_id']][0]['raw_score'] is not None
    finally:index.close()


def test_product_empty_title_does_not_index_or_embed_document_id():
    from benchmarks.product import ProductIndex
    from evidence_rag.retrieval import document_embedding_text,reranker_document_text
    index=ProductIndex({'docsecretunique':{'title':'','text':'actual text'}},{'dataset':'fixture','archive_sha256':'a'*64,'url':'https://example.invalid','corpus_sha256':'b'*64})
    try:
        unit=index.unit('docsecretunique')
        assert document_embedding_text(unit)=='\nactual text'
        assert reranker_document_text(unit,unit['parts'][0])=='\nactual text'
        assert index.lexical('docsecretunique')==[]
    finally:index.close()


def test_product_runner_fake_models_shared_core_no_gold_leak(tmp_path,monkeypatch):
    import sys,types
    from benchmarks import run_product
    root,manifest=make_tiny_dataset(tmp_path);m=json.loads(manifest.read_text());m.update(archive_sha256='a'*64,url='https://example.invalid');manifest.write_text(json.dumps(m))
    import jieba
    with jieba.get_dict_file() as f: dictionary_sha256=filehash(f.name)
    freeze={'jieba_dictionary_sha256':dictionary_sha256,'source_sha256':run_product.source_hashes(),'datasets':{'fixture':filehash(manifest)},'candidate_top_per_source':40,'candidate_cap':60,'batch_size':2,'device':'mps','dtype':'float16','embedding_models':{k:list(v) for k,v in run_product.EMBED.items()},'rerank_models':{k:list(v) for k,v in run_product.RERANK.items()},'platform':{'system':run_product.platform.system(),'machine':run_product.platform.machine()},'library_versions':{x:(run_product.importlib.metadata.version('jieba') if x=='jieba' else 'fixture') for x in ('torch','transformers','sentence-transformers','numpy','jieba')}}
    frozen=tmp_path/'freeze.json';frozen.write_text(json.dumps(freeze));events=[];inputs=[]
    class Param:
        dtype='torch.float16'
        def is_floating_point(self):return True
    class Embedding:
        device='mps';max_seq_length=100
        def __init__(self,*a,**kw):events.append('embedding_load')
        def parameters(self):return [Param()]
        def tokenizer(self,text,**kw):return {'input_ids':[1]*len(text.split())}
        def get_sentence_embedding_dimension(self):return 2
        def encode(self,texts,**kw):inputs.extend(texts);return np.array([[1.,0.] for _ in texts])
        def __del__(self):events.append('embedding_released')
    class FakeReranker:
        revision='fixture'
        def __init__(self,*args,**kwargs):events.append('reranker_load');self.model=types.SimpleNamespace(parameters=lambda:[Param()],model=types.SimpleNamespace(device='mps'),max_length=4096)
        def rank(self,query,candidates):return [(u,0.8) for u in candidates]
    torch=types.SimpleNamespace(float16='float16',backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda:True)),mps=types.SimpleNamespace(empty_cache=lambda:events.append('empty_cache')))
    monkeypatch.setitem(sys.modules,'torch',torch);monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(SentenceTransformer=Embedding));monkeypatch.setattr(run_product,'CrossEncoderReranker',FakeReranker)
    original=run_product.importlib.metadata.version;monkeypatch.setattr(run_product.importlib.metadata,'version',lambda x:freeze['library_versions'].get(x) or original(x))
    out=tmp_path/'product.json';monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(root),'--manifest',str(manifest),'--freeze',str(frozen),'--mode','product-hybrid','--embedding','bge','--reranker','bge','--cache-dir',str(tmp_path/'cache'),'--output',str(out)])
    assert run_product.main()==0
    report=json.loads(out.read_text());assert report['failures']==0 and report['query_count_recorded']==1
    journal_events=[json.loads(line) for line in Path(str(out)+'.journal.jsonl').read_text().splitlines()]
    assert [x['type'] for x in journal_events]==['header','runtime','query','complete']
    assert journal_events[0]['run_id']==report['run_id']==journal_events[-1]['run_id']
    assert journal_events[-1]['report_sha256']==filehash(out) and journal_events[2]['row']==report['results'][0]
    assert 'pre-evidence-assembly' in report['ranking_scope'] and 'not OpenViking' in report['dense_backend']
    assert events.index('embedding_released')<events.index('reranker_load')
    assert all('doc-secret' not in s and 'query-1' not in s for s in inputs)
    assert all(x['doc_id'] in ('doc-secret-1','doc-secret-2') for x in report['results'][0]['ranking'])


def _product_freeze_fixture(tmp_path,manifest,monkeypatch):
    from benchmarks import run_product
    import jieba
    with jieba.get_dict_file() as f:dictionary_sha256=filehash(f.name)
    versions={x:('fixture' if x!='jieba' else run_product.importlib.metadata.version('jieba')) for x in ('torch','transformers','sentence-transformers','numpy','jieba')}
    monkeypatch.setattr(run_product.importlib.metadata,'version',lambda x:versions[x])
    d={'jieba_dictionary_sha256':dictionary_sha256,'source_sha256':run_product.source_hashes(),'datasets':{'fixture':filehash(manifest)},'candidate_top_per_source':40,'candidate_cap':60,'batch_size':2,'device':'mps','dtype':'float16','embedding_models':{k:list(v) for k,v in run_product.EMBED.items()},'rerank_models':{k:list(v) for k,v in run_product.RERANK.items()},'platform':{'system':run_product.platform.system(),'machine':run_product.platform.machine()},'library_versions':versions}
    f=tmp_path/'freeze.json';f.write_text(json.dumps(d));return f


def test_product_rejects_existing_output_or_journal_before_init(tmp_path,monkeypatch):
    import sys
    from benchmarks import run_product
    out=tmp_path/'existing.json';out.write_text('retain')
    monkeypatch.setattr(run_product,'ProductIndex',lambda *_a:pytest.fail('must not initialize'))
    args=['benchmark','--data-dir','unused','--manifest','unused','--freeze','unused','--mode','product-lexical','--output',str(out)]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(FileExistsError):run_product.main()
    assert out.read_text()=='retain'
    out.unlink();journal=Path(str(out)+'.journal.jsonl');journal.write_text('retain')
    with pytest.raises(FileExistsError):run_product.main()
    assert journal.read_text()=='retain'


def test_product_interruption_preserves_completed_row_without_autoresume(tmp_path,monkeypatch,capsys):
    import sys
    from benchmarks import run_product
    root,manifest=make_tiny_dataset(tmp_path);corpus=root/'fixture';m=json.loads(manifest.read_text());m.update(archive_sha256='a'*64,url='https://example.invalid')
    with (corpus/'queries.jsonl').open('a') as f:f.write(json.dumps({'_id':'query-3','text':'blue dog'})+'\n')
    with (corpus/'qrels/train.tsv').open('a') as f:f.write('query-3\tdoc-secret-2\t1\n')
    m['queries_sha256']=filehash(corpus/'queries.jsonl');m['qrels']['train']['sha256']=filehash(corpus/'qrels/train.tsv');m['validation']['query_ids']=['query-1','query-3'];m['validation']['count']=2;manifest.write_text(json.dumps(m))
    freeze=_product_freeze_fixture(tmp_path,manifest,monkeypatch);out=tmp_path/'interrupted.json';calls=0;original=run_product.rank_candidates
    def stop_on_second(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise KeyboardInterrupt('synthetic interrupt')
        return original(*args,**kwargs)
    monkeypatch.setattr(run_product,'rank_candidates',stop_on_second)
    monkeypatch.setattr(sys,'argv',['benchmark','--data-dir',str(root),'--manifest',str(manifest),'--freeze',str(freeze),'--mode','product-lexical','--output',str(out)])
    with pytest.raises(KeyboardInterrupt):run_product.main()
    journal=Path(str(out)+'.journal.jsonl');events=[json.loads(line) for line in journal.read_text().splitlines()]
    assert [x['type'] for x in events]==['header','runtime','query'] and events[2]['completed']==1 and events[2]['row']['query_id']=='query-1'
    assert not out.exists() and '"completed": 1' in capsys.readouterr().out
    with pytest.raises(FileExistsError):run_product.main()


def test_product_sigkill_leaves_fsynced_row_and_refuses_restart(tmp_path,monkeypatch):
    import subprocess,sys,os
    root,manifest=make_tiny_dataset(tmp_path);corpus=root/'fixture';m=json.loads(manifest.read_text());m.update(archive_sha256='a'*64,url='https://example.invalid')
    with (corpus/'queries.jsonl').open('a') as f:f.write(json.dumps({'_id':'query-3','text':'blue dog'})+'\n')
    with (corpus/'qrels/train.tsv').open('a') as f:f.write('query-3\tdoc-secret-2\t1\n')
    m['queries_sha256']=filehash(corpus/'queries.jsonl');m['qrels']['train']['sha256']=filehash(corpus/'qrels/train.tsv');m['validation']['query_ids']=['query-1','query-3'];m['validation']['count']=2;manifest.write_text(json.dumps(m))
    freeze=_product_freeze_fixture(tmp_path,manifest,monkeypatch);out=tmp_path/'killed.json'
    code='''import os,signal,sys,json\nfrom benchmarks import run_product as r\nfrozen=json.load(open(sys.argv[3]));r.importlib.metadata.version=lambda x:frozen['library_versions'][x]\noriginal=r.rank_candidates;count=0\ndef kill_second(*a,**kw):\n global count\n count+=1\n if count==2:os.kill(os.getpid(),signal.SIGKILL)\n return original(*a,**kw)\nr.rank_candidates=kill_second\nsys.argv=['benchmark','--data-dir',sys.argv[1],'--manifest',sys.argv[2],'--freeze',sys.argv[3],'--mode','product-lexical','--output',sys.argv[4]]\nr.main()'''
    proc=subprocess.run([sys.executable,'-u','-c',code,str(root),str(manifest),str(freeze),str(out)],capture_output=True,text=True,timeout=20,env={**os.environ,'PYTHONPATH':str(Path.cwd())})
    assert proc.returncode<0
    journal=Path(str(out)+'.journal.jsonl');items=[json.loads(line) for line in journal.read_text().splitlines()]
    assert [x['type'] for x in items]==['header','runtime','query'] and items[2]['row']['query_id']=='query-1'
    assert not out.exists() and '"completed": 1' in proc.stdout
    proc2=subprocess.run([sys.executable,'-u','-c',code,str(root),str(manifest),str(freeze),str(out)],capture_output=True,text=True,timeout=20,env={**os.environ,'PYTHONPATH':str(Path.cwd())})
    assert proc2.returncode!=0 and 'already exists' in proc2.stderr and journal.read_text().count('\n')==3
