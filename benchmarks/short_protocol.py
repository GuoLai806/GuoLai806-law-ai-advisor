"""Verify frozen paired evidence before short formal calls and during artifact audit."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from .e2e import file_hash, source_hashes
from copy import deepcopy


def require(ok,message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())


def validate_selection(selection_path,config_path,manifest_path,formal_output,*,audit=False):
    selected=read(selection_path)
    required={'kind','config_path','config_sha256','manifest_path','manifest_sha256','source_sha256',
              'runtime_identity','formal_output','formal_query_ids','development6','paired_selection_path',
              'freeze_pair_sha256','freeze_tool_sha256','selected_utc'}
    require(required<=set(selected),'Short selection requires full development6 and paired freeze evidence')
    require(selected['kind'] in ('baseline','candidate'),'Unknown frozen mode')
    require(Path(selected['formal_output']).resolve()==Path(formal_output).resolve(),'Selection bound to a different formal output')
    require(Path(selected['config_path']).resolve()==Path(config_path).resolve()
            and selected['config_sha256']==file_hash(config_path),'Frozen config changed')
    require(Path(selected['manifest_path']).resolve()==Path(manifest_path).resolve()
            and selected['manifest_sha256']==file_hash(manifest_path),'Frozen manifest changed')
    manifest=read(manifest_path); config=read(config_path); source=source_hashes()
    require(selected['formal_query_ids']==manifest['short_evaluation']['query_ids'],'Frozen formal membership changed')
    require(selected['source_sha256']==config['source_sha256']==source,'Frozen source changed')
    require(selected['runtime_identity']==config['runtime_identity'],'Frozen runtime changed')
    require(selected['freeze_tool_sha256']==file_hash(Path(__file__).resolve().parents[1]/'tools/freeze_short_run.py'),
            'Freeze tool changed')
    peer_path=Path(selected['paired_selection_path'])
    require(peer_path.resolve()!=Path(selection_path).resolve(),'Pair must contain two selections')
    peer=read(peer_path)
    require(required<=set(peer),'Incomplete peer selection')
    require(peer['kind']!=selected['kind'] and peer['kind'] in ('baseline','candidate'),'Invalid paired modes')
    require(Path(peer['paired_selection_path']).resolve()==Path(selection_path).resolve(),'Non-reciprocal paired selection')
    pair=sorted((selected,peer),key=lambda x:0 if x['kind']=='baseline' else 1)
    digest=hashlib.sha256(json.dumps([{k:v for k,v in x.items() if k!='freeze_pair_sha256'} for x in pair],sort_keys=True).encode()).hexdigest()
    require(all(x['freeze_pair_sha256']==digest for x in pair),'Paired freeze hash changed')
    require(Path(peer['formal_output']).resolve()!=Path(selected['formal_output']).resolve(),'Shared paired output')
    for key in ('source_sha256','manifest_sha256','formal_query_ids','development6','selected_utc','freeze_tool_sha256'):
        require(peer[key]==selected[key],'Paired identity changed: '+key)
    peer_config=read(peer['config_path'])
    require(peer['config_sha256']==file_hash(peer['config_path']) and peer_config['source_sha256']==source
            and peer_config['runtime_identity']==peer['runtime_identity'],'Peer config changed')
    normalized=[]
    for item in pair:
        settings=read(item['config_path']);service=settings['service_configuration'];runtime=settings['runtime_identity']
        mode='text' if item['kind']=='baseline' else 'grounded_json'
        gate='reranker_threshold' if item['kind']=='baseline' else 'grounded_evidence'
        require(service.get('answer_format')==runtime.get('answer_format')==mode
                and service.get('generation_gate')==runtime.get('generation_gate')==gate,'Paired config mode changed')
        if item['kind']=='baseline':
            require(str(service.get('rerank_threshold')) in ('0','0.0') and str(runtime.get('rerank_threshold')) in ('0','0.0'),
                    'Historical baseline threshold changed')
        copy=deepcopy(settings)
        for key in ('answer_format','generation_gate','general_system_prompt'):copy['service_configuration'].pop(key,None)
        for key in ('answer_format','generation_gate','system_prompt_sha256'):copy['runtime_identity'].pop(key,None)
        normalized.append(copy)
    require(normalized[0]==normalized[1],'Paired configs differ outside the declared joint strategy')
    d=selected['development6']; report=read(d['report_path']); dev_config=read(d['config_path'])
    require(d['report_sha256']==file_hash(d['report_path']) and d['config_sha256']==file_hash(d['config_path'])
            and d['journal_sha256']==file_hash(d['journal_path']),'Development evidence changed')
    require(Path(d['journal_path']).resolve()==Path(str(d['report_path'])+'.journal.jsonl').resolve(),'Development journal path changed')
    require(report['split']=='validation' and report['queries_expected']==report['queries_recorded']==len(report['rows'])==6,
            'Complete development6 required')
    ids=[r['query_id'] for r in report['rows']]
    require(ids==d['query_ids']==manifest['validation']['query_ids'] and d['queries_recorded']==6,'Development membership changed')
    require(dict(Counter(r['gold_label'] for r in report['rows']))==manifest['validation']['label_counts'],'Development labels changed')
    require(report['source_sha256']==dev_config['source_sha256']==source and report['manifest_sha256']==file_hash(manifest_path)
            and report['config_sha256']==d['config_sha256'] and report['source_identity_checked']
            and report['source_unchanged'] and report['runtime_identity_unchanged'],'Development identity changed')
    require(all(report['runtime_identity'].get(k)==v for k,v in dev_config['runtime_identity'].items()),'Development runtime changed')
    candidate=next(x for x in pair if x['kind']=='candidate')
    require(dev_config==read(candidate['config_path']),'Candidate differs from development6')
    require(report['failures']==d['evaluation_failures']==sum(bool(r['failure']) for r in report['rows']),
            'Development failures changed')
    journal=[json.loads(line) for line in Path(d['journal_path']).read_text().splitlines()]
    require(len(journal)==8 and journal[0]['type']=='header' and journal[-1]['type']=='complete','Development journal incomplete')
    require(journal[0]['query_ids']==ids and journal[0]['config_sha256']==d['config_sha256']
            and journal[0]['manifest_sha256']==file_hash(manifest_path) and journal[0]['source_sha256']==source,'Development header changed')
    require(all(e['type']=='query' and e['completed']==i and e['row']==r for i,(e,r) in enumerate(zip(journal[1:-1],report['rows'],strict=True),1))
            and journal[-1]['report_sha256']==d['report_sha256'],'Development journal rows changed')
    frozen_at=datetime.fromisoformat(selected['selected_utc'].replace('Z','+00:00'))
    for item in pair:
        path=Path(item['formal_output']); jpath=Path(str(path)+'.journal.jsonl')
        if not audit and item['kind']==selected['kind']:
            require(not path.exists() and not jpath.exists(),'Preserve earlier formal record')
        if jpath.exists():
            header=json.loads(jpath.read_text().splitlines()[0])
            require(header['type']=='header' and header['source_sha256']==source
                    and header['manifest_sha256']==selected['manifest_sha256'] and header['config_sha256']==item['config_sha256']
                    and header['query_ids']==selected['formal_query_ids'] and header.get('freeze_pair_sha256')==digest,
                    'Formal journal is outside the frozen pair')
            require(datetime.fromisoformat(header['started_utc'].replace('Z','+00:00'))>=frozen_at,'Pair frozen after formal calls')
        elif path.exists(): require(False,'Formal report missing journal')
    return {'selection_sha256':file_hash(selection_path),'freeze_pair_sha256':digest,
            'development_report_sha256':d['report_sha256']}
