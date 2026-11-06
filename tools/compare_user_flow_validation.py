"""Paired retrospective uncertainty for the frozen validation pair; never selects a run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.verification_metrics import aggregate


def compare(args):
    left=json.loads(Path(args.left).read_text())
    right=json.loads(Path(args.right).read_text())
    assert left['split']==right['split']=='validation'
    assert left['source_sha256']==right['source_sha256']
    assert left['manifest_sha256']==right['manifest_sha256']
    assert left['source_unchanged'] and right['source_unchanged']
    assert left['runtime_identity_unchanged'] and right['runtime_identity_unchanged']
    rows_left=left['rows']; rows_right=right['rows']
    assert [row['query_id'] for row in rows_left]==[row['query_id'] for row in rows_right]
    assert len(rows_left)==32
    samples=np.random.default_rng(20261106).integers(0,len(rows_left),(10000,len(rows_left)))
    values={}
    for name in ('ndcg_at_10','recall_at_20','mrr_at_10'):
        delta=np.array([a['retrieval_metrics'][name]-b['retrieval_metrics'][name] for a,b in zip(rows_left,rows_right,strict=True)])
        distribution=delta[samples].mean(axis=1)
        values[name]={'difference':float(delta.mean()),'low':float(np.quantile(distribution,.025)),
                     'high':float(np.quantile(distribution,.975))}
    linear={name:[] for name in ('claim_label_accuracy','claim_label_macro_f1','cited_document_label_f1')}
    for sample in samples:
        a=aggregate([rows_left[int(index)] for index in sample])
        b=aggregate([rows_right[int(index)] for index in sample])
        for name in linear:linear[name].append(a[name]-b[name])
    full_left=aggregate(rows_left);full_right=aggregate(rows_right)
    for name,distribution in linear.items():
        values[name]={'difference':full_left[name]-full_right[name],
                     'low':float(np.quantile(distribution,.025)),'high':float(np.quantile(distribution,.975))}
    result={'left':Path(args.left).name,'right':Path(args.right).name,
            'left_sha256':hashlib.sha256(Path(args.left).read_bytes()).hexdigest(),
            'right_sha256':hashlib.sha256(Path(args.right).read_bytes()).hexdigest(),
            'queries':len(rows_left),'method':'paired query bootstrap 10000 samples, seed20261106, percentile95%; differences left minus right',
            'limits':'conditional on these two observed runs; does not measure stochastic generation or service variance; not an external-system significance claim; selection was already frozen before this analysis',
            'differences':values}
    with Path(args.output).open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('left','right','output'):parser.add_argument('--'+name,required=True)
    compare(parser.parse_args())
