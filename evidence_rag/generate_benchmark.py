"""Generate source-derived synthetic queries, never human labels or a blind benchmark."""
import argparse, json, re
from pathlib import Path
from .catalog import Catalog, tokens

def main():
    p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--revision',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    c=Catalog(args.db); rows=c.db.execute('SELECT label,text FROM units WHERE revision=? ORDER BY rowid',(args.revision,)).fetchall(); output=[]
    for kind in ('exact','excerpt','keywords','exception'):
        chosen=0
        for row in rows:
            text=''.join(row['text'].replace(row['label'],'',1).split());
            if kind=='exception' and not re.search('但是|除外|不得|除非',text): continue
            if kind=='exact': query='请查询'+row['label']
            elif kind=='excerpt': query='关于“'+text[:18]+'”的规定是什么？'
            elif kind=='keywords':
                terms=[x for x in tokens(text).split() if len(x)>1 and re.search('[\u4e00-\u9fff]',x)]
                if len(terms)<3: continue
                query=' '.join(terms[:3])+' 有什么规定？'
            else:
                match=re.search('但是|除外|不得|除非',text); pos=match.start(); query='规定中“'+text[max(0,pos-8):pos+10]+'”有哪些条件？'
            output.append({'query':query,'expected_labels':[row['label']],'generation':kind,'label_source':'source-derived-unverified'});chosen+=1
            if chosen>=75: break
    Path(args.output).write_text(json.dumps({'kind':'synthetic-source-derived-not-human-verified','notice':'Expected labels generated from the source; source text leaks into queries. Not a blind quality benchmark.','queries':output},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'count':len(output),'per_kind':{k:sum(x['generation']==k for x in output) for k in ('exact','excerpt','keywords','exception')}}))
if __name__=='__main__': main()
