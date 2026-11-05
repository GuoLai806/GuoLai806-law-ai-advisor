import argparse,collections,hashlib,json,re,sqlite3,time,unicodedata
from html.parser import HTMLParser
from legal_rag.catalog import canonical_label
p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--miit',required=True);p.add_argument('--samr',required=True);p.add_argument('--output',required=True);args=p.parse_args()
class Paragraphs(HTMLParser):
 def __init__(self):super().__init__();self.stack=0;self.content=[];self.items=[]
 def handle_starttag(self,tag,attrs):
  if tag=='p':self.stack+=1
  if tag=='br' and self.stack:self.content.append('\n')
 def handle_endtag(self,tag):
  if tag=='p' and self.stack:
   self.stack-=1
   if not self.stack:self.items.append(''.join(self.content));self.content=[]
 def handle_data(self,text):
  if self.stack:self.content.append(text)
head=re.compile(r'^\s*(第[一二三四五六七八九十百千万零〇两\d]+条(?:之[一二三四五六七八九十\d]+)?)\s*')
section=re.compile(r'^\s*(?:第[一二三四五六七八九十百千万零〇两\d]+(?:编|分编|章|节)|附\s*则)')
def extract(name):
 p=Paragraphs();p.feed(open(getattr(args,name),encoding='utf-8').read());articles={};current=None
 for para in p.items:
  text=para.strip(); m=head.match(text)
  if m:
   key=canonical_label(m.group(1));current=key
   articles.setdefault(key,[]).append(text)
  elif current and section.match(text):current=None
  elif current and text:articles[current].append(text)
 return {k:'\n'.join(v) for k,v in articles.items()}
def norm(text): return ''.join(unicodedata.normalize('NFKC',text).split())
c=sqlite3.connect(args.db);c.row_factory=sqlite3.Row
pdf={canonical_label(r['label']):r['text'] for r in c.execute('SELECT label,text FROM units WHERE revision=?',('repository-unverified',))}
results={}
for name in ['miit','samr']:
 articles=extract(name);missing=[i for i in range(1,1261) if (i,0) not in articles];differences=[]
 for key in sorted(set(pdf)&set(articles)):
  a=norm(pdf[key]);b=norm(articles[key]);
  if a!=b:
   prefix=0
   while prefix<min(len(a),len(b)) and a[prefix]==b[prefix]:prefix+=1
   differences.append({'article':key[0],'pdf_length':len(a),'official_length':len(b),'first_diff_offset':prefix,'pdf_excerpt':a[max(0,prefix-20):prefix+45],'official_excerpt':b[max(0,prefix-20):prefix+45]})
 results[name]={'article_count':len(articles),'missing_numbers':missing,'comparable_articles':len(set(pdf)&set(articles)),'normalized_exact_matches':len(set(pdf)&set(articles))-len(differences),'differences':differences}
print(json.dumps({k:{x:v for x,v in d.items() if x!='differences'}|{'difference_count':len(d['differences']),'first_differences':d['differences'][:5]} for k,d in results.items()},ensure_ascii=False,indent=2))
record={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'script_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest(),'corpus_manifest':json.loads(c.execute('SELECT manifest FROM revisions WHERE revision=?',('repository-unverified',)).fetchone()[0]),'input_sha256':{name:hashlib.sha256(open(getattr(args,name),'rb').read()).hexdigest() for name in ('miit','samr')},'method':'HTML paragraph boundaries and PDF extracted article text; NFKC plus whitespace removal; not visual review','results':results,'limitation':'Government repost is not the promulgating authority; software extraction and differences need human validation.'}
open(args.output,'w').write(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
