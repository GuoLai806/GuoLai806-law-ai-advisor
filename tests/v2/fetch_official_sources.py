import concurrent.futures,hashlib,json,time,urllib.request
sources={
 'miit':'https://tjca.miit.gov.cn/zwgk/zcwj/flfg/art/2020/art_20cf1a2e1b854924b5caa744c8045d1f.html',
 'samr':'https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2023/art_60fb3863b7364b759123a42d1df568ca.html',
 'court':'https://ipc.court.gov.cn/zh-cn/news/view-371.html',
 'npc_gazette':'https://www.npc.gov.cn/wxzlhgb/c27214/gb2020/202006/P020230313538731037747.pdf',
 'fuzhou_pdf':'https://wlj.fuzhou.gov.cn/zwgk/lyfg/202007/P020200708571723041605.pdf',
}
def fetch(item):
 name,url=item;r={'name':name,'url':url,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=18) as response:
   raw=response.read(10000001);r.update(status=response.status,final_url=response.url,content_type=response.headers.get('content-type'),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),pdf_header=raw.startswith(b'%PDF-'))
   if len(raw)<10000000:open('/tmp/law-source-'+name,'wb').write(raw)
 except Exception as exc:r.update(error=type(exc).__name__,message=str(exc))
 return r
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: results=list(pool.map(fetch,sources.items()))
open('docs/development/legal-rag-20261105/evidence/official-alternates-fetch.json','w').write(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False,indent=2))
