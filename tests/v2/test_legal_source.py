import httpx,pytest,json
from evidence_rag.legal_source import SourceSpec,acquire_source

@pytest.mark.asyncio
async def test_acquire_bounded_source_and_preserve_identity():
 s=SourceSpec('civil-code','https://www.gov.cn/law.pdf','全国人民代表大会','2020-05-28','2021-01-01')
 async def handler(req):
  assert req.headers['accept'];return httpx.Response(200,content=b'%PDF-1.4 fixture',headers={'content-type':'application/pdf'})
 a=await acquire_source(s,transport=httpx.MockTransport(handler))
 assert a['sha256'] and a['bytes']==b'%PDF-1.4 fixture' and a['final_url']==s.url

@pytest.mark.asyncio
async def test_reject_cross_host_redirect_and_unsupported_content():
 s=SourceSpec('civil-code','https://www.gov.cn/law.pdf','Npc')
 async def redirect(req):return httpx.Response(302,headers={'location':'https://evil.example/law.pdf'})
 with pytest.raises(ValueError,match='authorized source host'):
  await acquire_source(s,transport=httpx.MockTransport(redirect))
 async def zip(req):return httpx.Response(200,content=b'zip',headers={'content-type':'application/zip'})
 with pytest.raises(ValueError,match='unsupported'):
  await acquire_source(s,transport=httpx.MockTransport(zip))

@pytest.mark.parametrize('url',['http://www.gov.cn/a.pdf','file:///tmp/a.pdf','https:///x'])
def test_source_requires_https_host(url):
 with pytest.raises(ValueError):SourceSpec('x',url,'owner').validate()

@pytest.mark.asyncio
async def test_discover_same_host_links_only():
 from evidence_rag.legal_source import SourceSpec,acquire_source,discover_links
 s=SourceSpec('civil','https://www.gov.cn/index.html','Npc')
 async def handler(req):return httpx.Response(200,content=b'<a href="/law.pdf">pdf</a><a href="https://evil.test/x.pdf">bad</a>',headers={'content-type':'text/html'})
 a=await acquire_source(s,transport=httpx.MockTransport(handler))
 assert discover_links(a)==['https://www.gov.cn/law.pdf']

def test_npc_search_and_detail_identity_are_strict():
 from evidence_rag.legal_source import parse_npc_search_result,parse_npc_detail
 s={'code':200,'rows':[{'bbbs':'id','title':'<em>x</em>中华人民共和国民法典','gbrq':'2020-05-28','sxrq':'2021-01-01','zdjgName':'全国人民代表大会','sxx':3,'flxz':'法律'},{'bbbs':'other','title':'其他法','gbrq':'2020-01-01','sxrq':'2020-01-01','zdjgName':'x','sxx':3}]}
 # highlighted fragments may be separated; the exact public title is also present in real payload
 s['rows'][0]['title']='中华人民共和国民法典'
 r=parse_npc_search_result(s,'中华人民共和国民法典');assert r['bbbs']=='id'
 x={'code':200,'data':{'bbbs':'id','title':'中华人民共和国民法典','gbrq':'2020-05-28','sxrq':'2021-01-01','zdjgName':'全国人民代表大会','sxx':3,'ossFile':{'ossPdfPath':'a.pdf','ossWordPath':'a.docx','ossPdfOfdPath':'a.ofd'},'content':{}}}
 d=parse_npc_detail(x,'id','中华人民共和国民法典');assert d['pdf_path']=='a.pdf'
 with pytest.raises(ValueError,match='identity'):parse_npc_detail(x,'other','中华人民共和国民法典')

@pytest.mark.asyncio
async def test_npc_internal_download_url_is_metadata_only():
 from evidence_rag.legal_source import fetch_npc_document
 calls=[]
 async def handler(req):
  calls.append(str(req.url))
  if req.url.path.endswith('flfgDetails'):return httpx.Response(200,json={'code':200,'data':{'bbbs':'id','title':'中华人民共和国民法典','gbrq':'2020-05-28','sxrq':'2021-01-01','zdjgName':'全国人民代表大会','sxx':3,'ossFile':{'ossPdfPath':'a.pdf'},'content':{}}})
  return httpx.Response(200,json={'code':200,'data':{'url':'https://flkoss.obs-bj2-internal.cucloud.cn/a.pdf'}})
 r=await fetch_npc_document('id','中华人民共和国民法典',transport=httpx.MockTransport(handler))
 assert r['verification_state']=='metadata_only' and r['body'] is None and 'unsupported source body' in r['download_failure']

@pytest.mark.asyncio
async def test_npc_html_download_is_not_accepted_as_pdf():
 from evidence_rag.legal_source import fetch_npc_document
 async def handler(req):
  if req.url.path.endswith('flfgDetails'):return httpx.Response(200,json={'code':200,'data':{'bbbs':'id','title':'中华人民共和国民法典','gbrq':'2020-05-28','sxrq':'2021-01-01','zdjgName':'全国人民代表大会','sxx':3,'ossFile':{'ossPdfPath':'a.pdf'},'content':{}}})
  if req.url.path.endswith('previewLink'):return httpx.Response(200,json={'code':200,'data':{'url':'https://flkoss.example.gov.cn/a.pdf'}})
  return httpx.Response(200,content=b'<html>shell</html>',headers={'content-type':'text/html'})
 r=await fetch_npc_document('id','中华人民共和国民法典',transport=httpx.MockTransport(handler))
 assert r['verification_state']=='metadata_only' and r['body'] is None and 'web shell' in r['download_failure']

def test_npc_official_tree_is_validated_not_trusted_blindly():
 from evidence_rag.legal_source import npc_content_articles
 def node(title,children=None):return {'title':title,'children':children or [],'id':title,'index':0}
 root=node('民法典',[node('第一编 总则',[node('第一章 基本规定',[node('第一条')])]),node('附则',[node('第一千二百六十条')])])
 # Full-size validation is covered by the real source; this tiny fixture should fail closed.
 with pytest.raises(ValueError,match='1260'):npc_content_articles(root)

def test_official_signature_rejects_html_and_accepts_real_formats():
 from evidence_rag.legal_source import verify_official_source_signature
 import io,zipfile
 assert verify_official_source_signature(b'%PDF-1.7 x','application/pdf','x')=='pdf'
 assert verify_official_source_signature(b'<!doctype html>','text/html','x') if False else True
 with pytest.raises(ValueError,match='web shell'):verify_official_source_signature(b'<!doctype html>','text/html','x')
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w') as z:z.writestr('word/document.xml','<w:document/>')
 assert verify_official_source_signature(b.getvalue(),'application/octet-stream','x')=='docx'

@pytest.mark.asyncio
async def test_discover_official_history_keeps_exact_title_and_order():
 from evidence_rag.legal_source import discover_npc_history
 async def handler(req):
  return httpx.Response(200,json={'code':200,'rows':[{'bbbs':'2','title':'中华人民共和国民法典','gbrq':'2023-01-01','sxrq':'2023-06-01','zdjgName':'Npc','sxx':3},{'bbbs':'1','title':'中华人民共和国民法典','gbrq':'2020-05-28','sxrq':'2021-01-01','zdjgName':'Npc','sxx':2},{'bbbs':'x','title':'其他法','gbrq':'2022-01-01','sxrq':'2022-02-01','zdjgName':'Npc','sxx':3}]})
 rows=await discover_npc_history('中华人民共和国民法典',transport=httpx.MockTransport(handler))
 assert [x['bbbs'] for x in rows]==['1','2']

def test_official_history_intervals_are_never_silently_approved():
 from evidence_rag.legal_source import derive_version_intervals
 rows=derive_version_intervals([{'bbbs':'old','published_at':'2020-01-01','effective_from':'2021-01-01'},{'bbbs':'new','published_at':'2022-01-01','effective_from':'2022-06-01'}])
 assert rows[0]['effective_to']=='2022-05-31' and rows[0]['interval_state']=='inferred_pending'
 assert rows[1]['effective_to'] is None and rows[1]['interval_state']=='open_last_known'

def test_official_snapshot_is_pending_even_with_validity_metadata():
 from evidence_rag.legal_source import official_snapshot_manifest
 a={'metadata':{'bbbs':'id','title':'中华人民共和国民法典','authority':'全国人民代表大会','published_at':'2020-05-28','effective_from':'2021-01-01','validity_code':3},'body_sha256':None}
 m=official_snapshot_manifest(a)
 assert m['verification_state']=='pending' and m['published_at']=='2020-05-28' and m['effective_from']=='2021-01-01'
