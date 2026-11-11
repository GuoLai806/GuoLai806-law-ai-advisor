import json
import pytest
import pymupdf
from evidence_rag.ocr import rasterize_pdf,audit_pages,OCRPage

def pdf_bytes():
 d=pymupdf.open();p=d.new_page();p.insert_text((72,72),'Civil Code Article One');out=d.tobytes();d.close();return out

def test_rasterize_pdf_is_lossless_and_deterministic():
 a=rasterize_pdf(pdf_bytes(),dpi=120);b=rasterize_pdf(pdf_bytes(),dpi=120)
 assert len(a)==1 and a[0].startswith(b'\x89PNG') and a[0]==b[0]

def test_audit_flags_empty_and_low_confidence_pages_but_retains_them():
 pages=[OCRPage(1,'合同',(.99,),((1,2,3,4),),500,700),OCRPage(2,'',(),(),500,700),OCRPage(3,'条',(.55,),((1,2,3,4),),500,700)]
 r=audit_pages(pages,['合同','','条'])
 assert r['empty_pages']==[2] and r['low_confidence_pages']==[3] and r['page_count']==3

def test_ocr_missing_backend_is_explicit(monkeypatch):
 from evidence_rag.ocr import paddleocr_pages
 import builtins
 real=builtins.__import__
 def fail(name,*args,**kwargs):
  if name.startswith('paddle'):raise ImportError('fixture')
  return real(name,*args,**kwargs)
 monkeypatch.setattr(builtins,'__import__',fail)
 with pytest.raises(RuntimeError,match='not installed'):paddleocr_pages([b'not-an-image'])

def test_article_coverage_preserves_boundaries_and_reports_missing():
 from evidence_rag.ocr import article_coverage
 arts=[{'label':'第一条'},{'label':'第二条'},{'label':'第三条'}]
 r=article_coverage('第一条 内容一\n第二条 内容二',arts)
 assert r['found_count']==2 and r['missing']==['第三条']
 assert r['articles'][0]['label']=='第一条' and r['articles'][1]['label']=='第二条'

def test_article_text_comparison_is_deterministic_and_keeps_missing():
 from evidence_rag.ocr import compare_article_text
 native=[{'label':'第一条','text':'第一条 为了保护主体。','next_label':'第二条'},{'label':'第二条','text':'第二条 其他。','next_label':'第三条'},{'label':'第三条','text':'第三条 缺失。','next_label':''}]
 r=compare_article_text('第一条 为了保护主体。第二条 其他。',native)
 assert r['articles']==3 and r['found']==2 and r['comparable']==2 and r['missing']==['第三条']

def test_ocr_validation_keeps_one_disagreement_for_review():
 from evidence_rag.ocr import validate_ocr_articles
 native=[{'label':'第一条','text':'第一条 内容一。'},{'label':'第二条','text':'第二条 内容二。'}]
 merged,summary=validate_ocr_articles('第一条 内容一。\n第二条 内容三。',native)
 assert len(merged)==2 and summary['accepted']==1 and summary['review_required']==1
 assert summary['review_labels']==['第二条'] and merged[1]['text_source']=='native_ocr_review_required'
