"""PaddleOCR and image-only PDF adapter with explicit evidence retention.

PaddleOCR is optional. The deterministic PDF path remains available without
PaddlePaddle; an OCR result is never silently substituted when the backend is
missing. Source images are produced from rendered PDF pages.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import io
import json
import math
import difflib
from pathlib import Path


@dataclass(frozen=True)
class OCRPage:
    page: int
    text: str
    scores: tuple[float, ...]
    boxes: tuple[tuple[float,...], ...]
    width: int
    height: int


def rasterize_pdf(raw: bytes, *, dpi: int = 200, image_format: str = 'png') -> list[bytes]:
    """Render every PDF page into a lossless image for OCR benchmarking."""
    if not 72 <= dpi <= 400:
        raise ValueError('dpi must be in 72..400')
    if image_format.lower() not in ('png','jpg','jpeg'):
        raise ValueError('unsupported image format')
    import pymupdf
    images=[]
    with pymupdf.open(stream=raw,filetype='pdf') as doc:
        for page in doc:
            pix=page.get_pixmap(dpi=dpi,alpha=False)
            fmt='jpeg' if image_format.lower() in ('jpg','jpeg') else image_format.lower()
            images.append(pix.tobytes(fmt))
    if not images:raise ValueError('PDF has no pages')
    return images


def _jsonable(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    if isinstance(value,(list,tuple)):return [_jsonable(x) for x in value]
    if isinstance(value,dict):return {str(k):_jsonable(v) for k,v in value.items()}
    return str(value)


def paddleocr_pages(images, *, lang='ch', device=None, min_score=0.50, progress=None) -> tuple[list[OCRPage],dict]:
    """Run PaddleOCR 3.x over raster pages; retain boxes and line confidence."""
    if not isinstance(min_score,(int,float)) or isinstance(min_score,bool) or not 0<=min_score<=1:
        raise ValueError('min_score must be in [0,1]')
    try:
        import paddle
        import paddleocr
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError('PaddleOCR backend is not installed') from exc
    kwargs={'use_doc_orientation_classify':False,'use_doc_unwarping':False,'use_textline_orientation':True,'lang':lang}
    if device:kwargs['device']=device
    engine=PaddleOCR(**kwargs)
    pages=[];raw_results=[]
    import numpy as np
    import pymupdf
    for page_no,image in enumerate(images,1):
        pixels=image if isinstance(image,str) else image
        if isinstance(pixels,bytes):
            pix=pymupdf.Pixmap(pixels)
            array=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3].copy()
        else:
            pix=pymupdf.Pixmap(pixels)
            array=None
        height,width=pix.height,pix.width
        results=list(engine.predict(pixels if isinstance(pixels,str) else array))
        if progress:progress(page_no,results)
        if not results:
            # Keep an auditable empty page; a scanned cover can legitimately
            # yield no line while other pages remain valid.
            height,width=pix.height,pix.width
            pages.append(OCRPage(page_no,'',(),(),width,height));raw_results.append({'status':'no_results'});continue
        if len(results)!=1:raise ValueError(f'OCR returned {len(results)} results for page {page_no}')
        result=results[0]
        data=result.json if hasattr(result,'json') else result
        data=_jsonable(data)
        # PaddleOCR3 wraps predictions in a res object; accept legacy direct
        # pruned results as well while retaining the raw result for audit.
        pruned=data.get('pruned_result',data.get('res',data)) if isinstance(data,dict) else {}
        # Tests' fake adapters may still return the compact pruned object. In
        # that case data is the structure itself.
        if isinstance(data,dict) and 'rec_texts' in data and 'rec_texts' not in pruned:pruned=data
        texts=pruned.get('rec_texts',[]) if isinstance(pruned,dict) else []
        scores=pruned.get('rec_scores',[]) if isinstance(pruned,dict) else []
        boxes=pruned.get('rec_polys') or pruned.get('rec_boxes') or []
        if not isinstance(texts,list) or not isinstance(scores,list) or len(texts)!=len(scores):raise ValueError('invalid PaddleOCR text/score arrays')
        kept_texts=[];kept_scores=[];kept_boxes=[]
        for i,(text,score) in enumerate(zip(texts,scores)):
            score=float(score)
            if not 0<=score<=1 or not math.isfinite(score):raise ValueError('invalid OCR score')
            if score<min_score:continue
            kept_texts.append(str(text));kept_scores.append(score);kept_boxes.append(tuple(float(v) for v in _flatten_box(boxes[i])) if i<len(boxes) else ())
        pages.append(OCRPage(page_no,'\n'.join(kept_texts),tuple(kept_scores),tuple(kept_boxes),width,height))
        raw_results.append(data)
    versions={'paddle':getattr(paddle,'__version__',None),'paddleocr':getattr(paddleocr,'__version__',None),'lang':lang,'device':device,'min_score':min_score}
    return pages,{'versions':versions,'results':raw_results}


def _flatten_box(box):
    if isinstance(box,(int,float)):return [box]
    if isinstance(box,(list,tuple)):
        out=[]
        for value in box:out.extend(_flatten_box(value))
        return out
    if hasattr(box,'tolist'):return _flatten_box(box.tolist())
    return []


def image_dimensions(image):
    import pymupdf
    pix=pymupdf.Pixmap(image)
    return pix.height,pix.width


def article_segments(ocr_text: str, labels: list[str]) -> dict[str,str]:
    """Choose likely definition occurrences and keep them in document order."""
    import re,unicodedata
    text=unicodedata.normalize('NFKC',ocr_text).replace('第一干','第一千')
    chosen=[]
    previous=0
    for label in labels:
        candidates=[m.start() for m in re.finditer(re.escape(label),text)]
        if not candidates:
            chosen.append(None);continue
        after_previous=[p for p in candidates if p>=previous]
        if not after_previous:
            chosen.append(None);continue
        # References often have explicit legal framing. If direct candidates
        # are ambiguous, prefer the first after the prior article definition.
        picked=after_previous[0]
        for p in after_previous[:8]:
            left=text[max(0,p-10):p]
            right=text[p+len(label):p+len(label)+5].lstrip()
            if re.search(r'(?:本法|适用|依照|根据|符合|违反|按照)$',left) or re.match(r'^(?:规定|的|至|和|及|与)',right):
                continue
            picked=p;break
        chosen.append(picked);previous=picked+len(label)
    pairs=[(p,label) for p,label in zip(chosen,labels) if p is not None]
    pairs.sort()
    out={}
    for i,(p,label) in enumerate(pairs):
        end=pairs[i+1][0] if i+1<len(pairs) else len(text)
        out[label]=text[p:end]
    return out


def compare_article_text(ocr_text: str, native_articles: list[dict]) -> dict:
    """Compare OCR article bodies against clean native article bodies."""
    import re,unicodedata
    compact=lambda x:''.join(unicodedata.normalize('NFKC',x).split()).replace('，',',').replace('。','.')
    ocr=compact(ocr_text).replace('第一干','第一千')
    records=[];missing=[];exact=[];similarities=[]
    ordered=article_segments(ocr_text,[a.get('label','') for a in native_articles])
    for article in native_articles:
        label=compact(article.get('label',''));native=compact(article.get('text',''))
        segment=ordered.get(article.get('label'))
        if segment is not None:segment=compact(segment)
        if segment is None:
            missing.append(article.get('label'));records.append({'label':article.get('label'),'found':False});continue
        # A direct containment check is far safer than fuzzy text matching.
        matched=native and (native in segment or segment in native)
        ratio=difflib.SequenceMatcher(None,native,segment,autojunk=False).ratio() if native and segment else 0.0
        similarities.append(ratio)
        records.append({'label':article.get('label'),'found':True,'native_chars':len(native),'ocr_chars':len(segment),'native_in_ocr':bool(native and native in segment),'ocr_in_native':bool(native and segment in native),'comparable':matched,'normalized_edit_similarity':ratio,'cer_proxy':1.0-ratio})
        if matched:exact.append(article.get('label'))
    return {'articles':len(native_articles),'found':len(native_articles)-len(missing),'comparable':len(exact),'mean_normalized_edit_similarity':sum(similarities)/len(similarities) if similarities else None,'mean_cer_proxy':1.0-sum(similarities)/len(similarities) if similarities else None,'missing':missing,'records':records}


def validate_ocr_articles(ocr_text: str, native_articles: list[dict]) -> tuple[list[dict],dict]:
    """Merge native article text only when OCR agrees; retain disagreements."""
    comparison=compare_article_text(ocr_text,native_articles)
    segments=article_segments(ocr_text,[a.get('label','') for a in native_articles])
    accepted=[];rejected=[]
    review=[]
    by_label={x['label']:x for x in comparison['records']}
    for article in native_articles:
        result=by_label.get(article['label'],{})
        ocr=segments.get(article['label'],'')
        item={**article,'ocr_text':ocr,'ocr_match':bool(result.get('comparable'))}
        if result.get('comparable'):
            item['text_source']='native_ocr_verified'
            accepted.append(item)
        else:
            item['verification_state']='ocr_disagreement'
            item['text_source']='native_ocr_review_required'
            rejected.append(item);review.append({'label':article['label'],'ocr_text':ocr,'native_text':article['text'],'reason':'native/OCR article body did not pass compact containment'})
    summary={'accepted':len(accepted),'review_required':len(rejected),'review_labels':[x['label'] for x in rejected],'review':review,'comparison':{k:v for k,v in comparison.items() if k!='records'}}
    return accepted+rejected,summary


def article_coverage(ocr_text: str, articles: list[dict]) -> dict:
    """Locate article boundaries in OCR text while reducing visual line breaks."""
    import re
    # OCR bounding boxes commonly split wrapped lines; compare a compact form
    # while retaining start/end locators as an audit aid, not source offsets.
    normalized=ocr_text.replace('\n','').replace('\u3000','').replace(' ','').replace('第一干','第一千')
    found=[]
    missing=[]
    for index,article in enumerate(articles):
        label=article.get('label','').replace(' ','')
        if not label:continue
        start=normalized.find(label)
        if start<0:
            missing.append(label);continue
        next_label=articles[index+1].get('label','').replace(' ','') if index+1<len(articles) else None
        end=normalized.find(next_label,start+len(label)) if next_label else len(normalized)
        if end<0:end=len(normalized)
        body=normalized[start:end]
        found.append({'label':label,'char_start':start,'char_end':end,'chars':len(body),'raw_sha256':__import__('hashlib').sha256(body.encode()).hexdigest()})
    return {'article_count':len(articles),'found_count':len(found),'missing_count':len(missing),'missing':missing,'articles':found}


def audit_pages(pages, native_pages=None) -> dict:
    """Keep low-confidence/no-text pages visible for manual/automatic review."""
    records=[]
    for p in pages:
        line_count=len(p.scores)
        records.append({'page':p.page,'chars':len(p.text),'lines':line_count,'mean_score':sum(p.scores)/line_count if line_count else None,'min_score':min(p.scores) if p.scores else None,'width':p.width,'height':p.height})
    result={'pages':records,'page_count':len(pages),'total_chars':sum(x['chars'] for x in records),'empty_pages':[x['page'] for x in records if not x['chars']],'low_confidence_pages':[x['page'] for x in records if x['min_score'] is not None and x['min_score']<0.6]}
    if native_pages is not None:
        if len(native_pages)!=len(pages):raise ValueError('native and OCR page count differ')
        norms=[]
        for native,ocr in zip(native_pages,pages):
            a=''.join(native.split());b=''.join(ocr.text.split())
            norms.append({'page':ocr.page,'native_chars':len(a),'ocr_chars':len(b),'exact_normalized':a==b})
        result['native_comparison']=norms
    return result
