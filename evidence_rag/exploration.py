"""Bounded discovery previews, deliberately separate from generation evidence.

The caller supplies complete ranked paragraphs before the generation budget is
applied. Truncation affects only this response; hashes and the read pointer still
identify the complete pinned source. A preview is never an EvidenceBundle.
"""
import hashlib
from typing import Any, Literal
from urllib.parse import quote, urlencode

from pydantic import BaseModel, ConfigDict, Field


class ExploreItem(BaseModel):
    model_config = ConfigDict(extra='forbid')

    unit_id: str = Field(min_length=1)
    document_id: str | None = None
    title: str
    revision: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    document_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    locator: dict[str, Any]
    part_no: int | None = Field(default=None, ge=1)
    preview: str
    preview_truncated: bool
    has_more: bool
    read_pointer: str


class ExploreBundle(BaseModel):
    model_config = ConfigDict(extra='forbid')

    kind: Literal['exploration'] = 'exploration'
    status: Literal['exploration_only'] = 'exploration_only'
    corpus_revision: str = Field(min_length=1)
    trace_id: str | None = None
    degraded: bool
    degraded_reasons: list[str]
    retrieval_ms: float = Field(ge=0, allow_inf_nan=False)
    retrieval_trace: dict[str, Any] | None = None
    items: list[ExploreItem]


def explore_bundle(bundle: dict[str, Any], preview_chars: int = 320) -> dict[str, Any]:
    """Serialize bounded previews of complete, hash-verified ranked paragraphs.

    ``content_sha256`` refers to the untruncated paragraph. For a paragraph in a
    larger document, ``document_sha256`` refers to its enclosing full document;
    that hash is verified by the original-source endpoint when the client reads
    it. Missing document hashes are allowed only for whole-unit evidence.
    """
    if type(preview_chars) is not int or not 1 <= preview_chars <= 2000:
        raise ValueError('preview_chars must be an integer in 1..2000')
    revision = bundle['corpus_revision']
    if not isinstance(revision, str) or not revision:
        raise ValueError('exploration requires a pinned revision')
    items = []
    for item in bundle['evidence']:
        text = item['text']
        if not isinstance(text, str):
            raise ValueError('exploration requires complete source text')
        expected = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if item['content_sha256'] != expected:
            raise ValueError('exploration source hash mismatch')
        if item.get('part_no') is not None and not item.get('document_sha256'):
            raise ValueError('paragraph exploration requires its document hash')
        unit_id = item['unit_id']
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError('exploration requires a source unit')
        truncated = len(text) > preview_chars
        items.append(ExploreItem(
            unit_id=unit_id,
            document_id=item.get('document_id'),
            title=item['title'],
            revision=revision,
            content_sha256=expected,
            document_sha256=item.get('document_sha256') or expected,
            locator=item['source_locator'],
            part_no=item.get('part_no'),
            preview=text[:preview_chars],
            preview_truncated=truncated,
            has_more=bool(item.get('has_more', False)) or truncated,
            read_pointer='/api/v2/units/' + quote(unit_id, safe='') + '?' + urlencode({'revision': revision}),
        ))
    response = ExploreBundle(
        corpus_revision=revision,
        trace_id=bundle.get('trace_id'),
        degraded=bundle['degraded'],
        degraded_reasons=bundle['degraded_reasons'],
        retrieval_ms=bundle['retrieval_ms'],
        retrieval_trace=bundle.get('retrieval_trace'),
        items=items,
    )
    result = response.model_dump()
    if result['retrieval_trace'] is None:
        del result['retrieval_trace']
    return result
