import pytest
from evidence_rag.catalog import Catalog

M={'doc_id':'civil','version_id':'v1','title':'民法典','source_kind':'synthetic','verification_state':'approved'}
def test_temporal_metadata_requires_coherent_dates(tmp_path):
 c=Catalog(tmp_path/'db');c.stage(['第一条 内容。'],M,'r1')
 c.set_temporal_version('r1',valid_from='2021-01-01',published_at='2020-05-28',revision_note='fixture')
 row=c.temporal_version('r1');assert row['valid_from']=='2021-01-01' and row['published_at']=='2020-05-28'
 with pytest.raises(ValueError,match='valid_from'):c.set_temporal_version('r1',valid_from='2022-01-01',valid_to='2021-01-01')
 with pytest.raises(ValueError,match='published'):c.set_temporal_version('r1',valid_from='2020-01-01',published_at='2021-01-01')
