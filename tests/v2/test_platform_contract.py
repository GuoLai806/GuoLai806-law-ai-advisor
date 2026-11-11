from evidence_rag.legal_source import verify_official_source_signature

def test_contract_requires_pdf_signature_not_content_type_claim():
 import pytest
 with pytest.raises(ValueError,match='web shell'):
  verify_official_source_signature(b'<html>x</html>','application/pdf','x')
 assert verify_official_source_signature(b'%PDF-1.5 valid','text/plain','x')=='pdf'
