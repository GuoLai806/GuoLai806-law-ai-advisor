import json

import pytest

from benchmarks.prepare import jsonl


@pytest.mark.parametrize('delimiter', ['\n', '\r\n', '\r'])
def test_unicode_text_is_preserved_inside_json_strings(tmp_path, delimiter):
    data = [{'_id': 'a', 'text': 'before\u0085after\u2028paragraph\u2029end'},
            {'_id': 'b', 'text': '最后一条'}]
    path = tmp_path / 'corpus.jsonl'
    path.write_bytes(delimiter.join(json.dumps(row, ensure_ascii=False) for row in data).encode('utf-8'))
    assert jsonl(path) == {row['_id']: row for row in data}


def test_jsonl_rejects_an_actual_broken_record(tmp_path):
    path = tmp_path / 'broken.jsonl'
    path.write_bytes(b'{"_id":"one","text":"broken\nline"}\n')
    with pytest.raises(json.JSONDecodeError):
        jsonl(path)


def test_jsonl_rejects_duplicate_identifiers(tmp_path):
    path = tmp_path / 'duplicate.jsonl'
    path.write_text('{"_id":"same"}\n{"_id":"same"}\n')
    with pytest.raises(ValueError, match='duplicate or invalid identifier'):
        jsonl(path)
