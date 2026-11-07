"""CLI/index ownership and unknown-delivery preservation, without live index writes."""
import asyncio
import json
import sqlite3
import sys
from types import SimpleNamespace

import httpx
import pytest

from evidence_rag.catalog import Catalog, digest
from evidence_rag.retrieval import OpenViking, document_embedding_text
from evidence_rag import viking_index
from benchmarks import index_openviking
from benchmarks.e2e import file_hash

SCOPE = 'viking://resources/rag/fixture/r1/'


def prepared(tmp_path, mapped=False):
    db = tmp_path / 'catalog.sqlite3'
    cat = Catalog(db)
    cat.stage_documents([{'_id': 'd1', 'title': 'Synthetic', 'text': 'Complete alpha evidence.'}],
                        {'corpus_id': 'fixture', 'version_id': 'v1', 'source_kind': 'synthetic',
                         'verification_state': 'provided'}, 'r1')
    uid = cat.db.execute('SELECT unit_id FROM units WHERE revision=?', ('r1',)).fetchone()[0]
    content = document_embedding_text(cat.get('r1', uid))
    uri = SCOPE + uid + '/' + uid + '.md'
    if mapped:
        cat.db.execute('CREATE TABLE viking_journal(revision TEXT,unit_id TEXT,uri TEXT,content_hash TEXT,state TEXT,PRIMARY KEY(revision,unit_id))')
        cat.db.execute('INSERT INTO viking_journal VALUES(?,?,?,?,?)', ('r1', uid, uri, digest(content), 'delivery_unknown'))
        cat.db.execute('UPDATE units SET uri=? WHERE revision=? AND unit_id=?', (uri, 'r1', uid))
    cat.db.close()
    return db, uid, content, uri


def install_adapter(monkeypatch, module, handler):
    instances, loops = [], []

    class Tracked(OpenViking):
        async def find(self, query):
            loops.append(asyncio.get_running_loop())
            return await super().find(query)

        async def ingest(self, uid, content):
            loops.append(asyncio.get_running_loop())
            return await super().ingest(uid, content)

        async def aclose(self):
            loops.append(asyncio.get_running_loop())
            await super().aclose()

    def factory(*args, **kwargs):
        instance = Tracked(*args, transport=httpx.MockTransport(handler), **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(module, 'OpenViking', factory)
    return instances, loops


@pytest.mark.parametrize('operation', ['send', 'reconcile', 'validate'])
def test_real_cli_closes_adapter_on_its_operation_event_loop(tmp_path, monkeypatch, capsys, operation):
    db, uid, content, uri = prepared(tmp_path, mapped=operation != 'send')
    calls = []

    async def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('temp_upload'):
            return httpx.Response(200, json={'status': 'ok', 'result': {'temp_file_id': 'tmp'}})
        if request.url.path.endswith('/resources'):
            return httpx.Response(200, json={'status': 'ok', 'result': {'root_uri': SCOPE + uid}})
        if request.url.path.endswith('/read'):
            return httpx.Response(200, json={'status': 'ok', 'result': content})
        return httpx.Response(200, json={'status': 'ok', 'result': {'resources': [{'uri': uri}]}})

    instances, loops = install_adapter(monkeypatch, viking_index, handler)
    monkeypatch.setattr(sys, 'argv', ['viking_index', operation, '--db', str(db), '--revision', 'r1',
                                     '--unit-id', uid, '--url', 'http://fixture', '--scope', SCOPE])
    viking_index.main()
    result = json.loads(capsys.readouterr().out)
    assert len(instances) == 1 and instances[0]._state == 'closed'
    assert all(loop is loops[0] for loop in loops)
    assert loops[0].is_closed()  # asyncio.run has returned only after teardown.
    assert calls
    if operation == 'send':
        assert result['state'] == 'delivery_unknown'
    elif operation == 'reconcile':
        assert result['state'] == 'content_verified'
    else:
        assert result['index_visibility_validated'] is True


@pytest.mark.asyncio
async def test_cli_timeout_preserves_unknown_journal_and_second_call_does_not_replay(tmp_path, monkeypatch):
    db, uid, _, _ = prepared(tmp_path)
    calls = []

    async def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('temp_upload'):
            return httpx.Response(200, json={'status': 'ok', 'result': {'temp_file_id': 'tmp'}})
        raise httpx.ReadTimeout('unknown delivery', request=request)

    instances, loops = install_adapter(monkeypatch, viking_index, handler)
    args = SimpleNamespace(operation='send', db=db, revision='r1', unit_id=uid,
                           url='http://fixture', scope=SCOPE)
    with pytest.raises(httpx.ReadTimeout):
        await viking_index.execute(args)
    cat = Catalog(db)
    assert cat.db.execute('SELECT state FROM viking_journal').fetchone()[0] == 'delivery_unknown'
    cat.db.close()
    assert (await viking_index.execute(args))['state'] == 'delivery_unknown'
    assert calls == ['/api/v1/resources/temp_upload', '/api/v1/resources']
    assert all(instance._state == 'closed' for instance in instances)
    assert all(loop is asyncio.get_running_loop() for loop in loops)


@pytest.mark.asyncio
async def test_cli_cancelled_write_preserves_unknown_journal_and_closes_adapter(tmp_path, monkeypatch):
    db, uid, _, _ = prepared(tmp_path)
    entered = asyncio.Event()
    calls = []

    async def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('temp_upload'):
            return httpx.Response(200, json={'status': 'ok', 'result': {'temp_file_id': 'tmp'}})
        entered.set()
        await asyncio.Event().wait()

    instances, _ = install_adapter(monkeypatch, viking_index, handler)
    args = SimpleNamespace(operation='send', db=db, revision='r1', unit_id=uid,
                           url='http://fixture', scope=SCOPE)
    task = asyncio.create_task(viking_index.execute(args))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert instances[0]._state == 'closed' and instances[0]._client.is_closed
    cat = Catalog(db)
    assert cat.db.execute('SELECT state FROM viking_journal').fetchone()[0] == 'delivery_unknown'
    cat.db.close()
    assert calls == ['/api/v1/resources/temp_upload', '/api/v1/resources']


def benchmark_args(tmp_path):
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(json.dumps({'_id': 'd1', 'title': 'Synthetic', 'text': 'alpha evidence'}) + '\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'corpus_sha256': file_hash(corpus), 'corpus_count': 1,
                                    'archive_sha256': 'fixture', 'url': 'https://fixture.invalid'}))
    return SimpleNamespace(output=str(tmp_path / 'report.json'), manifest=str(manifest),
                           corpus=str(corpus), db=str(tmp_path / 'db.sqlite3'), corpus_id='fixture',
                           revision='r1', url='http://fixture', operation='index', concurrency=1)


@pytest.mark.asyncio
async def test_benchmark_stage_failure_still_closes_catalog(tmp_path, monkeypatch):
    args = benchmark_args(tmp_path)
    instances = []

    class BrokenCatalog(Catalog):
        def __init__(self, path):
            super().__init__(path)
            instances.append(self)

        def stage_documents(self, *args, **kwargs):
            raise ValueError('invalid source')

    monkeypatch.setattr(index_openviking, 'Catalog', BrokenCatalog)
    with pytest.raises(ValueError, match='invalid source'):
        await index_openviking.execute(args)
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        instances[0].db.execute('SELECT 1')


@pytest.mark.asyncio
async def test_benchmark_journal_initialization_failure_closes_adapter_and_catalog(tmp_path, monkeypatch):
    args = benchmark_args(tmp_path)
    catalogs = []

    class TrackedCatalog(Catalog):
        def __init__(self, path):
            super().__init__(path)
            catalogs.append(self)

    def broken_journal(path):
        raise OSError('disk unavailable')

    instances, loops = install_adapter(monkeypatch, index_openviking,
                                       lambda request: httpx.Response(200, json={}))
    monkeypatch.setattr(index_openviking, 'Catalog', TrackedCatalog)
    monkeypatch.setattr(index_openviking, 'Journal', broken_journal)
    with pytest.raises(OSError, match='disk unavailable'):
        await index_openviking.execute(args)
    assert instances[0]._state == 'closed' and loops == [asyncio.get_running_loop()]
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        catalogs[0].db.execute('SELECT 1')


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['timeout', 'cancel'])
async def test_benchmark_failed_write_closes_resources_without_replaying(tmp_path, monkeypatch, failure):
    args = benchmark_args(tmp_path)
    entered = asyncio.Event()
    calls = []
    catalogs = []
    journals = []
    original_journal = index_openviking.Journal

    def tracked_journal(path):
        journal = original_journal(path)
        journals.append(journal)
        return journal

    class TrackedCatalog(Catalog):
        def __init__(self, path):
            super().__init__(path)
            catalogs.append(self)

    async def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith('temp_upload'):
            return httpx.Response(200, json={'status': 'ok', 'result': {'temp_file_id': 'tmp'}})
        entered.set()
        if failure == 'timeout':
            raise httpx.ReadTimeout('unknown delivery', request=request)
        await asyncio.Event().wait()

    instances, loops = install_adapter(monkeypatch, index_openviking, handler)
    original_client = httpx.AsyncClient
    health_transport = httpx.MockTransport(lambda request: httpx.Response(200, json={'status': 'ok'}))

    def client(*args, **kwargs):
        kwargs.setdefault('transport', health_transport)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(index_openviking, 'Catalog', TrackedCatalog)
    monkeypatch.setattr(index_openviking, 'Journal', tracked_journal)
    monkeypatch.setattr(httpx, 'AsyncClient', client)
    if failure == 'cancel':
        task = asyncio.create_task(index_openviking.execute(args))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await index_openviking.execute(args) == 1
        report = json.loads((tmp_path / 'report.json').read_text())
        assert report['complete'] is False
        assert report['units'][0]['failure']['type'] == 'ReadTimeout'
    assert instances[0]._state == 'closed' and instances[0]._client.is_closed
    assert all(loop is asyncio.get_running_loop() for loop in loops)
    with pytest.raises(sqlite3.ProgrammingError, match='closed database'):
        catalogs[0].db.execute('SELECT 1')
    cat = Catalog(args.db)
    assert cat.db.execute('SELECT state FROM viking_journal').fetchone()[0] == 'delivery_unknown'
    cat.db.close()
    assert calls == ['/api/v1/resources/temp_upload', '/api/v1/resources']
    assert len(journals) == 1 and journals[0].stream.closed
    assert json.loads(journals[0].path.read_text().splitlines()[0])['type'] == 'header'


@pytest.mark.asyncio
async def test_validation_observer_failure_drains_sibling_reader_before_return(tmp_path):
    cat = Catalog(tmp_path / 'db.sqlite3')
    cat.stage_documents([{'_id': 'a', 'title': 'A', 'text': 'alpha'},
                         {'_id': 'b', 'title': 'B', 'text': 'beta'}],
                        {'corpus_id': 'fixture', 'version_id': 'v1', 'source_kind': 'synthetic',
                         'verification_state': 'provided'}, 'r1')
    units = [cat.get('r1', row[0]) for row in cat.db.execute('SELECT unit_id FROM units')]
    uris = {item['text']: SCOPE + item['unit_id'] + '/' + item['unit_id'] + '.md' for item in units}
    for item in units:
        cat.db.execute('UPDATE units SET uri=? WHERE revision=? AND unit_id=?',
                       (uris[item['text']], 'r1', item['unit_id']))
    entered, cancelled = asyncio.Event(), asyncio.Event()

    class Reader:
        scope = SCOPE

        async def find(self, query):
            if 'beta' in query:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            await entered.wait()
            return [{'uri': uris['alpha']}]

    def broken_observer(row):
        raise OSError('journal unavailable')

    try:
        with pytest.raises(OSError, match='journal unavailable'):
            await asyncio.wait_for(viking_index.validate_index(cat, Reader(), 'r1', 2, broken_observer), 1)
        assert cancelled.is_set()
    finally:
        cat.db.close()


@pytest.mark.asyncio
async def test_benchmark_journal_failure_cancels_sibling_write_before_close(tmp_path, monkeypatch):
    args = benchmark_args(tmp_path)
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text('\n'.join(json.dumps(row) for row in [
        {'_id': 'a', 'title': 'A', 'text': 'alpha'}, {'_id': 'b', 'title': 'B', 'text': 'beta'}]) + '\n')
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest.update(corpus_count=2, corpus_sha256=file_hash(corpus))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    args.concurrency = 2
    entered, cancelled = asyncio.Event(), asyncio.Event()
    writes = []
    journals = []
    original_journal = index_openviking.Journal

    class BrokenJournal(original_journal):
        def __init__(self, path):
            super().__init__(path)
            journals.append(self)

        def record(self, row):
            if row['type'] == 'unit':
                raise OSError('journal unavailable')
            super().record(row)

    async def handler(request):
        if request.url.path.endswith('temp_upload'):
            return httpx.Response(200, json={'status': 'ok', 'result': {'temp_file_id': 'tmp'}})
        body = json.loads(request.content)
        writes.append(body['to'])
        if len(writes) == 1:
            await entered.wait()
            return httpx.Response(200, json={'status': 'ok', 'result': {'root_uri': body['to']}})
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def synthetic_reconcile(*args):
        return {'state': 'content_verified'}  # Only worker-drain behavior is under test.

    instances, _ = install_adapter(monkeypatch, index_openviking, handler)
    original_client = httpx.AsyncClient

    def health_client(*args, **kwargs):
        kwargs.setdefault('transport', httpx.MockTransport(lambda request: httpx.Response(200, json={'status': 'ok'})))
        return original_client(*args, **kwargs)

    monkeypatch.setattr(index_openviking, 'Journal', BrokenJournal)
    monkeypatch.setattr(index_openviking, 'reconcile_one', synthetic_reconcile)
    monkeypatch.setattr(httpx, 'AsyncClient', health_client)
    with pytest.raises(OSError, match='journal unavailable'):
        await asyncio.wait_for(index_openviking.execute(args), 1)
    assert cancelled.is_set() and instances[0]._state == 'closed'
    assert len(writes) == 2 and len(set(writes)) == 2
    assert journals[0].stream.closed
    cat = Catalog(args.db)
    assert [row[0] for row in cat.db.execute('SELECT state FROM viking_journal')] == ['delivery_unknown'] * 2
    cat.db.close()
