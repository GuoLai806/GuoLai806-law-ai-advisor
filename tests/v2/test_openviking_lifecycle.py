"""HTTP contract and shutdown tests; transport timings are not performance data."""
import asyncio
import json

import httpx
import pytest

from evidence_rag.retrieval import OpenViking

SCOPE = 'viking://resources/rag/fixture/r1/'


def found():
    return httpx.Response(200, json={'status': 'ok', 'result': {'resources': [
        {'uri': SCOPE + 'a/a.md'}, {'uri': 'viking://resources/rag/other/r1/a.md'}]}})


class CountTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler=None):
        self.handler = handler
        self.requests = []
        self.closed = 0

    async def handle_async_request(self, request):
        self.requests.append(request)
        return await self.handler(request) if self.handler else found()

    async def aclose(self):
        self.closed += 1


@pytest.mark.asyncio
async def test_owned_client_reused_until_explicit_close():
    transport = CountTransport()
    adapter = OpenViking('http://fixture', SCOPE, transport=transport)
    assert adapter._client is None  # Startup need not open any connection.
    first = await adapter.find('alpha')
    client = adapter._client
    assert await adapter.find('beta') == first == [{'uri': SCOPE + 'a/a.md'}]
    assert adapter._client is client and not client.is_closed
    assert transport.closed == 0
    await adapter.aclose()
    await adapter.aclose()
    assert client.is_closed and transport.closed == 1
    with pytest.raises(RuntimeError, match='closing or closed'):
        await adapter.find('after-close')
    with pytest.raises(RuntimeError, match='closing or closed'):
        await adapter.ingest('a' * 32, 'after-close')
    assert len(transport.requests) == 2


@pytest.mark.asyncio
async def test_borrowed_client_remains_usable_after_adapter_shutdown():
    transport = CountTransport()
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        adapter = OpenViking('http://fixture', SCOPE, client=client)
        await adapter.find('alpha')
        await adapter.aclose()
        assert not client.is_closed and transport.closed == 0
        assert (await client.get('http://fixture/health')).status_code == 200
    assert transport.closed == 1


@pytest.mark.asyncio
async def test_injected_client_can_explicitly_transfer_ownership():
    transport = CountTransport()
    client = httpx.AsyncClient(transport=transport, trust_env=False)
    adapter = OpenViking('http://fixture', SCOPE, client=client, owns_client=True)
    await adapter.aclose()
    assert client.is_closed and transport.closed == 1


@pytest.mark.asyncio
async def test_conflicting_ownership_and_injection_settings_rejected():
    with pytest.raises(ValueError, match='must be owned'):
        OpenViking('http://fixture', SCOPE, owns_client=False)
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(ValueError, match='conflicts'):
            OpenViking('http://fixture', SCOPE, client=client, reuse_client=False)
        with pytest.raises(ValueError, match='conflicts'):
            OpenViking('http://fixture', SCOPE, client=client, transport=CountTransport())
    with pytest.raises(ValueError, match='must be open'):
        OpenViking('http://fixture', SCOPE, client=client)
    async with httpx.AsyncClient() as environment_client:
        with pytest.raises(ValueError, match='trust_env=False'):
            OpenViking('http://fixture', SCOPE, client=environment_client)


@pytest.mark.asyncio
async def test_cancelling_one_find_does_not_cancel_another_or_close_client():
    entered = {name: asyncio.Event() for name in ('cancel', 'keep')}
    release = asyncio.Event()

    async def handler(request):
        name = json.loads(request.content)['query']
        entered[name].set()
        await release.wait()
        return found()

    transport = CountTransport(handler)
    async with OpenViking('http://fixture', SCOPE, transport=transport) as adapter:
        cancelled = asyncio.create_task(adapter.find('cancel'))
        surviving = asyncio.create_task(adapter.find('keep'))
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert not surviving.done() and transport.closed == 0
        release.set()
        assert await surviving == [{'uri': SCOPE + 'a/a.md'}]
        assert not adapter._client.is_closed
    assert transport.closed == 1


@pytest.mark.asyncio
async def test_shutdown_drains_request_and_refuses_new_work_even_if_waiter_cancelled():
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        entered.set()
        await release.wait()
        return found()

    transport = CountTransport(handler)
    adapter = OpenViking('http://fixture', SCOPE, transport=transport)
    request = asyncio.create_task(adapter.find('in-flight'))
    await asyncio.wait_for(entered.wait(), 1)
    shutdown = asyncio.create_task(adapter.aclose())
    await asyncio.sleep(0)
    assert not shutdown.done() and transport.closed == 0
    with pytest.raises(RuntimeError, match='closing or closed'):
        await adapter.find('too-late')
    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    release.set()
    assert await request == [{'uri': SCOPE + 'a/a.md'}]
    await asyncio.wait_for(adapter.aclose(), 1)
    assert transport.closed == 1 and adapter._client.is_closed


@pytest.mark.asyncio
async def test_shutdown_waits_for_upload_and_ingest_as_one_operation():
    uploaded, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        if request.url.path.endswith('temp_upload'):
            uploaded.set()
            await release.wait()
            return httpx.Response(200, json={'status': 'ok', 'result': {'temp_file_id': 'tmp'}})
        body = json.loads(request.content)
        assert body['to'] == SCOPE + 'b' * 32 and body['args'] == {'parse_mode': 'no_split'}
        return httpx.Response(200, json={'status': 'ok', 'result': {'root_uri': body['to']}})

    transport = CountTransport(handler)
    adapter = OpenViking('http://fixture', SCOPE, transport=transport)
    ingest = asyncio.create_task(adapter.ingest('b' * 32, 'complete source'))
    await asyncio.wait_for(uploaded.wait(), 1)
    close = asyncio.create_task(adapter.aclose())
    await asyncio.sleep(0)
    assert transport.closed == 0 and not close.done()
    release.set()
    assert (await ingest)['state'] == 'needs_reconciliation'
    await close
    assert len(transport.requests) == 2 and transport.closed == 1


@pytest.mark.asyncio
async def test_injected_defaults_cannot_relax_deadline_or_follow_redirects():
    async def handler(request):
        assert request.headers['X-API-Key'] == 'adapter-key'
        assert request.extensions['timeout'] == dict(connect=0.25, read=0.25, write=0.25, pool=0.25)
        return httpx.Response(307, headers={'location': 'http://untrusted.example/forward'})

    transport = CountTransport(handler)
    async with httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=True,
                                 timeout=99, headers={'X-API-Key': 'wrong-key'}) as client:
        async with OpenViking('http://fixture', SCOPE, client=client, timeout=0.25,
                              headers={'X-API-Key': 'adapter-key'}) as adapter:
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.find('alpha')
        assert len(transport.requests) == 1 and not client.is_closed


@pytest.mark.asyncio
async def test_per_operation_mode_retains_old_lifecycle():
    transport = CountTransport()
    adapter = OpenViking('http://fixture', SCOPE, transport=transport, reuse_client=False)
    first = await adapter.find('alpha')
    assert transport.closed == 1 and adapter._client is None
    assert await adapter.find('beta') == first
    assert transport.closed == 2
    await adapter.aclose()
    assert transport.closed == 2


@pytest.mark.asyncio
async def test_transport_error_has_no_automatic_retry_and_does_not_leak_active_request():
    async def handler(request):
        raise httpx.ReadTimeout('delivery unknown', request=request)

    transport = CountTransport(handler)
    adapter = OpenViking('http://fixture', SCOPE, transport=transport)
    with pytest.raises(httpx.ReadTimeout):
        await adapter.ingest('a' * 32, 'source')
    await asyncio.wait_for(adapter.aclose(), 1)
    assert len(transport.requests) == 1 and transport.closed == 1


@pytest.mark.asyncio
async def test_real_http_connection_is_reused_and_environment_proxy_ignored(monkeypatch):
    connections = []
    handlers = set()
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('NO_PROXY', '')

    async def serve(reader, writer):
        handlers.add(asyncio.current_task())
        connections.append(writer.get_extra_info('peername'))
        try:
            while True:
                head = await reader.readuntil(b'\r\n\r\n')
                length = next(int(line.split(b':', 1)[1]) for line in head.split(b'\r\n')
                              if line.lower().startswith(b'content-length:'))
                body = json.loads(await reader.readexactly(length))
                assert body['target_uri'] == SCOPE
                raw = json.dumps({'status': 'ok', 'result': {'resources': []}}).encode()
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: ' + str(len(raw)).encode()
                             + b'\r\nContent-Type: application/json\r\n\r\n' + raw)
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    server = await asyncio.start_server(serve, '127.0.0.1', 0)
    base = 'http://127.0.0.1:' + str(server.sockets[0].getsockname()[1])
    try:
        async with OpenViking(base, SCOPE) as shared:
            assert await shared.find('one') == await shared.find('two') == []
            assert len(connections) == 1
        async with OpenViking(base, SCOPE, reuse_client=False) as transient:
            assert await transient.find('one') == await transient.find('two') == []
            assert len(connections) == 3
    finally:
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 1)


@pytest.mark.asyncio
async def test_real_socket_timeout_is_enforced_and_shutdown_can_finish():
    entered, release = asyncio.Event(), asyncio.Event()
    handlers = set()

    async def hang(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            await reader.readuntil(b'\r\n\r\n')
            entered.set()
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    server = await asyncio.start_server(hang, '127.0.0.1', 0)
    base = 'http://127.0.0.1:' + str(server.sockets[0].getsockname()[1])
    adapter = OpenViking(base, SCOPE, timeout=0.03)
    try:
        with pytest.raises(httpx.ReadTimeout):
            await asyncio.wait_for(adapter.find('hang'), 1)
        assert entered.is_set()
        await asyncio.wait_for(adapter.aclose(), 1)
        assert adapter._client.is_closed
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 1)
