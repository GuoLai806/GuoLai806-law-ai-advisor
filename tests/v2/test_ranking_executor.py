import asyncio
import threading

import pytest
from evidence_rag.retrieval import RankingExecutor

@pytest.mark.asyncio
async def test_timeout_keeps_model_slot_until_worker_finishes():
    executor=RankingExecutor()
    entered=threading.Event()
    finish=threading.Event()
    subsequent=threading.Event()
    def first():
        entered.set()
        finish.wait(2)
        return 'first'
    try:
        with pytest.raises(TimeoutError):
            await executor.run(first,.03)
        assert entered.is_set()
        with pytest.raises(TimeoutError):
            await executor.run(lambda:subsequent.set(),.03)
        assert not subsequent.is_set()
        finish.set()
        assert await executor.run(lambda:'second',1)=='second'
    finally:
        finish.set()

@pytest.mark.asyncio
async def test_failed_model_releases_slot_and_preserves_error():
    executor=RankingExecutor()
    def fail():
        raise ValueError('model failure')
    with pytest.raises(ValueError,match='model failure'):
        await executor.run(fail,1)
    assert await executor.run(lambda:'recovered',1)=='recovered'

@pytest.mark.asyncio
async def test_cancelled_request_does_not_overlap_model_workers():
    executor=RankingExecutor()
    entered=threading.Event()
    finish=threading.Event()
    def first():
        entered.set()
        finish.wait(2)
    task=asyncio.create_task(executor.run(first,1))
    try:
        while not entered.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(TimeoutError):
            await executor.run(lambda:None,.03)
        finish.set()
        assert await executor.run(lambda:42,1)==42
    finally:
        finish.set()
