"""
性能监控工具
用于记录和输出各个操作的耗时
"""

import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def timed(operation_name: str):
    """
    装饰器：记录函数执行时间

    Args:
        operation_name: 操作名称，用于日志输出
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⏱️  [{operation_name}] 耗时: {elapsed:.1f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                logger.error(f"⏱️  [{operation_name}] 出错 ({elapsed:.1f}ms): {e}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⏱️  [{operation_name}] 耗时: {elapsed:.1f}ms")
                return result
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                logger.error(f"⏱️  [{operation_name}] 出错 ({elapsed:.1f}ms): {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class PerformanceMonitor:
    """性能监控上下文管理器"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"🚀 开始 [{self.operation_name}]")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.time() - self.start_time) * 1000
        if exc_type:
            logger.error(f"⏱️  [{self.operation_name}] 失败 ({elapsed:.1f}ms): {exc_val}")
        else:
            logger.info(f"⏱️  [{self.operation_name}] 耗时: {elapsed:.1f}ms")

    @property
    def elapsed(self) -> float:
        """获取已耗时（毫秒）"""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000


def log_operation_summary(
    operation_name: str,
    duration_ms: float,
    details: dict = None
):
    """
    记录操作总结

    Args:
        operation_name: 操作名称
        duration_ms: 总耗时
        details: 详细信息（各步骤耗时）
    """
    msg = f"📊 [{operation_name}] 总耗时: {duration_ms:.1f}ms"
    if details:
        parts = []
        for key, value in details.items():
            if isinstance(value, float):
                parts.append(f"{key}: {value:.1f}ms")
            else:
                parts.append(f"{key}: {value}")
        msg += " | " + " | ".join(parts)
    logger.info(msg)


# 确保导入asyncio
import asyncio
