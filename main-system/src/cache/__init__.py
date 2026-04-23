"""
缓存模块
提供 Redis 管理、查询缓存、高频问题存储等功能

作者：GuoLai
版本：1.0.0
"""

from .redis_manager import (
    RedisConfig,
    RedisManager,
    QueryCache,
    HighFrequencyQA,
    get_redis_manager,
    get_query_cache,
    get_hfqa_manager
)

__all__ = [
    "RedisConfig",
    "RedisManager",
    "QueryCache",
    "HighFrequencyQA",
    "get_redis_manager",
    "get_query_cache",
    "get_hfqa_manager"
]
