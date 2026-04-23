"""
Redis 管理模块
提供 Redis 连接、缓存管理、高频问题存储等功能

作者：GuoLai
版本：1.0.0
"""

import os
import json
import redis
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta


class RedisConfig:
    """Redis 配置类"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None,
        decode_responses: bool = True
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD", None)
        self.decode_responses = decode_responses


class RedisManager:
    """Redis 管理器"""

    def __init__(self, config: Optional[RedisConfig] = None):
        """
        初始化 Redis 管理器

        Args:
            config: Redis 配置对象，为 None 时使用默认配置
        """
        self.config = config or RedisConfig()
        self._client = None
        self._connected = False

    def connect(self) -> bool:
        """
        连接 Redis

        Returns:
            bool: 是否连接成功
        """
        try:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=self.config.decode_responses
            )

            # 测试连接
            self._client.ping()
            self._connected = True
            print(f"[OK] Redis 连接成功: {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            print(f"[ERROR] Redis 连接失败: {e}")
            self._connected = False
            return False

    def get_client(self) -> Optional[redis.Redis]:
        """
        获取 Redis 客户端

        Returns:
            Optional[redis.Redis]: Redis 客户端
        """
        if not self._connected:
            if not self.connect():
                return None
        return self._client

    def is_connected(self) -> bool:
        """
        检查是否已连接

        Returns:
            bool: 是否已连接
        """
        return self._connected and self._client is not None

    def disconnect(self):
        """断开 Redis 连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._connected = False
            print("[INFO] Redis 连接已断开")


class QueryCache:
    """查询结果缓存"""

    def __init__(self, redis_manager: RedisManager):
        """
        初始化查询缓存

        Args:
            redis_manager: Redis 管理器
        """
        self.redis = redis_manager
        self.namespace = "query_cache"
        self.default_ttl = 3600  # 默认 1 小时过期

    def _get_cache_key(self, query: str) -> str:
        """
        生成缓存键

        Args:
            query: 查询字符串

        Returns:
            str: 缓存键
        """
        # 使用 MD5 哈希查询字符串作为键
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"{self.namespace}:{query_hash}"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的查询结果

        Args:
            query: 查询字符串

        Returns:
            Optional[Dict[str, Any]]: 缓存结果，未命中时返回 None
        """
        client = self.redis.get_client()
        if not client:
            return None

        cache_key = self._get_cache_key(query)
        cached_data = client.get(cache_key)

        if cached_data:
            try:
                return json.loads(cached_data)
            except json.JSONDecodeError:
                print(f"[WARN] 缓存数据解析失败: {cache_key}")
                return None

        return None

    def set(
        self,
        query: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """
        缓存查询结果

        Args:
            query: 查询字符串
            result: 查询结果
            ttl: 过期时间（秒），为 None 时使用默认值
        """
        client = self.redis.get_client()
        if not client:
            return

        cache_key = self._get_cache_key(query)
        ttl = ttl or self.default_ttl

        try:
            client.setex(
                cache_key,
                ttl,
                json.dumps(result, ensure_ascii=False)
            )
            print(f"[OK] 查询已缓存: {query[:50]}...")
        except Exception as e:
            print(f"[ERROR] 缓存查询失败: {e}")

    def delete(self, query: str):
        """
        删除缓存

        Args:
            query: 查询字符串
        """
        client = self.redis.get_client()
        if not client:
            return

        cache_key = self._get_cache_key(query)
        client.delete(cache_key)

    def clear(self):
        """清空所有查询缓存"""
        client = self.redis.get_client()
        if not client:
            return

        pattern = f"{self.namespace}:*"
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
            print(f"[OK] 已清空 {len(keys)} 个查询缓存")


class HighFrequencyQA:
    """高频问题管理"""

    def __init__(self, redis_manager: RedisManager):
        """
        初始化高频问题管理器

        Args:
            redis_manager: Redis 管理器
        """
        self.redis = redis_manager
        self.namespace = "hfqa"
        self.default_threshold = 3  # 默认访问次数阈值

    def _get_hfqa_key(self, query: str) -> str:
        """
        生成高频问题键

        Args:
            query: 查询字符串

        Returns:
            str: 高频问题键
        """
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"{self.namespace}:{query_hash}"

    def increment_access_count(self, query: str) -> int:
        """
        增加问题访问计数

        Args:
            query: 查询字符串

        Returns:
            int: 当前访问次数
        """
        client = self.redis.get_client()
        if not client:
            return 0

        key = f"{self.namespace}:access:{query}"
        count = client.incr(key)

        # 设置过期时间（7天）
        if count == 1:
            client.expire(key, 86400 * 7)

        return count

    def get_access_count(self, query: str) -> int:
        """
        获取问题访问次数

        Args:
            query: 查询字符串

        Returns:
            int: 访问次数
        """
        client = self.redis.get_client()
        if not client:
            return 0

        key = f"{self.namespace}:access:{query}"
        count = client.get(key)
        return int(count) if count else 0

    def is_high_frequency(self, query: str, threshold: Optional[int] = None) -> bool:
        """
        判断是否为高频问题

        Args:
            query: 查询字符串
            threshold: 访问次数阈值，为 None 时使用默认值

        Returns:
            bool: 是否为高频问题
        """
        threshold = threshold or self.default_threshold
        count = self.get_access_count(query)
        return count >= threshold

    def set_answer(self, query: str, answer: str, ttl: int = 86400 * 30):
        """
        设置高频问题答案

        Args:
            query: 查询字符串
            answer: 答案
            ttl: 过期时间（秒），默认 30 天
        """
        client = self.redis.get_client()
        if not client:
            return

        hfqa_key = self._get_hfqa_key(query)
        data = {
            "query": query,
            "answer": answer,
            "created_at": datetime.now().isoformat(),
            "access_count": self.get_access_count(query)
        }

        client.setex(hfqa_key, ttl, json.dumps(data, ensure_ascii=False))
        print(f"[OK] 高频问题已缓存: {query[:50]}...")

    def get_answer(self, query: str) -> Optional[str]:
        """
        获取高频问题答案

        Args:
            query: 查询字符串

        Returns:
            Optional[str]: 答案，未找到时返回 None
        """
        client = self.redis.get_client()
        if not client:
            return None

        hfqa_key = self._get_hfqa_key(query)
        cached_data = client.get(hfqa_key)

        if cached_data:
            try:
                data = json.loads(cached_data)
                return data.get("answer")
            except json.JSONDecodeError:
                print(f"[WARN] 高频问题数据解析失败: {hfqa_key}")
                return None

        return None

    def get_all_high_frequency_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取所有高频问题

        Args:
            limit: 最大返回数量

        Returns:
            List[Dict[str, Any]]: 高频问题列表
        """
        client = self.redis.get_client()
        if not client:
            return []

        pattern = f"{self.namespace}:*"
        keys = client.keys(pattern)

        queries = []
        for key in keys:
            if ":access:" in key:
                continue  # 跳过访问计数键

            data = client.get(key)
            if data:
                try:
                    queries.append(json.loads(data))
                except json.JSONDecodeError:
                    continue

        # 按访问次数排序
        queries.sort(key=lambda x: x.get("access_count", 0), reverse=True)
        return queries[:limit]

    def clear_answer(self, query: str):
        """
        清除高频问题答案

        Args:
            query: 查询字符串
        """
        client = self.redis.get_client()
        if not client:
            return

        hfqa_key = self._get_hfqa_key(query)
        client.delete(hfqa_key)

    def clear_all(self):
        """清空所有高频问题"""
        client = self.redis.get_client()
        if not client:
            return

        pattern = f"{self.namespace}:*"
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
            print(f"[OK] 已清空 {len(keys)} 个高频问题")


# 全局实例
_redis_manager = None
_query_cache = None
_hfqa_manager = None


def get_redis_manager() -> RedisManager:
    """
    获取 Redis 管理器全局实例

    Returns:
        RedisManager: Redis 管理器
    """
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager


def get_query_cache() -> QueryCache:
    """
    获取查询缓存全局实例

    Returns:
        QueryCache: 查询缓存
    """
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(get_redis_manager())
    return _query_cache


def get_hfqa_manager() -> HighFrequencyQA:
    """
    获取高频问题管理器全局实例

    Returns:
        HighFrequencyQA: 高频问题管理器
    """
    global _hfqa_manager
    if _hfqa_manager is None:
        _hfqa_manager = HighFrequencyQA(get_redis_manager())
    return _hfqa_manager
