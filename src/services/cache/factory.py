import logging
from typing import Optional

import redis
from src.config import Settings
from src.services.cache.client import CacheClient

logger = logging.getLogger(__name__)


def make_redis_client(settings: Settings) -> redis.Redis:
    """Create Redis client with connection pooling."""
    redis_settings = settings.redis
    return redis.Redis(
        host=redis_settings.host,
        port=redis_settings.port,
        password=redis_settings.password or None,
        db=redis_settings.db,
        decode_responses=redis_settings.decode_responses,
        socket_timeout=redis_settings.socket_timeout,
        socket_connect_timeout=redis_settings.socket_connect_timeout,
        retry_on_timeout=True,
        retry_on_error=[redis.ConnectionError, redis.TimeoutError],
    )


def make_cache_client(settings: Settings) -> Optional[CacheClient]:
    """Create exact match cache client. Returns None if Redis is unavailable."""
    try:
        redis_client = make_redis_client(settings)
        redis_client.ping()
        logger.info("Connected to Redis at %s:%s", settings.redis.host, settings.redis.port)
        return CacheClient(redis_client, settings.redis)
    except Exception as exc:
        logger.warning("Cache disabled (Redis unavailable): %s", exc)
        return None
