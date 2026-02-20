from functools import lru_cache
from typing import Optional

from src.config import Settings, get_settings

from .client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client() -> OpenSearchClient:
    """Factory function to create cached OpenSearch client instance.

    Uses lru_cache to maintain a singleton instance, consistent with
    other service factories in the codebase.

    :returns: Cached instance of the OpenSearch client
    :rtype: OpenSearchClient
    """
    settings = get_settings()
    return OpenSearchClient(host=settings.opensearch.host, settings=settings)


def make_opensearch_client_fresh(settings: Optional[Settings] = None, host: Optional[str] = None) -> OpenSearchClient:
    """Factory function to create a fresh OpenSearch client (not cached)."""
    if settings is None:
        settings = get_settings()

    opensearch_host = host or settings.opensearch.host
    return OpenSearchClient(host=opensearch_host, settings=settings)
