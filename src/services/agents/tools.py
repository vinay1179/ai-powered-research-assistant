from typing import List

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient


async def retrieve_papers(
    query: str,
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    top_k: int,
    use_hybrid: bool,
) -> List[dict]:
    query_embedding = await embeddings_client.embed_query(query)
    search_results = opensearch_client.search_unified(
        query=query,
        query_embedding=query_embedding,
        size=top_k,
        use_hybrid=use_hybrid,
    )
    return search_results.get("hits", [])
