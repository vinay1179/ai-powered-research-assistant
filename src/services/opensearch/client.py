import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError
from src.config import Settings, get_settings

from .index_config import ARXIV_PAPERS_INDEX, ARXIV_PAPERS_MAPPING
from .index_config_hybrid import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """
    Client for OpenSearch operations including index management and search.

    This client provides methods for creating indices, indexing papers,
    searching with BM25 scoring, and managing OpenSearch cluster operations.
    """

    def __init__(self, host: str = "http://localhost:9200", settings: Optional[Settings] = None):
        """Initialize OpenSearch client.

        :param host: OpenSearch cluster endpoint URL
        :param settings: Application settings instance (uses default if None)
        :type host: str
        :type settings: Optional[Settings]
        """
        self.host = host
        self.settings = settings or get_settings()
        self.index_name = self.settings.opensearch.index_name
        self.client = OpenSearch(
            hosts=[host],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        # Use configured index name, fall back to constant if not set
        self.index_name = self.settings.opensearch.index_name or ARXIV_PAPERS_INDEX
        self.chunk_index_name = f"{self.index_name}-{self.settings.opensearch.chunk_index_suffix}"
        logger.info(f"OpenSearch client initialized with host: {host}")

    def create_index(self, force: bool = False) -> bool:
        """Create the arxiv-papers index with proper mappings.

        :param force: If True, delete existing index before creating
        :type force: bool
        :returns: True if index was created, False if it already exists
        :rtype: bool
        """
        try:
            # Check if index exists
            if self.client.indices.exists(index=self.index_name):
                if force:
                    logger.info(f"Deleting existing index: {self.index_name}")
                    self.client.indices.delete(index=self.index_name)
                else:
                    logger.info(f"Index {self.index_name} already exists")
                    return False

            # Create index with mappings
            response = self.client.indices.create(index=self.index_name, body=ARXIV_PAPERS_MAPPING)

            if response.get("acknowledged"):
                logger.info(f"Successfully created index: {self.index_name}")
                return True
            else:
                logger.error(f"Failed to create index: {response}")
                return False

        except RequestError as e:
            logger.error(f"Error creating index: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating index: {e}")
            return False

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """Setup hybrid chunk index and RRF pipeline."""
        results = {}
        results["hybrid_index"] = self._create_chunk_index(force)
        results["rrf_pipeline"] = self._create_rrf_pipeline(force)
        return results

    def _create_chunk_index(self, force: bool = False) -> bool:
        """Create hybrid chunk index for vector + BM25 search."""
        try:
            if force and self.client.indices.exists(index=self.chunk_index_name):
                self.client.indices.delete(index=self.chunk_index_name)
                logger.info(f"Deleted existing hybrid index: {self.chunk_index_name}")

            if not self.client.indices.exists(index=self.chunk_index_name):
                self.client.indices.create(index=self.chunk_index_name, body=ARXIV_PAPERS_CHUNKS_MAPPING)
                logger.info(f"Created hybrid index: {self.chunk_index_name}")
                return True

            logger.info(f"Hybrid index already exists: {self.chunk_index_name}")
            return False

        except Exception as e:
            logger.error(f"Error creating hybrid index: {e}")
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """Create RRF search pipeline for native hybrid search."""
        try:
            pipeline_id = self.settings.opensearch.rrf_pipeline_name or HYBRID_RRF_PIPELINE["id"]

            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info(f"Deleted existing RRF pipeline: {pipeline_id}")
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info(f"RRF pipeline already exists: {pipeline_id}")
                return False
            except Exception:
                pass

            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }
            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body)
            logger.info(f"Created RRF search pipeline: {pipeline_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating RRF pipeline: {e}")
            raise

    def index_paper(self, paper_data: Dict[str, Any]) -> bool:
        """Index a single paper document.

        :param paper_data: Paper data to index
        :type paper_data: Dict[str, Any]
        :returns: True if successful, False otherwise
        :rtype: bool
        """
        try:
            # Ensure required fields
            if "arxiv_id" not in paper_data:
                logger.error("Missing arxiv_id in paper data")
                return False

            # Add timestamps if not present
            if "created_at" not in paper_data:
                paper_data["created_at"] = datetime.now(timezone.utc).isoformat()
            if "updated_at" not in paper_data:
                paper_data["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Convert authors list to string if needed
            if isinstance(paper_data.get("authors"), list):
                paper_data["authors"] = ", ".join(paper_data["authors"])

            # Index the document
            response = self.client.index(
                index=self.index_name,
                id=paper_data["arxiv_id"],
                body=paper_data,
                refresh=True,  # Make it immediately searchable
            )

            if response.get("result") in ["created", "updated"]:
                logger.debug(f"Indexed paper: {paper_data['arxiv_id']}")
                return True
            else:
                logger.error(f"Failed to index paper: {response}")
                return False

        except Exception as e:
            logger.error(f"Error indexing paper {paper_data.get('arxiv_id', 'unknown')}: {e}")
            return False

    def bulk_index_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple papers.

        :param papers: List of paper data to index
        :type papers: List[Dict[str, Any]]
        :returns: Dictionary with counts of successful and failed indexing
        :rtype: Dict[str, int]
        """
        results = {"success": 0, "failed": 0}

        for paper in papers:
            if self.index_paper(paper):
                results["success"] += 1
            else:
                results["failed"] += 1

        logger.info(f"Bulk indexing complete: {results['success']} successful, {results['failed']} failed")
        return results

    def search_papers(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
    ) -> Dict[str, Any]:
        """Search papers using BM25 scoring with query builder.

        :param query: Search query text
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param fields: List of fields to search in (default: title, abstract, authors)
        :param categories: Filter by categories
        :param track_total_hits: Whether to track total hits accurately
        :param latest_papers: Sort by publication date instead of relevance
        :type query: str
        :type size: int
        :type from_: int
        :type fields: Optional[List[str]]
        :type categories: Optional[List[str]]
        :type track_total_hits: bool
        :type latest_papers: bool
        :returns: Search results with hits and metadata
        :rtype: Dict[str, Any]
        """
        try:
            # Build query using query builder
            query_builder = QueryBuilder(
                query=query,
                size=size,
                from_=from_,
                fields=fields,
                categories=categories,
                track_total_hits=track_total_hits,
                latest_papers=latest_papers,
                search_chunks=False,
            )

            search_body = query_builder.build()

            # Execute search
            response = self.client.search(index=self.index_name, body=search_body)

            # Format results
            results = {"total": response["hits"]["total"]["value"], "hits": []}

            for hit in response["hits"]["hits"]:
                paper = hit["_source"]
                paper["score"] = hit["_score"]
                if "highlight" in hit:
                    paper["highlights"] = hit["highlight"]
                results["hits"].append(paper)

            logger.info(f"Search for '{query}' returned {results['total']} results")
            return results

        except NotFoundError:
            logger.error(f"Index {self.index_name} not found")
            return {"total": 0, "hits": [], "error": "Index not found"}
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"total": 0, "hits": [], "error": str(e)}

    def search_chunks_vector(
        self, query_embedding: List[float], size: int = 10, categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Pure vector search on chunks."""
        try:
            filter_clause = []
            if categories:
                filter_clause.append({"terms": {"categories": categories}})

            search_body = {
                "size": size,
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": size}}},
                "_source": {"excludes": ["embedding"]},
            }

            if filter_clause:
                search_body["query"] = {"bool": {"must": [search_body["query"]], "filter": filter_clause}}

            response = self.client.search(index=self.chunk_index_name, body=search_body)

            results = {"total": response["hits"]["total"]["value"], "hits": []}
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["score"] = hit["_score"]
                chunk["chunk_id"] = hit["_id"]
                results["hits"].append(chunk)

            return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"total": 0, "hits": []}

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Unified search method supporting BM25, vector, and hybrid modes."""
        try:
            if not query_embedding or not use_hybrid:
                return self._search_bm25_only(query=query, size=size, from_=from_, categories=categories, latest=latest)

            return self._search_hybrid_native(
                query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
            )

        except Exception as e:
            logger.error(f"Unified search error: {e}")
            return {"total": 0, "hits": []}

    def _search_bm25_only(
        self, query: str, size: int, from_: int, categories: Optional[List[str]], latest: bool
    ) -> Dict[str, Any]:
        """Pure BM25 search implementation on chunks index."""
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest_papers=latest,
            search_chunks=True,
        )
        search_body = builder.build()

        response = self.client.search(index=self.chunk_index_name, body=search_body)

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        logger.info(f"BM25 search for '{query[:50]}...' returned {results['total']} results")
        return results

    def _search_hybrid_native(
        self, query: str, query_embedding: List[float], size: int, categories: Optional[List[str]], min_score: float
    ) -> Dict[str, Any]:
        """Native OpenSearch hybrid search with RRF pipeline."""
        size_multiplier = max(self.settings.opensearch.hybrid_search_size_multiplier, 1)
        expanded_size = size * size_multiplier
        builder = QueryBuilder(
            query=query, size=expanded_size, from_=0, categories=categories, latest_papers=False, search_chunks=True
        )
        bm25_search_body = builder.build()

        bm25_query = bm25_search_body["query"]
        hybrid_query = {"hybrid": {"queries": [bm25_query, {"knn": {"embedding": {"vector": query_embedding, "k": expanded_size}}}]}}

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        pipeline_id = self.settings.opensearch.rrf_pipeline_name or HYBRID_RRF_PIPELINE["id"]
        response = self.client.search(
            index=self.chunk_index_name, body=search_body, params={"search_pipeline": pipeline_id}
        )

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue

            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        results["total"] = len(results["hits"])
        logger.info(f"Native hybrid search for '{query[:50]}...' returned {results['total']} results")
        return results

    def search_chunks_hybrid(
        self,
        query: str,
        query_embedding: List[float],
        size: int = 10,
        categories: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Hybrid search combining BM25 and vector similarity using native RRF."""
        return self._search_hybrid_native(
            query=query, query_embedding=query_embedding, size=size, categories=categories, min_score=min_score
        )

    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """Index a single chunk with its embedding."""
        try:
            chunk_data["embedding"] = embedding
            response = self.client.index(index=self.chunk_index_name, body=chunk_data, refresh=True)
            return response.get("result") in ["created", "updated"]
        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple chunks with embeddings."""
        from opensearchpy import helpers

        try:
            actions = []
            for chunk in chunks:
                chunk_data = chunk["chunk_data"].copy()
                chunk_data["embedding"] = chunk["embedding"]
                actions.append({"_index": self.chunk_index_name, "_source": chunk_data})

            success, failed = helpers.bulk(self.client, actions, refresh=True)
            logger.info(f"Bulk indexed {success} chunks, {len(failed)} failed")
            return {"success": success, "failed": len(failed)}

        except Exception as e:
            logger.error(f"Bulk chunk indexing error: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str) -> bool:
        """Delete all chunks for a specific paper."""
        try:
            response = self.client.delete_by_query(
                index=self.chunk_index_name, body={"query": {"term": {"arxiv_id": arxiv_id}}}, refresh=True
            )

            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for paper {arxiv_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def get_chunks_by_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper."""
        try:
            search_body = {
                "query": {"term": {"arxiv_id": arxiv_id}},
                "size": 1000,
                "sort": [{"chunk_index": "asc"}],
                "_source": {"excludes": ["embedding"]},
            }

            response = self.client.search(index=self.chunk_index_name, body=search_body)

            chunks = []
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["chunk_id"] = hit["_id"]
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index.

        :returns: Dictionary with index statistics
        :rtype: Dict[str, Any]
        """
        try:
            stats = self.client.indices.stats(index=self.index_name)
            count = self.client.count(index=self.index_name)

            return {
                "index_name": self.index_name,
                "document_count": count["count"],
                "size_in_bytes": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"],
                "health": self.client.cluster.health(index=self.index_name)["status"],
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> bool:
        """Check if OpenSearch is healthy and accessible.

        :returns: True if healthy, False otherwise
        :rtype: bool
        """
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_cluster_info(self) -> Optional[Dict[str, Any]]:
        """Get OpenSearch cluster information.

        :returns: Dictionary with cluster info or None if error
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            info = self.client.info()
            return info
        except Exception as e:
            logger.error(f"Error getting cluster info: {e}")
            return None

    def get_cluster_health(self) -> Optional[Dict[str, Any]]:
        """Get detailed cluster health information.

        :returns: Dictionary with cluster health details or None if error
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            health = self.client.cluster.health()
            return health
        except Exception as e:
            logger.error(f"Error getting cluster health: {e}")
            return None

    def get_index_mapping(self) -> Optional[Dict[str, Any]]:
        """Get index mapping (alias for get_mappings for compatibility).

        :returns: Dictionary with index mapping or None if error
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            mappings = self.client.indices.get_mapping(index=self.index_name)
            # Extract just the properties from the nested structure
            if mappings and self.index_name in mappings:
                return mappings[self.index_name].get("mappings", {})
            return {}
        except Exception as e:
            logger.error(f"Error getting index mapping: {e}")
            return None

    def get_index_settings(self) -> Optional[Dict[str, Any]]:
        """Get index settings (alias for get_settings for compatibility).

        :returns: Dictionary with index settings or None if error
        :rtype: Optional[Dict[str, Any]]
        """
        try:
            settings = self.client.indices.get_settings(index=self.index_name)
            # Extract just the settings for this index
            if settings and self.index_name in settings:
                return settings[self.index_name].get("settings", {})
            return {}
        except Exception as e:
            logger.error(f"Error getting index settings: {e}")
            return None
