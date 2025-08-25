"""
Qdrant Vector Store implementation for Graphiti
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning(
        'Qdrant not available. Install with: pip install graphiti-core[qdrant] '
        'or pip install qdrant-client'
    )


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store"""

    url: str = Field(default='localhost', description='Qdrant server URL')
    port: int = Field(default=6333, description='Qdrant server port')
    api_key: str | None = Field(default=None, description='API key for Qdrant Cloud')
    use_memory: bool = Field(default=False, description='Use in-memory storage for testing')
    collection_prefix: str = Field(default='graphiti', description='Prefix for collection names')


class QdrantVectorStore:
    """
    Manages vector storage and retrieval using Qdrant.
    Separate collections for different embedding types.
    """

    def __init__(self, config: QdrantConfig):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                'Qdrant client not available. Install with: pip install graphiti-core[qdrant]'
            )

        self.config = config

        # Initialize Qdrant client
        if config.use_memory:
            self.client = QdrantClient(':memory:')
        else:
            self.client = QdrantClient(
                host=config.url,
                port=config.port,
                api_key=config.api_key,
            )

        # Collection names
        self.entity_collection = f'{config.collection_prefix}_entities'
        self.edge_collection = f'{config.collection_prefix}_edges'
        self.community_collection = f'{config.collection_prefix}_communities'

        # Initialize collections
        self._initialize_collections()

    def _initialize_collections(self):
        """Create collections if they don't exist"""
        collections = [
            (self.entity_collection, 1024),  # Default embedding dimension
            (self.edge_collection, 1024),
            (self.community_collection, 1024),
        ]

        for collection_name, dim in collections:
            try:
                self.client.get_collection(collection_name)
                logger.info(f'Collection {collection_name} already exists')
            except Exception:
                logger.info(f'Creating collection {collection_name}')
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                    ),
                )

    async def upsert_entity_embedding(
        self, uuid: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        """Store entity name embedding in Qdrant"""
        point = PointStruct(
            id=uuid,
            vector=embedding,
            payload={
                'uuid': uuid,
                'name': metadata.get('name', ''),
                'group_id': metadata.get('group_id', ''),
                'labels': metadata.get('labels', []),
                'created_at': metadata.get('created_at', ''),
            },
        )
        self.client.upsert(collection_name=self.entity_collection, points=[point], wait=True)
        logger.debug(f'Stored entity embedding for {uuid}')

    async def upsert_edge_embedding(
        self, uuid: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        """Store edge fact embedding in Qdrant"""
        point = PointStruct(
            id=uuid,
            vector=embedding,
            payload={
                'uuid': uuid,
                'name': metadata.get('name', ''),
                'fact': metadata.get('fact', ''),
                'source_node_uuid': metadata.get('source_node_uuid', ''),
                'target_node_uuid': metadata.get('target_node_uuid', ''),
                'group_id': metadata.get('group_id', ''),
                'created_at': metadata.get('created_at', ''),
                'valid_at': metadata.get('valid_at', ''),
            },
        )

        self.client.upsert(collection_name=self.edge_collection, points=[point], wait=True)
        logger.debug(f'Stored edge embedding for {uuid}')

    async def upsert_community_embedding(
        self, uuid: str, embedding: list[float], metadata: dict[str, Any]
    ) -> None:
        """Store community name embedding in Qdrant"""
        point = PointStruct(
            id=uuid,
            vector=embedding,
            payload={
                'uuid': uuid,
                'name': metadata.get('name', ''),
                'group_id': metadata.get('group_id', ''),
                'summary': metadata.get('summary', ''),
                'created_at': metadata.get('created_at', ''),
            },
        )

        self.client.upsert(collection_name=self.community_collection, points=[point], wait=True)
        logger.debug(f'Stored community embedding for {uuid}')

    async def search_entities(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Search for similar entities.
        Returns list of (uuid, score, metadata) tuples.
        """
        # Build filter
        must_conditions = []
        if group_ids:
            must_conditions.append(
                models.FieldCondition(
                    key='group_id',
                    match=models.MatchAny(any=group_ids),
                )
            )

        if filter_conditions:
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        search_filter = models.Filter(must=must_conditions) if must_conditions else None

        # Use query_points instead of search (deprecated)
        results = self.client.query_points(
            collection_name=self.entity_collection,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=min_score,
        )

        return [(str(hit.id), hit.score, hit.payload or {}) for hit in results.points]

    async def search_edges(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        source_node_uuid: str | None = None,
        target_node_uuid: str | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Search for similar edges.
        Returns list of (uuid, score, metadata) tuples.
        """
        # Build filter
        must_conditions = []
        if group_ids:
            must_conditions.append(
                models.FieldCondition(
                    key='group_id',
                    match=models.MatchAny(any=group_ids),
                )
            )

        # Handle node filtering
        if source_node_uuid and target_node_uuid:
            # Both specified - find edges between these nodes
            must_conditions.extend(
                [
                    models.FieldCondition(
                        key='source_node_uuid',
                        match=models.MatchAny(any=[source_node_uuid, target_node_uuid]),
                    ),
                    models.FieldCondition(
                        key='target_node_uuid',
                        match=models.MatchAny(any=[source_node_uuid, target_node_uuid]),
                    ),
                ]
            )
        elif source_node_uuid:
            # Only source specified
            must_conditions.append(
                models.FieldCondition(
                    key='source_node_uuid',
                    match=models.MatchValue(value=source_node_uuid),
                )
            )
        elif target_node_uuid:
            # Only target specified
            must_conditions.append(
                models.FieldCondition(
                    key='target_node_uuid',
                    match=models.MatchValue(value=target_node_uuid),
                )
            )

        search_filter = models.Filter(must=must_conditions) if must_conditions else None

        # Use query_points instead of search
        results = self.client.query_points(
            collection_name=self.edge_collection,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=min_score,
        )

        return [(str(hit.id), hit.score, hit.payload or {}) for hit in results.points]

    async def search_communities(
        self,
        query_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """
        Search for similar communities.
        Returns list of (uuid, score, metadata) tuples.
        """
        # Build filter
        must_conditions = []
        if group_ids:
            must_conditions.append(
                models.FieldCondition(
                    key='group_id',
                    match=models.MatchAny(any=group_ids),
                )
            )

        search_filter = models.Filter(must=must_conditions) if must_conditions else None

        # Use query_points instead of search
        results = self.client.query_points(
            collection_name=self.community_collection,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=min_score,
        )

        return [(str(hit.id), hit.score, hit.payload or {}) for hit in results.points]

    async def delete_by_uuids(self, collection_name: str, uuids: list[str]) -> None:
        """Delete vectors by UUIDs"""
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=uuids),  # type: ignore
        )

    async def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection"""
        info = self.client.get_collection(collection_name)
        return {
            'vectors_count': info.vectors_count,
            'points_count': info.points_count,
            'indexed_vectors_count': info.indexed_vectors_count,
            'status': info.status,
        }
