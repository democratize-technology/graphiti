"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import TYPE_CHECKING, Any

from .client import VectorCollection, VectorSearchResult, VectorStore
from .config import QdrantConfig

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

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


if QDRANT_AVAILABLE:

    class QdrantVectorStore(VectorStore):
        """
        Qdrant implementation of VectorStore.

        Manages vector storage and retrieval using Qdrant with separate collections
        for different embedding types (entities, edges, communities).
        """

        def __init__(self, config: QdrantConfig):
            if not QDRANT_AVAILABLE:
                raise ImportError(
                    'Qdrant client not available. Install with: pip install graphiti-core[qdrant]'
                )

            self.config = config
            self._client = None
            self._collection_names = {
                VectorCollection.ENTITIES: config.get_collection_name('entities'),
                VectorCollection.EDGES: config.get_collection_name('edges'),
                VectorCollection.COMMUNITIES: config.get_collection_name('communities'),
            }

        @property
        def client(self) -> 'QdrantClient':
            """Lazy initialization of Qdrant client"""
            if self._client is None:
                if self.config.use_memory:
                    self._client = QdrantClient(':memory:')
                    logger.info('Initialized in-memory Qdrant client')
                else:
                    self._client = QdrantClient(
                        host=self.config.url,
                        port=self.config.port,
                        api_key=self.config.api_key,
                        timeout=int(self.config.timeout),
                    )
                    logger.info(f'Initialized Qdrant client: {self.config.url}:{self.config.port}')
            return self._client

        async def initialize_collections(self, embedding_dim: int = 1024) -> None:
            """Create collections if they don't exist"""
            if not (1 <= embedding_dim <= 4096):
                raise ValueError('Embedding dimension must be between 1 and 4096')
                
            for _collection_enum, collection_name in self._collection_names.items():
                try:
                    self.client.get_collection(collection_name)
                    logger.info(f'Collection {collection_name} already exists')
                except models.UnexpectedResponse as e:
                    if 'not found' in str(e).lower():
                        logger.info(f'Creating collection {collection_name}')
                        try:
                            self.client.create_collection(
                                collection_name=collection_name,
                                vectors_config=VectorParams(
                                    size=embedding_dim,
                                    distance=Distance.COSINE,
                                ),
                            )
                        except Exception as create_error:
                            logger.error(f'Failed to create collection {collection_name}: {create_error}')
                            raise ConnectionError(f'Unable to create Qdrant collection: {create_error}') from create_error
                    else:
                        logger.error(f'Unexpected error accessing collection {collection_name}: {e}')
                        raise ConnectionError(f'Unable to access Qdrant collection: {e}') from e
                except Exception as e:
                    logger.error(f'Failed to check collection {collection_name}: {e}')
                    raise ConnectionError(f'Unable to connect to Qdrant: {e}') from e

        async def upsert_vectors(
            self,
            collection: VectorCollection,
            vectors: list[tuple[str, list[float], dict[str, Any]]],
        ) -> None:
            """Store or update vectors in the specified collection"""
            if not vectors:
                return
                
            # Security: Limit batch size to prevent resource exhaustion
            MAX_BATCH_SIZE = 1000
            if len(vectors) > MAX_BATCH_SIZE:
                raise ValueError(f'Batch size {len(vectors)} exceeds maximum {MAX_BATCH_SIZE}')
            
            collection_name = self._collection_names[collection]
            
            points = []
            for i, (uuid, embedding, metadata) in enumerate(vectors):
                # Input validation
                if not uuid or not isinstance(uuid, str):
                    raise ValueError(f'Invalid UUID at index {i}: must be non-empty string')
                if not embedding or not isinstance(embedding, list):
                    raise ValueError(f'Invalid embedding at index {i}: must be non-empty list')
                if len(embedding) > 4096:
                    raise ValueError(f'Embedding at index {i} too large: {len(embedding)} dimensions')
                if not isinstance(metadata, dict):
                    raise ValueError(f'Invalid metadata at index {i}: must be dictionary')
                    
                point = PointStruct(
                    id=uuid,
                    vector=embedding,
                    payload=metadata,
                )
                points.append(point)

            try:
                self.client.upsert(collection_name=collection_name, points=points, wait=True)
                logger.debug(f'Stored {len(points)} vectors in {collection_name}')
            except Exception as e:
                logger.error(f'Failed to upsert vectors to {collection_name}: {e}')
                raise ConnectionError(f'Unable to store vectors in Qdrant: {e}') from e

        async def search_vectors(
            self,
            collection: VectorCollection,
            query_vector: list[float],
            limit: int = 10,
            min_score: float = 0.0,
            filter_conditions: dict[str, Any] | None = None,
            group_ids: list[str] | None = None,
        ) -> list[VectorSearchResult]:
            """Search for similar vectors in the specified collection"""
            # Input validation
            if not query_vector or not isinstance(query_vector, list):
                raise ValueError('Query vector must be a non-empty list')
            if len(query_vector) > 4096:
                raise ValueError(f'Query vector too large: {len(query_vector)} dimensions')
            if not (1 <= limit <= 1000):
                raise ValueError('Limit must be between 1 and 1000')
            if not (0.0 <= min_score <= 1.0):
                raise ValueError('Min score must be between 0.0 and 1.0')
                
            collection_name = self._collection_names[collection]

            # Build filter conditions
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
                    if isinstance(value, list):
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value),
                            )
                        )
                    else:
                        must_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value),
                            )
                        )

            search_filter = models.Filter(must=must_conditions) if must_conditions else None

            # Perform search using query_points
            try:
                results = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    score_threshold=min_score,
                )
            except Exception as e:
                logger.error(f'Failed to search vectors in {collection_name}: {e}')
                raise ConnectionError(f'Unable to search vectors in Qdrant: {e}') from e

            # Convert to VectorSearchResult objects
            search_results = []
            for hit in results.points:
                result = VectorSearchResult(
                    uuid=str(hit.id),
                    score=hit.score,
                    metadata=hit.payload or {},
                )
                search_results.append(result)

            logger.debug(f'Found {len(search_results)} results in {collection_name}')
            return search_results

        async def delete_vectors(
            self,
            collection: VectorCollection,
            uuids: list[str],
        ) -> None:
            """Delete vectors by UUIDs from the specified collection"""
            collection_name = self._collection_names[collection]

            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=uuids),  # type: ignore
            )
            logger.debug(f'Deleted {len(uuids)} vectors from {collection_name}')

        async def get_collection_info(self, collection: VectorCollection) -> dict[str, Any]:
            """Get information about a collection"""
            collection_name = self._collection_names[collection]

            try:
                info = self.client.get_collection(collection_name)
                return {
                    'name': collection_name,
                    'vectors_count': info.vectors_count,
                    'points_count': info.points_count,
                    'indexed_vectors_count': info.indexed_vectors_count,
                    'status': str(info.status),
                    'config': {
                        'distance': (
                            str(info.config.params.vectors.distance)  # type: ignore
                            if info.config
                            else None
                        ),
                        'vector_size': info.config.params.vectors.size if info.config else None,  # type: ignore
                    },
                }
            except Exception as e:
                logger.error(f'Error getting info for collection {collection_name}: {e}')
                raise

        # Legacy methods for backward compatibility with existing qdrant_store.py
        async def upsert_entity_embedding(
            self, uuid: str, embedding: list[float], metadata: dict[str, Any]
        ) -> None:
            """Store entity name embedding in Qdrant (legacy method)"""
            await self.upsert_vectors(VectorCollection.ENTITIES, [(uuid, embedding, metadata)])

        async def upsert_edge_embedding(
            self, uuid: str, embedding: list[float], metadata: dict[str, Any]
        ) -> None:
            """Store edge fact embedding in Qdrant (legacy method)"""
            await self.upsert_vectors(VectorCollection.EDGES, [(uuid, embedding, metadata)])

        async def upsert_community_embedding(
            self, uuid: str, embedding: list[float], metadata: dict[str, Any]
        ) -> None:
            """Store community name embedding in Qdrant (legacy method)"""
            await self.upsert_vectors(VectorCollection.COMMUNITIES, [(uuid, embedding, metadata)])

        async def search_entities(
            self,
            query_vector: list[float],
            group_ids: list[str] | None = None,
            limit: int = 10,
            min_score: float = 0.0,
            filter_conditions: dict[str, Any] | None = None,
        ) -> list[tuple[str, float, dict[str, Any]]]:
            """Search for similar entities (legacy method)"""
            results = await self.search_vectors(
                VectorCollection.ENTITIES,
                query_vector,
                limit=limit,
                min_score=min_score,
                filter_conditions=filter_conditions,
                group_ids=group_ids,
            )
            # Convert to legacy format
            return [(result.uuid, result.score, result.metadata) for result in results]

        async def search_edges(
            self,
            query_vector: list[float],
            group_ids: list[str] | None = None,
            source_node_uuid: str | None = None,
            target_node_uuid: str | None = None,
            limit: int = 10,
            min_score: float = 0.0,
        ) -> list[tuple[str, float, dict[str, Any]]]:
            """Search for similar edges (legacy method)"""
            filter_conditions = {}

            # Handle node filtering
            if source_node_uuid and target_node_uuid:
                # Both specified - find edges between these nodes
                filter_conditions['source_node_uuid'] = [source_node_uuid, target_node_uuid]
                filter_conditions['target_node_uuid'] = [source_node_uuid, target_node_uuid]
            elif source_node_uuid:
                filter_conditions['source_node_uuid'] = source_node_uuid
            elif target_node_uuid:
                filter_conditions['target_node_uuid'] = target_node_uuid

            results = await self.search_vectors(
                VectorCollection.EDGES,
                query_vector,
                limit=limit,
                min_score=min_score,
                filter_conditions=filter_conditions,
                group_ids=group_ids,
            )
            # Convert to legacy format
            return [(result.uuid, result.score, result.metadata) for result in results]

        async def search_communities(
            self,
            query_vector: list[float],
            group_ids: list[str] | None = None,
            limit: int = 10,
            min_score: float = 0.0,
        ) -> list[tuple[str, float, dict[str, Any]]]:
            """Search for similar communities (legacy method)"""
            results = await self.search_vectors(
                VectorCollection.COMMUNITIES,
                query_vector,
                limit=limit,
                min_score=min_score,
                group_ids=group_ids,
            )
            # Convert to legacy format
            return [(result.uuid, result.score, result.metadata) for result in results]

        async def delete_by_uuids(self, collection_name: str, uuids: list[str]) -> None:
            """Delete vectors by UUIDs (legacy method)"""
            # Map collection name to enum
            collection_map = {v: k for k, v in self._collection_names.items()}

            if collection_name in collection_map:
                await self.delete_vectors(collection_map[collection_name], uuids)
            else:
                # Fallback for direct collection name usage
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=uuids),  # type: ignore
                )
                logger.debug(f'Deleted {len(uuids)} vectors from {collection_name} (legacy)')

    async def health_check(self) -> bool:
        """Check if Qdrant connection is healthy"""
        try:
            # Simple health check by trying to list collections
            self.client.get_collections()
            return True
        except Exception as e:
            logger.warning(f'Qdrant health check failed: {e}')
            return False

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            logger.debug('Closed Qdrant vector store connection')
        except Exception as e:
            logger.warning(f'Error during Qdrant connection close: {e}')
