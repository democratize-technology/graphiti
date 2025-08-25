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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class VectorCollection(Enum):
    """Supported vector collections for different graph element types"""

    ENTITIES = 'entities'
    EDGES = 'edges'
    COMMUNITIES = 'communities'


class VectorSearchResult:
    """Result from vector similarity search"""

    def __init__(self, uuid: str, score: float, metadata: dict[str, Any]):
        self.uuid = uuid
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f'VectorSearchResult(uuid={self.uuid}, score={self.score})'


class VectorStore(ABC):
    """
    Abstract base class for vector storage implementations.

    Provides a consistent interface for storing and retrieving vector embeddings
    across different collection types (entities, edges, communities).
    """

    @abstractmethod
    async def initialize_collections(self, embedding_dim: int = 1024) -> None:
        """
        Initialize vector collections if they don't exist.

        Args:
            embedding_dim: Dimension of embeddings to store
        """
        pass

    @abstractmethod
    async def upsert_vectors(
        self,
        collection: VectorCollection,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """
        Store or update vectors in the specified collection.

        Args:
            collection: Target collection type
            vectors: List of (uuid, embedding, metadata) tuples
        """
        pass

    @abstractmethod
    async def search_vectors(
        self,
        collection: VectorCollection,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
        group_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors in the specified collection.

        Args:
            collection: Collection to search in
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            min_score: Minimum similarity score threshold
            filter_conditions: Additional metadata filters
            group_ids: Filter by specific group IDs

        Returns:
            List of search results with UUIDs, scores, and metadata
        """
        pass

    @abstractmethod
    async def delete_vectors(
        self,
        collection: VectorCollection,
        uuids: list[str],
    ) -> None:
        """
        Delete vectors by UUIDs from the specified collection.

        Args:
            collection: Collection to delete from
            uuids: List of vector UUIDs to delete
        """
        pass

    @abstractmethod
    async def get_collection_info(self, collection: VectorCollection) -> dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection: Collection to get info for

        Returns:
            Dictionary with collection statistics
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the vector store connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        # Default implementation assumes always healthy
        return True

    async def close(self) -> None:
        """
        Close the vector store connection.
        
        Default implementation does nothing. Subclasses should override
        if they need to perform cleanup operations.
        """
        # Default implementation: no cleanup needed
        return

    # Convenience methods for entity-specific operations
    async def upsert_entity_embeddings(
        self, vectors: list[tuple[str, list[float], dict[str, Any]]]
    ) -> None:
        """Store entity embeddings"""
        await self.upsert_vectors(VectorCollection.ENTITIES, vectors)

    async def search_entity_embeddings(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
        group_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search entity embeddings"""
        return await self.search_vectors(
            VectorCollection.ENTITIES, query_vector, limit, min_score, filter_conditions, group_ids
        )

    async def delete_entity_embeddings(self, uuids: list[str]) -> None:
        """Delete entity embeddings"""
        await self.delete_vectors(VectorCollection.ENTITIES, uuids)

    # Convenience methods for edge-specific operations
    async def upsert_edge_embeddings(
        self, vectors: list[tuple[str, list[float], dict[str, Any]]]
    ) -> None:
        """Store edge embeddings"""
        await self.upsert_vectors(VectorCollection.EDGES, vectors)

    async def search_edge_embeddings(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
        group_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search edge embeddings"""
        return await self.search_vectors(
            VectorCollection.EDGES, query_vector, limit, min_score, filter_conditions, group_ids
        )

    async def delete_edge_embeddings(self, uuids: list[str]) -> None:
        """Delete edge embeddings"""
        await self.delete_vectors(VectorCollection.EDGES, uuids)

    # Convenience methods for community-specific operations
    async def upsert_community_embeddings(
        self, vectors: list[tuple[str, list[float], dict[str, Any]]]
    ) -> None:
        """Store community embeddings"""
        await self.upsert_vectors(VectorCollection.COMMUNITIES, vectors)

    async def search_community_embeddings(
        self,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
        group_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Search community embeddings"""
        return await self.search_vectors(
            VectorCollection.COMMUNITIES,
            query_vector,
            limit,
            min_score,
            filter_conditions,
            group_ids,
        )

    async def delete_community_embeddings(self, uuids: list[str]) -> None:
        """Delete community embeddings"""
        await self.delete_vectors(VectorCollection.COMMUNITIES, uuids)
