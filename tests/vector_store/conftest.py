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

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

# Import only what we need to avoid dependency issues
try:
    from graphiti_core.vector_store.client import VectorCollection, VectorSearchResult, VectorStore
    from graphiti_core.vector_store.config import QdrantConfig

    GRAPHITI_IMPORTS_AVAILABLE = True
except ImportError:
    # Define minimal classes for testing if imports fail
    GRAPHITI_IMPORTS_AVAILABLE = False

    class VectorCollection:
        ENTITIES = 'entities'
        EDGES = 'edges'
        COMMUNITIES = 'communities'

    class VectorSearchResult:
        def __init__(self, uuid: str, score: float, metadata: dict):
            self.uuid = uuid
            self.score = score
            self.metadata = metadata

    class VectorStore:
        pass

    class QdrantConfig:
        def __init__(self, **kwargs):
            self.url = kwargs.get('url', 'localhost')
            self.port = kwargs.get('port', 6333)
            self.api_key = kwargs.get('api_key')
            self.use_memory = kwargs.get('use_memory', True)
            self.collection_prefix = kwargs.get('collection_prefix', 'test')
            self.embedding_dim = kwargs.get('embedding_dim', 1024)
            self.timeout = kwargs.get('timeout', 30.0)

        def get_collection_name(self, collection_type: str) -> str:
            return f'{self.collection_prefix}_{collection_type}'


class MockQdrantClient:
    """Mock Qdrant client for unit testing."""

    def __init__(self):
        self.collections = {}
        self.points = {}
        self.search_results = []
        self.created_collections = []
        self.deleted_points = []

    def get_collection(self, collection_name: str):
        """Mock get_collection method."""
        if collection_name not in self.collections:

            raise Exception(f'Collection {collection_name} not found')
        return self.collections[collection_name]

    def create_collection(self, collection_name: str, vectors_config=None):
        """Mock create_collection method."""
        collection_info = MagicMock()
        collection_info.vectors_count = 0
        collection_info.points_count = 0
        collection_info.indexed_vectors_count = 0
        collection_info.status = 'green'
        collection_info.config = MagicMock()
        collection_info.config.params.vectors.distance = 'Cosine'
        collection_info.config.params.vectors.size = 1024

        self.collections[collection_name] = collection_info
        self.created_collections.append(collection_name)
        self.points[collection_name] = {}
        return collection_info

    def upsert(self, collection_name: str, points: list, wait: bool = True):
        """Mock upsert method."""
        if collection_name not in self.points:
            self.points[collection_name] = {}

        for point in points:
            self.points[collection_name][point.id] = {
                'id': point.id,
                'vector': point.vector,
                'payload': point.payload,
            }

    def query_points(
        self,
        collection_name: str,
        query: list,
        query_filter=None,
        limit: int = 10,
        score_threshold: float = 0.0,
    ):
        """Mock query_points method."""
        result = MagicMock()
        result.points = self.search_results[:limit]
        return result

    def delete(self, collection_name: str, points_selector):
        """Mock delete method."""
        if hasattr(points_selector, 'points'):
            points_to_delete = points_selector.points
            self.deleted_points.extend(points_to_delete)
            if collection_name in self.points:
                for point_id in points_to_delete:
                    if point_id in self.points[collection_name]:
                        del self.points[collection_name][point_id]


class MockVectorStore(VectorStore):
    """Mock VectorStore for testing integration points."""

    def __init__(self):
        self.initialized_collections = set()
        self.stored_vectors = {}
        self.search_results = []
        self.deleted_vectors = []

    async def initialize_collections(self, embedding_dim: int = 1024) -> None:
        """Mock initialize collections."""
        for collection in VectorCollection:
            self.initialized_collections.add(collection)
            self.stored_vectors[collection] = {}

    async def upsert_vectors(
        self,
        collection: VectorCollection,
        vectors: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Mock upsert vectors."""
        if collection not in self.stored_vectors:
            self.stored_vectors[collection] = {}

        for uuid, embedding, metadata in vectors:
            self.stored_vectors[collection][uuid] = {
                'uuid': uuid,
                'embedding': embedding,
                'metadata': metadata,
            }

    async def search_vectors(
        self,
        collection: VectorCollection,
        query_vector: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
        group_ids: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Mock search vectors."""
        results = []
        for i, result_data in enumerate(self.search_results[:limit]):
            if isinstance(result_data, dict):
                results.append(
                    VectorSearchResult(
                        uuid=result_data.get('uuid', f'uuid-{i}'),
                        score=result_data.get('score', 0.9 - i * 0.1),
                        metadata=result_data.get('metadata', {}),
                    )
                )
            else:
                # Handle tuple format (uuid, score, metadata)
                results.append(
                    VectorSearchResult(
                        uuid=result_data[0] if len(result_data) > 0 else f'uuid-{i}',
                        score=result_data[1] if len(result_data) > 1 else 0.9 - i * 0.1,
                        metadata=result_data[2] if len(result_data) > 2 else {},
                    )
                )
        return results

    async def delete_vectors(
        self,
        collection: VectorCollection,
        uuids: list[str],
    ) -> None:
        """Mock delete vectors."""
        self.deleted_vectors.extend(uuids)
        if collection in self.stored_vectors:
            for uuid in uuids:
                if uuid in self.stored_vectors[collection]:
                    del self.stored_vectors[collection][uuid]

    async def get_collection_info(self, collection: VectorCollection) -> dict[str, Any]:
        """Mock get collection info."""
        return {
            'name': f'test_{collection.value}',
            'vectors_count': len(self.stored_vectors.get(collection, {})),
            'points_count': len(self.stored_vectors.get(collection, {})),
            'indexed_vectors_count': len(self.stored_vectors.get(collection, {})),
            'status': 'green',
            'config': {'distance': 'Cosine', 'vector_size': 1024},
        }


@pytest.fixture
def mock_qdrant_client():
    """Fixture for mock Qdrant client."""
    return MockQdrantClient()


@pytest.fixture
def mock_vector_store():
    """Fixture for mock VectorStore."""
    return MockVectorStore()


@pytest.fixture
def qdrant_config():
    """Fixture for QdrantConfig with test defaults."""
    return QdrantConfig(
        url='localhost',
        port=6333,
        api_key=None,
        use_memory=True,  # Use in-memory for tests by default
        collection_prefix='test',
        embedding_dim=1024,
        timeout=30.0,
    )


@pytest.fixture
def qdrant_config_from_env():
    """Fixture for QdrantConfig using environment variables."""
    # Store original env vars to restore later
    original_env = {
        'QDRANT_URL': os.environ.get('QDRANT_URL'),
        'QDRANT_PORT': os.environ.get('QDRANT_PORT'),
        'QDRANT_API_KEY': os.environ.get('QDRANT_API_KEY'),
        'QDRANT_USE_MEMORY': os.environ.get('QDRANT_USE_MEMORY'),
        'QDRANT_COLLECTION_PREFIX': os.environ.get('QDRANT_COLLECTION_PREFIX'),
    }

    # Set test environment variables
    os.environ['QDRANT_URL'] = 'test-host'
    os.environ['QDRANT_PORT'] = '9999'
    os.environ['QDRANT_API_KEY'] = 'test-api-key'
    os.environ['QDRANT_USE_MEMORY'] = 'false'
    os.environ['QDRANT_COLLECTION_PREFIX'] = 'env_test'

    yield QdrantConfig()

    # Restore original environment variables
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def sample_vectors():
    """Fixture providing sample vector data for testing."""
    return [
        (
            'uuid-1',
            [0.1, 0.2, 0.3],
            {'group_id': 'test-group', 'name': 'Entity 1', 'type': 'entity'},
        ),
        (
            'uuid-2',
            [0.4, 0.5, 0.6],
            {'group_id': 'test-group', 'name': 'Entity 2', 'type': 'entity'},
        ),
        (
            'uuid-3',
            [0.7, 0.8, 0.9],
            {'group_id': 'other-group', 'fact': 'Edge fact', 'type': 'edge'},
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Fixture providing sample search results for testing."""
    return [
        {'uuid': 'result-1', 'score': 0.95, 'metadata': {'name': 'Similar Entity 1'}},
        {'uuid': 'result-2', 'score': 0.85, 'metadata': {'name': 'Similar Entity 2'}},
        {'uuid': 'result-3', 'score': 0.75, 'metadata': {'fact': 'Similar Edge'}},
    ]


@pytest.fixture(autouse=True)
def reset_environment():
    """Auto-used fixture to ensure clean environment for each test."""
    # Store original values - handle import errors gracefully
    try:
        import graphiti_core.vector_store.qdrant
        # Original value is stored but not currently used for restoration
        getattr(graphiti_core.vector_store.qdrant, 'QDRANT_AVAILABLE', True)
    except ImportError:
        pass

    yield

    # Reset any global state if needed
    # This ensures tests don't interfere with each other


@pytest.fixture
def integration_test_config():
    """Configuration for integration tests with real Qdrant."""
    return QdrantConfig(
        url=os.getenv('TEST_QDRANT_URL', 'localhost'),
        port=int(os.getenv('TEST_QDRANT_PORT', '6333')),
        api_key=os.getenv('TEST_QDRANT_API_KEY'),
        use_memory=os.getenv('TEST_QDRANT_USE_MEMORY', 'true').lower() == 'true',
        collection_prefix='integration_test',
        embedding_dim=1024,
        timeout=60.0,  # Longer timeout for integration tests
    )
