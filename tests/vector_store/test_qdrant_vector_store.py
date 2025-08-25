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

from unittest.mock import MagicMock, Mock, patch

import pytest

from graphiti_core.vector_store.client import VectorCollection, VectorSearchResult
from graphiti_core.vector_store.config import QdrantConfig


class TestQdrantVectorStore:
    """Test QdrantVectorStore functionality with mocked dependencies."""

    def test_import_error_when_qdrant_unavailable(self):
        """Test ImportError is raised when Qdrant client is not available."""
        with patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', False):
            from graphiti_core.vector_store.qdrant import QdrantVectorStore

            config = QdrantConfig()

            with pytest.raises(ImportError) as exc_info:
                QdrantVectorStore(config)

            assert 'Qdrant client not available' in str(exc_info.value)
            assert 'pip install graphiti-core[qdrant]' in str(exc_info.value)

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_initialization_with_memory(self, mock_qdrant_client_class):
        """Test QdrantVectorStore initialization with in-memory storage."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        config = QdrantConfig(use_memory=True, collection_prefix='test')
        store = QdrantVectorStore(config)

        assert store.config == config
        assert store._client is None  # Lazy initialization
        assert store._collection_names[VectorCollection.ENTITIES] == 'test_entities'
        assert store._collection_names[VectorCollection.EDGES] == 'test_edges'
        assert store._collection_names[VectorCollection.COMMUNITIES] == 'test_communities'

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_initialization_with_remote_server(self, mock_qdrant_client_class):
        """Test QdrantVectorStore initialization with remote server."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        config = QdrantConfig(
            url='remote-host',
            port=9999,
            api_key='test-key',
            use_memory=False,
            collection_prefix='prod',
        )
        store = QdrantVectorStore(config)

        assert store.config == config
        assert store._client is None  # Lazy initialization
        assert store._collection_names[VectorCollection.ENTITIES] == 'prod_entities'

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_client_lazy_initialization_memory(self, mock_qdrant_client_class):
        """Test lazy initialization of client with memory storage."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_client_instance = Mock()
        mock_qdrant_client_class.return_value = mock_client_instance

        config = QdrantConfig(use_memory=True)
        store = QdrantVectorStore(config)

        # Client should not be initialized yet
        assert store._client is None

        # Access client property to trigger initialization
        client = store.client

        # Should initialize with in-memory configuration
        mock_qdrant_client_class.assert_called_once_with(':memory:')
        assert client == mock_client_instance
        assert store._client == mock_client_instance

        # Subsequent calls should return the same instance
        client2 = store.client
        assert client2 == mock_client_instance
        assert mock_qdrant_client_class.call_count == 1

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_client_lazy_initialization_remote(self, mock_qdrant_client_class):
        """Test lazy initialization of client with remote server."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_client_instance = Mock()
        mock_qdrant_client_class.return_value = mock_client_instance

        config = QdrantConfig(
            url='test-host', port=7777, api_key='test-key', use_memory=False, timeout=45.0
        )
        store = QdrantVectorStore(config)

        # Access client property to trigger initialization
        client = store.client

        # Should initialize with remote configuration
        mock_qdrant_client_class.assert_called_once_with(
            host='test-host', port=7777, api_key='test-key', timeout=45
        )
        assert client == mock_client_instance

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_initialize_collections_new(self, mock_qdrant_client_class):
        """Test initializing collections when they don't exist."""
        from qdrant_client.http.models import Distance

        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Create a proper mock client instance
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collection.side_effect = Exception('Collection not found')
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig(collection_prefix='test')
        store = QdrantVectorStore(config)

        await store.initialize_collections(embedding_dim=512)

        # Should attempt to get each collection
        assert mock_qdrant_client.get_collection.call_count == 3

        # Should create each collection
        assert mock_qdrant_client.create_collection.call_count == 3

        # Verify create_collection calls
        create_calls = mock_qdrant_client.create_collection.call_args_list
        collection_names = {call[1]['collection_name'] for call in create_calls}
        expected_names = {'test_entities', 'test_edges', 'test_communities'}
        assert collection_names == expected_names

        # Verify vector configuration
        for call in create_calls:
            vectors_config = call[1]['vectors_config']
            assert vectors_config.size == 512
            assert vectors_config.distance == Distance.COSINE

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_initialize_collections_existing(self, mock_qdrant_client_class):
        """Test initializing collections when they already exist."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Create a proper mock client instance
        mock_qdrant_client = MagicMock()
        mock_collection = Mock()
        mock_qdrant_client.get_collection.return_value = mock_collection
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        await store.initialize_collections()

        # Should check each collection
        assert mock_qdrant_client.get_collection.call_count == 3

        # Should not create any collections
        mock_qdrant_client.create_collection.assert_not_called()

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_upsert_vectors(
        self, mock_qdrant_client_class, mock_qdrant_client, sample_vectors
    ):
        """Test upserting vectors."""

        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig(collection_prefix='test')
        store = QdrantVectorStore(config)

        await store.upsert_vectors(VectorCollection.ENTITIES, sample_vectors)

        # Should call upsert with correct parameters
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args

        assert call_args[1]['collection_name'] == 'test_entities'
        assert call_args[1]['wait'] is True

        points = call_args[1]['points']
        assert len(points) == 3

        # Verify point structure
        for i, (uuid, embedding, metadata) in enumerate(sample_vectors):
            point = points[i]
            assert point.id == uuid
            assert point.vector == embedding
            assert point.payload == metadata

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_search_vectors_basic(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test basic vector search functionality."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Mock search results
        mock_point1 = Mock()
        mock_point1.id = 'result-1'
        mock_point1.score = 0.95
        mock_point1.payload = {'name': 'Entity 1'}

        mock_point2 = Mock()
        mock_point2.id = 'result-2'
        mock_point2.score = 0.85
        mock_point2.payload = {'name': 'Entity 2'}

        mock_result = Mock()
        mock_result.points = [mock_point1, mock_point2]
        mock_qdrant_client.query_points.return_value = mock_result
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig(collection_prefix='test')
        store = QdrantVectorStore(config)

        query_vector = [0.1, 0.2, 0.3]
        results = await store.search_vectors(
            VectorCollection.ENTITIES, query_vector, limit=5, min_score=0.7
        )

        # Verify query_points was called correctly
        mock_qdrant_client.query_points.assert_called_once()
        call_args = mock_qdrant_client.query_points.call_args[1]

        assert call_args['collection_name'] == 'test_entities'
        assert call_args['query'] == query_vector
        assert call_args['limit'] == 5
        assert call_args['score_threshold'] == 0.7
        assert call_args['query_filter'] is None

        # Verify results
        assert len(results) == 2
        assert isinstance(results[0], VectorSearchResult)
        assert results[0].uuid == 'result-1'
        assert results[0].score == 0.95
        assert results[0].metadata == {'name': 'Entity 1'}

        assert results[1].uuid == 'result-2'
        assert results[1].score == 0.85
        assert results[1].metadata == {'name': 'Entity 2'}

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_search_vectors_with_filters(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test vector search with filter conditions."""

        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_result = Mock()
        mock_result.points = []
        mock_qdrant_client.query_points.return_value = mock_result
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        filter_conditions = {'type': 'entity', 'category': ['person', 'organization']}
        group_ids = ['group1', 'group2']

        await store.search_vectors(
            VectorCollection.ENTITIES,
            [0.1, 0.2, 0.3],
            filter_conditions=filter_conditions,
            group_ids=group_ids,
        )

        # Verify filter construction
        call_args = mock_qdrant_client.query_points.call_args[1]
        query_filter = call_args['query_filter']

        assert query_filter is not None
        assert hasattr(query_filter, 'must')
        assert len(query_filter.must) == 3  # group_id + type + category

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_delete_vectors(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test deleting vectors by UUIDs."""

        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig(collection_prefix='test')
        store = QdrantVectorStore(config)

        uuids_to_delete = ['uuid-1', 'uuid-2', 'uuid-3']

        await store.delete_vectors(VectorCollection.EDGES, uuids_to_delete)

        # Verify delete was called correctly
        mock_qdrant_client.delete.assert_called_once()
        call_args = mock_qdrant_client.delete.call_args[1]

        assert call_args['collection_name'] == 'test_edges'
        points_selector = call_args['points_selector']
        assert points_selector.points == uuids_to_delete

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_get_collection_info(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test getting collection information."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 1000
        mock_info.points_count = 1000
        mock_info.indexed_vectors_count = 950
        mock_info.status = 'green'

        mock_config = Mock()
        mock_config.params.vectors.distance = 'Cosine'
        mock_config.params.vectors.size = 1024
        mock_info.config = mock_config

        mock_qdrant_client.get_collection.return_value = mock_info
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig(collection_prefix='test')
        store = QdrantVectorStore(config)

        info = await store.get_collection_info(VectorCollection.COMMUNITIES)

        # Verify get_collection was called
        mock_qdrant_client.get_collection.assert_called_once_with('test_communities')

        # Verify returned information
        assert info['name'] == 'test_communities'
        assert info['vectors_count'] == 1000
        assert info['points_count'] == 1000
        assert info['indexed_vectors_count'] == 950
        assert info['status'] == 'green'
        assert info['config']['distance'] == 'Cosine'
        assert info['config']['vector_size'] == 1024

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_get_collection_info_error(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test get_collection_info error handling."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Mock an error
        mock_qdrant_client.get_collection.side_effect = Exception('Connection error')
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        with pytest.raises(Exception) as exc_info:
            await store.get_collection_info(VectorCollection.ENTITIES)

        assert 'Connection error' in str(exc_info.value)

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_convenience_methods(
        self, mock_qdrant_client_class, mock_qdrant_client, sample_vectors
    ):
        """Test convenience methods for specific collection types."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_qdrant_client_class.return_value = mock_qdrant_client

        # Mock search results
        mock_result = Mock()
        mock_result.points = []
        mock_qdrant_client.query_points.return_value = mock_result

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        # Test entity convenience methods
        await store.upsert_entity_embeddings(sample_vectors)
        mock_qdrant_client.upsert.assert_called()

        await store.search_entity_embeddings([0.1, 0.2, 0.3])
        mock_qdrant_client.query_points.assert_called()

        await store.delete_entity_embeddings(['uuid-1'])
        mock_qdrant_client.delete.assert_called()

        # Reset mocks
        mock_qdrant_client.reset_mock()

        # Test edge convenience methods
        await store.upsert_edge_embeddings(sample_vectors)
        mock_qdrant_client.upsert.assert_called()

        await store.search_edge_embeddings([0.1, 0.2, 0.3])
        mock_qdrant_client.query_points.assert_called()

        await store.delete_edge_embeddings(['uuid-1'])
        mock_qdrant_client.delete.assert_called()

        # Reset mocks
        mock_qdrant_client.reset_mock()

        # Test community convenience methods
        await store.upsert_community_embeddings(sample_vectors)
        mock_qdrant_client.upsert.assert_called()

        await store.search_community_embeddings([0.1, 0.2, 0.3])
        mock_qdrant_client.query_points.assert_called()

        await store.delete_community_embeddings(['uuid-1'])
        mock_qdrant_client.delete.assert_called()

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_legacy_methods(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test legacy methods for backward compatibility."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_result = Mock()
        mock_result.points = [
            Mock(id='uuid-1', score=0.9, payload={'name': 'Entity 1'}),
            Mock(id='uuid-2', score=0.8, payload={'name': 'Entity 2'}),
        ]
        mock_qdrant_client.query_points.return_value = mock_result
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        # Test single upsert methods
        await store.upsert_entity_embedding('uuid-1', [0.1, 0.2], {'name': 'Test'})
        await store.upsert_edge_embedding('edge-1', [0.3, 0.4], {'fact': 'Test fact'})
        await store.upsert_community_embedding('comm-1', [0.5, 0.6], {'summary': 'Test'})

        # Should have called upsert 3 times
        assert mock_qdrant_client.upsert.call_count == 3

        # Test legacy search methods
        entity_results = await store.search_entities([0.1, 0.2, 0.3], limit=5)
        edge_results = await store.search_edges([0.1, 0.2, 0.3], source_node_uuid='src-1')
        community_results = await store.search_communities([0.1, 0.2, 0.3])

        # Should return tuple format for backward compatibility
        assert len(entity_results) == 2
        assert entity_results[0] == ('uuid-1', 0.9, {'name': 'Entity 1'})
        assert entity_results[1] == ('uuid-2', 0.8, {'name': 'Entity 2'})
        assert len(edge_results) == 2  # Should also return results
        assert len(community_results) == 2  # Should also return results

        # Test delete by collection name (legacy)
        await store.delete_by_uuids('graphiti_entities', ['uuid-1', 'uuid-2'])
        mock_qdrant_client.delete.assert_called()

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_search_edges_with_node_filtering(
        self, mock_qdrant_client_class, mock_qdrant_client
    ):
        """Test edge search with source/target node filtering."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_result = Mock()
        mock_result.points = []
        mock_qdrant_client.query_points.return_value = mock_result
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        # Test with both source and target nodes
        await store.search_edges(
            [0.1, 0.2, 0.3], source_node_uuid='src-1', target_node_uuid='tgt-1'
        )

        # Verify filter conditions were set correctly
        call_args = mock_qdrant_client.query_points.call_args[1]
        query_filter = call_args['query_filter']
        assert query_filter is not None

        # Test with only source node
        await store.search_edges([0.1, 0.2, 0.3], source_node_uuid='src-1')

        # Test with only target node
        await store.search_edges([0.1, 0.2, 0.3], target_node_uuid='tgt-1')

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_empty_payload_handling(self, mock_qdrant_client_class, mock_qdrant_client):
        """Test handling of empty or None payloads in search results."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Mock point with None payload
        mock_point = Mock()
        mock_point.id = 'uuid-1'
        mock_point.score = 0.9
        mock_point.payload = None

        mock_result = Mock()
        mock_result.points = [mock_point]
        mock_qdrant_client.query_points.return_value = mock_result
        mock_qdrant_client_class.return_value = mock_qdrant_client

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        # This should not raise an error
        import asyncio

        results = asyncio.run(store.search_vectors(VectorCollection.ENTITIES, [0.1, 0.2, 0.3]))

        assert len(results) == 1
        assert results[0].uuid == 'uuid-1'
        assert results[0].score == 0.9
        assert results[0].metadata == {}  # Should default to empty dict
