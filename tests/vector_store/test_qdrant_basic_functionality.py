"""
Basic functionality tests for Qdrant integration.

These tests focus on the key functionality without complex mocking.
"""

from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.vector_store.config import QdrantConfig


class TestQdrantBasicFunctionality:
    """Test basic Qdrant functionality."""

    def test_qdrant_available_when_imported(self):
        """Test QDRANT_AVAILABLE flag behavior."""
        # Test when qdrant is available
        with patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True):
            try:
                from graphiti_core.vector_store.qdrant import QdrantVectorStore

                assert QdrantVectorStore is not None
            except ImportError:
                pytest.fail('Should not raise ImportError when QDRANT_AVAILABLE is True')

    def test_qdrant_import_error_when_unavailable(self):
        """Test that ImportError is properly raised when Qdrant is unavailable."""
        with patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', False):
            from graphiti_core.vector_store.qdrant import QdrantVectorStore

            config = QdrantConfig()

            with pytest.raises(ImportError) as exc_info:
                QdrantVectorStore(config)

            assert 'Qdrant client not available' in str(exc_info.value)

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_vector_store_initialization(self, mock_client_class):
        """Test basic vector store initialization."""
        from graphiti_core.vector_store.client import VectorCollection
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = QdrantConfig(collection_prefix='test')
        store = QdrantVectorStore(config)

        # Test basic properties
        assert store.config == config
        assert store._collection_names[VectorCollection.ENTITIES] == 'test_entities'
        assert store._collection_names[VectorCollection.EDGES] == 'test_edges'
        assert store._collection_names[VectorCollection.COMMUNITIES] == 'test_communities'

        # Test lazy client initialization
        assert store._client is None

        # Access client to trigger initialization
        client = store.client
        assert client == mock_client
        assert store._client == mock_client

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_memory_vs_remote_client_creation(self, mock_client_class):
        """Test client creation for memory vs remote configurations."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Test memory configuration
        config_memory = QdrantConfig(use_memory=True)
        store_memory = QdrantVectorStore(config_memory)

        # Trigger client creation
        _ = store_memory.client

        # Should call with ':memory:'
        mock_client_class.assert_called_with(':memory:')

        # Reset mock
        mock_client_class.reset_mock()

        # Test remote configuration
        config_remote = QdrantConfig(
            url='test-host', port=7777, api_key='test-key', use_memory=False, timeout=60.0
        )
        store_remote = QdrantVectorStore(config_remote)

        # Trigger client creation
        _ = store_remote.client

        # Should call with connection parameters
        mock_client_class.assert_called_with(
            host='test-host', port=7777, api_key='test-key', timeout=60
        )

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    @pytest.mark.asyncio
    async def test_basic_crud_operations_structure(self, mock_client_class):
        """Test that CRUD operations have the right structure."""
        from graphiti_core.vector_store.client import VectorCollection
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        # Test that initialize_collections calls the right methods
        mock_client.get_collection.side_effect = Exception('Not found')
        await store.initialize_collections(embedding_dim=512)

        # Should try to get collections first
        assert mock_client.get_collection.called

        # Should create collections when they don't exist
        assert mock_client.create_collection.called

        # Test upsert structure
        test_vectors = [
            ('uuid-1', [0.1, 0.2, 0.3], {'name': 'Test'}),
            ('uuid-2', [0.4, 0.5, 0.6], {'name': 'Test 2'}),
        ]

        await store.upsert_vectors(VectorCollection.ENTITIES, test_vectors)

        # Should call upsert with correct structure
        mock_client.upsert.assert_called()
        call_args = mock_client.upsert.call_args
        assert call_args[1]['collection_name'] == 'graphiti_entities'
        assert call_args[1]['wait'] is True
        assert len(call_args[1]['points']) == 2

        # Test search structure
        mock_result = MagicMock()
        mock_result.points = []
        mock_client.query_points.return_value = mock_result

        await store.search_vectors(VectorCollection.ENTITIES, [0.1, 0.2, 0.3], limit=5)

        # Should call query_points
        mock_client.query_points.assert_called()
        search_call_args = mock_client.query_points.call_args[1]
        assert search_call_args['collection_name'] == 'graphiti_entities'
        assert search_call_args['query'] == [0.1, 0.2, 0.3]
        assert search_call_args['limit'] == 5

        # Test delete structure
        await store.delete_vectors(VectorCollection.ENTITIES, ['uuid-1', 'uuid-2'])

        # Should call delete
        mock_client.delete.assert_called()
        delete_call_args = mock_client.delete.call_args[1]
        assert delete_call_args['collection_name'] == 'graphiti_entities'

    def test_collection_name_generation(self):
        """Test collection name generation with different prefixes."""
        config = QdrantConfig(collection_prefix='custom')

        assert config.get_collection_name('entities') == 'custom_entities'
        assert config.get_collection_name('edges') == 'custom_edges'
        assert config.get_collection_name('communities') == 'custom_communities'
        assert config.get_collection_name('custom_type') == 'custom_custom_type'

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    def test_backwards_compatibility_with_existing_interface(self):
        """Test that the store implements the VectorStore interface."""
        from graphiti_core.vector_store.client import VectorStore
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        config = QdrantConfig()
        store = QdrantVectorStore(config)

        # Should be an instance of VectorStore
        assert isinstance(store, VectorStore)

        # Should have all required methods
        required_methods = [
            'initialize_collections',
            'upsert_vectors',
            'search_vectors',
            'delete_vectors',
            'get_collection_info',
        ]

        for method_name in required_methods:
            assert hasattr(store, method_name)
            assert callable(getattr(store, method_name))

        # Should have convenience methods
        convenience_methods = [
            'upsert_entity_embeddings',
            'search_entity_embeddings',
            'delete_entity_embeddings',
            'upsert_edge_embeddings',
            'search_edge_embeddings',
            'delete_edge_embeddings',
            'upsert_community_embeddings',
            'search_community_embeddings',
            'delete_community_embeddings',
        ]

        for method_name in convenience_methods:
            assert hasattr(store, method_name)
            assert callable(getattr(store, method_name))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
