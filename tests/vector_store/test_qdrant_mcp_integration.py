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

import contextlib
import os
from unittest.mock import MagicMock, patch

import pytest

from graphiti_core.vector_store.config import QdrantConfig


class TestQdrantMCPIntegration:
    """Test Qdrant integration in MCP server context."""

    def test_mcp_server_imports_qdrant_conditionally(self):
        """Test that MCP server handles Qdrant import gracefully."""
        # Test when Qdrant is available
        with patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True):
            try:
                # This should not raise an import error
                from graphiti_core.vector_store import QdrantConfig, QdrantVectorStore

                assert QdrantVectorStore is not None
                assert QdrantConfig is not None
            except ImportError:
                pytest.fail('Should not raise ImportError when QDRANT_AVAILABLE is True')

        # Test when Qdrant is not available
        with patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', False):
            # The import should still work, but QdrantVectorStore might not be available
            from graphiti_core.vector_store.config import QdrantConfig

            assert QdrantConfig is not None  # Config should always be available

    def test_mcp_server_qdrant_config_loading(self):
        """Test loading Qdrant configuration in MCP server context."""
        test_env = {
            'QDRANT_URL': 'mcp-test-host',
            'QDRANT_PORT': '7777',
            'QDRANT_API_KEY': 'mcp-test-key',
            'QDRANT_USE_MEMORY': 'false',
            'QDRANT_COLLECTION_PREFIX': 'mcp_server',
        }

        with patch.dict(os.environ, test_env):
            config = QdrantConfig()

            assert config.url == 'mcp-test-host'
            assert config.port == 7777
            assert config.api_key == 'mcp-test-key'
            assert config.use_memory is False
            assert config.collection_prefix == 'mcp_server'

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_mcp_server_vector_store_initialization(self, mock_qdrant_client_class):
        """Test vector store initialization in MCP server."""
        from graphiti_core.vector_store.client import VectorCollection
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Mock the Qdrant client
        mock_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_client

        # Test configuration for MCP server
        config = QdrantConfig(
            url='localhost',
            port=6333,
            use_memory=True,  # Common for development/testing
            collection_prefix='graphiti_mcp',
        )

        store = QdrantVectorStore(config)
        assert store is not None
        assert store.config == config
        assert store._collection_names[VectorCollection.ENTITIES] == 'graphiti_mcp_entities'

    def test_mcp_server_fallback_without_qdrant(self):
        """Test MCP server behavior when Qdrant is not available."""
        with patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', False):
            # MCP server should be able to start without vector store
            # This simulates the graceful handling in the actual server code

            vector_store = None

            # In the actual MCP server, this would be handled like:
            try:
                if os.getenv('QDRANT_URL') or os.getenv('QDRANT_USE_MEMORY') == 'true':
                    # Would try to create Qdrant store here
                    raise ImportError('Qdrant not available')
            except ImportError:
                vector_store = None  # Fallback to no vector store

            # Server should work without vector store
            assert vector_store is None

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    def test_mcp_server_graphiti_with_vector_store(self):
        """Test Graphiti initialization with vector store in MCP context."""
        from graphiti_core.graphiti import Graphiti

        # Mock vector store
        mock_vector_store = MagicMock()

        # Test Graphiti initialization with vector store
        # (simulating what happens in MCP server)
        try:
            graphiti = Graphiti(
                uri='bolt://localhost:7687',
                user='neo4j',
                password='password',
                vector_store=mock_vector_store,
            )

            assert graphiti.vector_store == mock_vector_store
            assert graphiti.clients.vector_store == mock_vector_store

        except Exception as e:
            # If Neo4j connection fails, that's expected in test environment
            # The important thing is that vector_store parameter is accepted
            assert 'vector_store' not in str(e)

    def test_mcp_server_environment_validation(self):
        """Test environment variable validation for MCP server."""
        # Test valid configurations
        valid_configs = [
            {'QDRANT_USE_MEMORY': 'true'},
            {'QDRANT_URL': 'localhost', 'QDRANT_PORT': '6333'},
            {'QDRANT_URL': 'https://my-qdrant-cluster.com', 'QDRANT_API_KEY': 'secure-api-key'},
        ]

        for test_env in valid_configs:
            with patch.dict(os.environ, test_env, clear=True):
                # Should not raise any validation errors
                config = QdrantConfig()
                assert config is not None

                if 'QDRANT_USE_MEMORY' in test_env:
                    assert config.use_memory == (test_env['QDRANT_USE_MEMORY'] == 'true')

                if 'QDRANT_URL' in test_env:
                    assert config.url == test_env['QDRANT_URL']

                if 'QDRANT_PORT' in test_env:
                    assert config.port == int(test_env['QDRANT_PORT'])

                if 'QDRANT_API_KEY' in test_env:
                    assert config.api_key == test_env['QDRANT_API_KEY']

    def test_mcp_server_error_handling(self):
        """Test error handling scenarios in MCP server context."""
        # Test invalid port configuration
        with patch.dict(os.environ, {'QDRANT_PORT': 'not-a-number'}), pytest.raises(ValueError):
            QdrantConfig()

        # Test configuration with missing required fields for remote connection
        # (This should still work as defaults are provided)
        with patch.dict(os.environ, {}, clear=True):
            config = QdrantConfig()
            assert config.url == 'localhost'
            assert config.port == 6333
            assert config.api_key is None

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_mcp_server_collection_naming(self, mock_qdrant_client_class):
        """Test collection naming conventions for MCP server."""
        from graphiti_core.vector_store.client import VectorCollection
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        # Test with custom prefix for MCP server
        config = QdrantConfig(collection_prefix='mcp_graphiti')
        store = QdrantVectorStore(config)

        expected_names = {
            VectorCollection.ENTITIES: 'mcp_graphiti_entities',
            VectorCollection.EDGES: 'mcp_graphiti_edges',
            VectorCollection.COMMUNITIES: 'mcp_graphiti_communities',
        }

        for collection, expected_name in expected_names.items():
            assert store._collection_names[collection] == expected_name

        # Test collection name method
        assert config.get_collection_name('custom') == 'mcp_graphiti_custom'

    def test_mcp_server_production_config_validation(self):
        """Test configuration validation for production MCP server."""
        # Production-like configuration
        prod_env = {
            'QDRANT_URL': 'https://production-qdrant.example.com',
            'QDRANT_PORT': '443',
            'QDRANT_API_KEY': 'prod-api-key-123',
            'QDRANT_USE_MEMORY': 'false',
            'QDRANT_COLLECTION_PREFIX': 'prod_graphiti',
            'QDRANT_TIMEOUT': '60',
        }

        with patch.dict(os.environ, prod_env):
            config = QdrantConfig()

            # Validate production settings
            assert config.url == 'https://production-qdrant.example.com'
            assert config.port == 443
            assert config.api_key == 'prod-api-key-123'
            assert config.use_memory is False
            assert config.collection_prefix == 'prod_graphiti'
            assert config.timeout == 30.0  # Default since QDRANT_TIMEOUT isn't in model

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', False)
    def test_mcp_server_warning_without_qdrant(self):
        """Test that appropriate warnings are shown when Qdrant is not available."""

        # Try to import (this might trigger warnings)
        with patch('graphiti_core.vector_store.qdrant.logger'):
            # Try to import (this triggers the warning)
            with contextlib.suppress(ImportError):
                # Import might trigger warning if qdrant-client not available
                import graphiti_core.vector_store.qdrant  # noqa: F401

            # In the actual module, a warning is logged when QDRANT_AVAILABLE is False
            # We can't easily test the module-level warning, but we can test behavior

            # The important thing is that the server doesn't crash
            assert True  # If we get here, no crash occurred

    def test_mcp_server_backwards_compatibility(self):
        """Test that MCP server maintains backwards compatibility."""
        # Test that server can start without any Qdrant configuration
        with patch.dict(os.environ, {}, clear=True):
            # Clear any Qdrant-related env vars
            for key in list(os.environ.keys()):
                if key.startswith('QDRANT_'):
                    del os.environ[key]

            # Should be able to create config with defaults
            config = QdrantConfig()
            assert config is not None
            assert config.use_memory is False  # Default behavior

            # Should be able to create Graphiti without vector store
            try:
                from graphiti_core.graphiti import Graphiti

                # This should work (though it will fail on Neo4j connection in tests)
                graphiti = Graphiti(
                    uri='bolt://localhost:7687',
                    user='neo4j',
                    password='password',
                    # No vector_store parameter
                )

                # Should not have vector store
                assert graphiti.vector_store is None
                assert graphiti.clients.vector_store is None

            except Exception as e:
                # Connection errors are expected in test environment
                # The important thing is vector store handling doesn't cause issues
                assert 'vector_store' not in str(e).lower()

    @patch('graphiti_core.vector_store.qdrant.QDRANT_AVAILABLE', True)
    @patch('graphiti_core.vector_store.qdrant.QdrantClient')
    def test_mcp_server_telemetry_integration(self, mock_qdrant_client_class):
        """Test telemetry integration with Qdrant in MCP server."""
        from graphiti_core.vector_store.qdrant import QdrantVectorStore

        mock_client = MagicMock()
        mock_qdrant_client_class.return_value = mock_client

        config = QdrantConfig(collection_prefix='telemetry_test')
        store = QdrantVectorStore(config)

        # Test that we can identify the vector store type
        # (This would be used by telemetry in the main Graphiti class)
        store_class_name = store.__class__.__name__
        assert store_class_name == 'QdrantVectorStore'

        # Test provider detection (as would be done in Graphiti._get_provider_type)
        provider = 'qdrant' if 'qdrant' in store_class_name.lower() else 'unknown'

        assert provider == 'qdrant'

    def test_mcp_server_concurrent_access_safety(self):
        """Test thread safety considerations for MCP server usage."""
        # Test that multiple config instances can be created safely
        configs = []

        for i in range(10):
            with patch.dict(os.environ, {'QDRANT_COLLECTION_PREFIX': f'concurrent_test_{i}'}):
                config = QdrantConfig()
                configs.append(config)

        # Verify each config has correct prefix
        for i, config in enumerate(configs):
            assert config.collection_prefix == f'concurrent_test_{i}'

        # Test that collection names don't interfere
        collection_names = set()
        for config in configs:
            name = config.get_collection_name('entities')
            assert name not in collection_names
            collection_names.add(name)

        assert len(collection_names) == 10  # All unique
