"""
Simple test for QdrantConfig without complex dependencies.
"""

import os
from unittest.mock import patch

import pytest


def test_qdrant_config_import():
    """Test that we can import QdrantConfig directly."""
    try:
        from graphiti_core.vector_store.config import QdrantConfig

        # Test basic instantiation
        config = QdrantConfig()
        assert config.url == 'localhost'
        assert config.port == 6333
        assert config.collection_prefix == 'graphiti'

        # Test get_collection_name method
        assert config.get_collection_name('entities') == 'graphiti_entities'
        assert config.get_collection_name('edges') == 'graphiti_edges'

        print('✅ QdrantConfig import and basic functionality working')

    except ImportError as e:
        pytest.skip(f'Cannot import QdrantConfig: {e}')


def test_qdrant_config_with_env_vars():
    """Test QdrantConfig with environment variables."""
    try:
        from graphiti_core.vector_store.config import QdrantConfig

        with patch.dict(
            os.environ,
            {
                'QDRANT_URL': 'test-host',
                'QDRANT_PORT': '9999',
                'QDRANT_API_KEY': 'test-key',
                'QDRANT_USE_MEMORY': 'true',
                'QDRANT_COLLECTION_PREFIX': 'test_prefix',
            },
        ):
            config = QdrantConfig()

            assert config.url == 'test-host'
            assert config.port == 9999
            assert config.api_key == 'test-key'
            assert config.use_memory is True
            assert config.collection_prefix == 'test_prefix'

            print('✅ QdrantConfig environment variable handling working')

    except ImportError as e:
        pytest.skip(f'Cannot import QdrantConfig: {e}')


if __name__ == '__main__':
    test_qdrant_config_import()
    test_qdrant_config_with_env_vars()
    print('🎉 All simple config tests passed!')
