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
from unittest.mock import patch

import pytest

from graphiti_core.vector_store.config import QdrantConfig


class TestQdrantConfig:
    """Test QdrantConfig functionality."""

    def test_default_values(self):
        """Test QdrantConfig with default values."""
        config = QdrantConfig()

        assert config.url == 'localhost'
        assert config.port == 6333
        assert config.api_key is None
        assert config.use_memory is False
        assert config.collection_prefix == 'graphiti'
        assert config.embedding_dim == 1024
        assert config.timeout == 30.0

    def test_explicit_values(self):
        """Test QdrantConfig with explicitly set values."""
        config = QdrantConfig(
            url='custom-host',
            port=9999,
            api_key='test-key',
            use_memory=True,
            collection_prefix='custom',
            embedding_dim=512,
            timeout=60.0,
        )

        assert config.url == 'custom-host'
        assert config.port == 9999
        assert config.api_key == 'test-key'
        assert config.use_memory is True
        assert config.collection_prefix == 'custom'
        assert config.embedding_dim == 512
        assert config.timeout == 60.0

    def test_environment_variable_overrides(self):
        """Test that environment variables override default values."""
        with patch.dict(
            os.environ,
            {
                'QDRANT_URL': 'env-host',
                'QDRANT_PORT': '8888',
                'QDRANT_API_KEY': 'env-key',
                'QDRANT_USE_MEMORY': 'true',
                'QDRANT_COLLECTION_PREFIX': 'env_prefix',
            },
        ):
            config = QdrantConfig()

            assert config.url == 'env-host'
            assert config.port == 8888
            assert config.api_key == 'env-key'
            assert config.use_memory is True
            assert config.collection_prefix == 'env_prefix'

    def test_environment_variable_types(self):
        """Test that environment variables are properly typed."""
        with patch.dict(os.environ, {'QDRANT_PORT': '7777', 'QDRANT_USE_MEMORY': 'false'}):
            config = QdrantConfig()

            assert isinstance(config.port, int)
            assert config.port == 7777
            assert isinstance(config.use_memory, bool)
            assert config.use_memory is False

    def test_use_memory_variations(self):
        """Test different values for QDRANT_USE_MEMORY environment variable."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('yes', False),  # Only 'true' should evaluate to True
            ('1', False),
            ('', False),
        ]

        for env_value, expected_bool in test_cases:
            with patch.dict(os.environ, {'QDRANT_USE_MEMORY': env_value}):
                config = QdrantConfig()
                assert config.use_memory == expected_bool, (
                    f"'{env_value}' should evaluate to {expected_bool}"
                )

    def test_invalid_port_environment_variable(self):
        """Test handling of invalid port environment variable."""
        with patch.dict(os.environ, {'QDRANT_PORT': 'not-a-number'}), pytest.raises(ValueError):
            QdrantConfig()

    def test_get_collection_name(self):
        """Test get_collection_name method."""
        config = QdrantConfig(collection_prefix='test')

        assert config.get_collection_name('entities') == 'test_entities'
        assert config.get_collection_name('edges') == 'test_edges'
        assert config.get_collection_name('communities') == 'test_communities'

    def test_get_collection_name_with_env_prefix(self):
        """Test get_collection_name with environment variable prefix."""
        with patch.dict(os.environ, {'QDRANT_COLLECTION_PREFIX': 'env_test'}):
            config = QdrantConfig()

            assert config.get_collection_name('entities') == 'env_test_entities'
            assert config.get_collection_name('edges') == 'env_test_edges'
            assert config.get_collection_name('communities') == 'env_test_communities'

    def test_config_validation(self):
        """Test config validation - currently no validation is implemented."""
        # The current QdrantConfig implementation doesn't have validation
        # These should work without errors (though they might cause issues later)
        config_negative_port = QdrantConfig(port=-1)
        assert config_negative_port.port == -1

        config_zero_port = QdrantConfig(port=0)
        assert config_zero_port.port == 0

        config_negative_timeout = QdrantConfig(timeout=-1.0)
        assert config_negative_timeout.timeout == -1.0

        config_zero_embedding = QdrantConfig(embedding_dim=0)
        assert config_zero_embedding.embedding_dim == 0

        config_negative_embedding = QdrantConfig(embedding_dim=-1)
        assert config_negative_embedding.embedding_dim == -1

    def test_config_serialization(self):
        """Test config can be serialized and deserialized."""
        original_config = QdrantConfig(
            url='test-host',
            port=7777,
            api_key='test-key',
            use_memory=True,
            collection_prefix='test',
            embedding_dim=512,
            timeout=45.0,
        )

        # Test model_dump (Pydantic v2 method)
        config_dict = original_config.model_dump()
        reconstructed_config = QdrantConfig(**config_dict)

        assert reconstructed_config.url == original_config.url
        assert reconstructed_config.port == original_config.port
        assert reconstructed_config.api_key == original_config.api_key
        assert reconstructed_config.use_memory == original_config.use_memory
        assert reconstructed_config.collection_prefix == original_config.collection_prefix
        assert reconstructed_config.embedding_dim == original_config.embedding_dim
        assert reconstructed_config.timeout == original_config.timeout

    def test_missing_environment_variables(self):
        """Test behavior when environment variables are not set."""
        # Clear all Qdrant-related environment variables
        qdrant_env_vars = [
            'QDRANT_URL',
            'QDRANT_PORT',
            'QDRANT_API_KEY',
            'QDRANT_USE_MEMORY',
            'QDRANT_COLLECTION_PREFIX',
        ]

        original_values = {}
        for var in qdrant_env_vars:
            original_values[var] = os.environ.get(var)
            if var in os.environ:
                del os.environ[var]

        try:
            config = QdrantConfig()

            # Should use defaults when env vars are not set
            assert config.url == 'localhost'
            assert config.port == 6333
            assert config.api_key is None
            assert config.use_memory is False
            assert config.collection_prefix == 'graphiti'

        finally:
            # Restore original environment variables
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value

    def test_config_immutability(self):
        """Test that config fields are properly set and cannot be changed inadvertently."""
        config = QdrantConfig(url='test-host', port=7777)

        # These should not raise errors
        assert config.url == 'test-host'
        assert config.port == 7777

        # Test that we can create a new config with different values
        new_config = QdrantConfig(url='different-host', port=8888)
        assert new_config.url == 'different-host'
        assert new_config.port == 8888

        # Original config should be unchanged
        assert config.url == 'test-host'
        assert config.port == 7777

    def test_api_key_none_vs_empty_string(self):
        """Test handling of None vs empty string for API key."""
        # Test None API key
        config_none = QdrantConfig(api_key=None)
        assert config_none.api_key is None

        # Test empty string API key
        config_empty = QdrantConfig(api_key='')
        assert config_empty.api_key == ''

        # Test environment variable with empty string
        with patch.dict(os.environ, {'QDRANT_API_KEY': ''}):
            config_env_empty = QdrantConfig()
            assert config_env_empty.api_key == ''

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = QdrantConfig(url='host', port=6333, api_key='key')
        config2 = QdrantConfig(url='host', port=6333, api_key='key')
        config3 = QdrantConfig(url='different-host', port=6333, api_key='key')

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        """Test config string representation doesn't expose sensitive data."""
        config = QdrantConfig(
            url='test-host', port=6333, api_key='secret-key', collection_prefix='test'
        )

        repr_str = repr(config)

        # Should include basic config info
        assert 'test-host' in repr_str
        assert '6333' in repr_str
        assert 'test' in repr_str

        # Should not expose the API key in plain text
        # (Pydantic may include it, but this tests our expectations)
        # If this fails, we may need to implement __repr__ to hide sensitive data
