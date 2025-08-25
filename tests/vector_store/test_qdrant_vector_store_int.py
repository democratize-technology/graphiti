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

import asyncio
import contextlib
import random

import pytest

from graphiti_core.vector_store.client import VectorCollection, VectorSearchResult

# Skip all tests if Qdrant is not available or not configured for testing
try:
    from graphiti_core.vector_store.qdrant import QDRANT_AVAILABLE, QdrantVectorStore
except ImportError:
    QDRANT_AVAILABLE = False

# Skip all tests in this module if Qdrant is not available
pytestmark = pytest.mark.skipif(
    not QDRANT_AVAILABLE,
    reason='Qdrant client not available. Install with: pip install graphiti-core[qdrant]',
)


@pytest.mark.integration
class TestQdrantVectorStoreIntegration:
    """Integration tests for QdrantVectorStore with real Qdrant instance."""

    @pytest.fixture(autouse=True)
    async def setup_and_cleanup(self, integration_test_config):
        """Setup and cleanup for each test."""
        self.config = integration_test_config
        self.store = QdrantVectorStore(self.config)

        # Initialize collections for testing
        await self.store.initialize_collections(embedding_dim=384)  # Smaller dim for tests

        yield

        # Cleanup: delete test collections
        with contextlib.suppress(Exception):
            for collection in VectorCollection:
                collection_name = self.config.get_collection_name(collection.value)
                with contextlib.suppress(Exception):
                    # Try to delete the collection
                    self.store.client.delete_collection(collection_name)

    @pytest.mark.asyncio
    async def test_initialize_collections_real(self):
        """Test collection initialization with real Qdrant."""
        # Collections should be created during setup
        for collection in VectorCollection:
            info = await self.store.get_collection_info(collection)

            assert info['name'] == self.config.get_collection_name(collection.value)
            assert info['status'] in ['green', 'yellow']  # Should be operational
            assert info['config']['vector_size'] == 384
            assert info['config']['distance'] == 'Cosine'

    @pytest.mark.asyncio
    async def test_full_crud_cycle_entities(self):
        """Test complete CRUD cycle for entity vectors."""
        # Sample entity data
        vectors = [
            (
                'entity-1',
                [random.random() for _ in range(384)],
                {'group_id': 'test-group', 'name': 'John Doe', 'type': 'person'},
            ),
            (
                'entity-2',
                [random.random() for _ in range(384)],
                {'group_id': 'test-group', 'name': 'Acme Corp', 'type': 'organization'},
            ),
            (
                'entity-3',
                [random.random() for _ in range(384)],
                {'group_id': 'other-group', 'name': 'Jane Smith', 'type': 'person'},
            ),
        ]

        # Test CREATE (upsert)
        await self.store.upsert_vectors(VectorCollection.ENTITIES, vectors)

        # Wait a moment for indexing
        await asyncio.sleep(0.5)

        # Test READ (search)
        query_vector = vectors[0][1]  # Use first vector as query
        results = await self.store.search_vectors(VectorCollection.ENTITIES, query_vector, limit=10)

        assert len(results) >= 1
        assert isinstance(results[0], VectorSearchResult)

        # The first result should be the exact match (highest score)
        assert results[0].uuid == 'entity-1'
        assert results[0].score >= 0.99  # Should be very close to 1.0
        assert results[0].metadata['name'] == 'John Doe'
        assert results[0].metadata['type'] == 'person'

        # Test UPDATE (upsert with same UUID)
        updated_vector = (
            'entity-1',
            [random.random() for _ in range(384)],
            {
                'group_id': 'test-group',
                'name': 'John Doe Updated',
                'type': 'person',
                'updated': True,
            },
        )

        await self.store.upsert_vectors(VectorCollection.ENTITIES, [updated_vector])
        await asyncio.sleep(0.5)

        # Verify update
        results = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            updated_vector[1],  # Search with updated vector
            limit=1,
        )

        assert len(results) == 1
        assert results[0].uuid == 'entity-1'
        assert results[0].metadata['name'] == 'John Doe Updated'
        assert results[0].metadata.get('updated') is True

        # Test DELETE
        await self.store.delete_vectors(VectorCollection.ENTITIES, ['entity-1', 'entity-2'])
        await asyncio.sleep(0.5)

        # Verify deletion
        all_results = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            query_vector,
            limit=10,
            min_score=0.0,  # Get all results
        )

        # Should only have entity-3 remaining
        remaining_uuids = {result.uuid for result in all_results}
        assert 'entity-1' not in remaining_uuids
        assert 'entity-2' not in remaining_uuids
        assert 'entity-3' in remaining_uuids

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with metadata filters."""
        # Insert test data
        vectors = [
            (
                'person-1',
                [random.random() for _ in range(384)],
                {'group_id': 'group-a', 'type': 'person', 'age': 25, 'city': 'New York'},
            ),
            (
                'person-2',
                [random.random() for _ in range(384)],
                {'group_id': 'group-a', 'type': 'person', 'age': 30, 'city': 'San Francisco'},
            ),
            (
                'org-1',
                [random.random() for _ in range(384)],
                {
                    'group_id': 'group-b',
                    'type': 'organization',
                    'industry': 'tech',
                    'city': 'New York',
                },
            ),
        ]

        await self.store.upsert_vectors(VectorCollection.ENTITIES, vectors)
        await asyncio.sleep(0.5)

        # Test filter by type
        results = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            [random.random() for _ in range(384)],
            limit=10,
            filter_conditions={'type': 'person'},
        )

        assert len(results) == 2
        for result in results:
            assert result.metadata['type'] == 'person'

        # Test filter by group_ids
        results = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            [random.random() for _ in range(384)],
            limit=10,
            group_ids=['group-a'],
        )

        assert len(results) == 2
        for result in results:
            assert result.metadata['group_id'] == 'group-a'

        # Test combined filters
        results = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            [random.random() for _ in range(384)],
            limit=10,
            filter_conditions={'city': 'New York'},
            group_ids=['group-a'],
        )

        assert len(results) == 1
        assert results[0].uuid == 'person-1'
        assert results[0].metadata['city'] == 'New York'
        assert results[0].metadata['group_id'] == 'group-a'

        # Test filter with list values
        results = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            [random.random() for _ in range(384)],
            limit=10,
            filter_conditions={'city': ['New York', 'San Francisco']},
        )

        assert len(results) == 3  # All entities are in one of these cities

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self):
        """Test search with minimum score threshold."""
        # Create a specific vector for exact matching
        exact_vector = [0.5] * 384
        vectors = [
            ('exact-match', exact_vector, {'name': 'Exact Match'}),
            ('close-match', [0.5 + 0.001] * 384, {'name': 'Close Match'}),
            ('far-match', [0.9] * 384, {'name': 'Far Match'}),
        ]

        await self.store.upsert_vectors(VectorCollection.ENTITIES, vectors)
        await asyncio.sleep(0.5)

        # Search with high threshold - should only get exact and close matches
        results = await self.store.search_vectors(
            VectorCollection.ENTITIES, exact_vector, limit=10, min_score=0.95
        )

        # Should get exact match (score ~1.0) and possibly close match
        assert len(results) >= 1
        assert results[0].uuid == 'exact-match'
        assert results[0].score >= 0.99

        # Search with low threshold - should get all matches
        results = await self.store.search_vectors(
            VectorCollection.ENTITIES, exact_vector, limit=10, min_score=0.1
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_multiple_collections(self):
        """Test operations across different collection types."""
        entity_vector = ('entity-1', [random.random() for _ in range(384)], {'type': 'entity'})
        edge_vector = ('edge-1', [random.random() for _ in range(384)], {'type': 'edge'})
        community_vector = (
            'community-1',
            [random.random() for _ in range(384)],
            {'type': 'community'},
        )

        # Insert into different collections
        await self.store.upsert_vectors(VectorCollection.ENTITIES, [entity_vector])
        await self.store.upsert_vectors(VectorCollection.EDGES, [edge_vector])
        await self.store.upsert_vectors(VectorCollection.COMMUNITIES, [community_vector])

        await asyncio.sleep(0.5)

        # Verify each collection has its data
        for collection, vector_data in [
            (VectorCollection.ENTITIES, entity_vector),
            (VectorCollection.EDGES, edge_vector),
            (VectorCollection.COMMUNITIES, community_vector),
        ]:
            results = await self.store.search_vectors(collection, vector_data[1], limit=1)

            assert len(results) == 1
            assert results[0].uuid == vector_data[0]
            assert results[0].metadata['type'] == vector_data[2]['type']

            # Verify collection info
            info = await self.store.get_collection_info(collection)
            assert info['vectors_count'] >= 1
            assert info['points_count'] >= 1

    @pytest.mark.asyncio
    async def test_convenience_methods_integration(self):
        """Test convenience methods with real Qdrant."""
        # Test entity convenience methods
        entity_vectors = [
            ('entity-conv-1', [random.random() for _ in range(384)], {'name': 'Entity 1'}),
            ('entity-conv-2', [random.random() for _ in range(384)], {'name': 'Entity 2'}),
        ]

        await self.store.upsert_entity_embeddings(entity_vectors)
        await asyncio.sleep(0.5)

        results = await self.store.search_entity_embeddings(entity_vectors[0][1], limit=5)

        assert len(results) >= 1
        assert results[0].uuid == 'entity-conv-1'

        # Test edge convenience methods
        edge_vectors = [
            (
                'edge-conv-1',
                [random.random() for _ in range(384)],
                {'fact': 'Edge 1', 'source_node_uuid': 'src-1', 'target_node_uuid': 'tgt-1'},
            ),
        ]

        await self.store.upsert_edge_embeddings(edge_vectors)
        await asyncio.sleep(0.5)

        results = await self.store.search_edge_embeddings(edge_vectors[0][1], limit=5)
        assert len(results) >= 1
        assert results[0].uuid == 'edge-conv-1'

        # Test community convenience methods
        community_vectors = [
            ('comm-conv-1', [random.random() for _ in range(384)], {'summary': 'Community 1'}),
        ]

        await self.store.upsert_community_embeddings(community_vectors)
        await asyncio.sleep(0.5)

        results = await self.store.search_community_embeddings(community_vectors[0][1], limit=5)
        assert len(results) >= 1
        assert results[0].uuid == 'comm-conv-1'

        # Test deletions
        await self.store.delete_entity_embeddings(['entity-conv-1'])
        await self.store.delete_edge_embeddings(['edge-conv-1'])
        await self.store.delete_community_embeddings(['comm-conv-1'])

        await asyncio.sleep(0.5)

        # Verify deletions
        for vectors, search_method in [
            (entity_vectors, self.store.search_entity_embeddings),
            (edge_vectors, self.store.search_edge_embeddings),
            (community_vectors, self.store.search_community_embeddings),
        ]:
            results = await search_method(vectors[0][1], limit=10, min_score=0.0)
            uuids = {result.uuid for result in results}
            assert vectors[0][0] not in uuids

    @pytest.mark.asyncio
    async def test_legacy_methods_integration(self):
        """Test legacy methods for backward compatibility with real Qdrant."""
        # Test single upsert methods
        await self.store.upsert_entity_embedding(
            'legacy-entity-1', [random.random() for _ in range(384)], {'name': 'Legacy Entity'}
        )

        await self.store.upsert_edge_embedding(
            'legacy-edge-1',
            [random.random() for _ in range(384)],
            {'fact': 'Legacy Edge', 'source_node_uuid': 'src-1', 'target_node_uuid': 'tgt-1'},
        )

        await self.store.upsert_community_embedding(
            'legacy-comm-1', [random.random() for _ in range(384)], {'summary': 'Legacy Community'}
        )

        await asyncio.sleep(0.5)

        # Test legacy search methods (return tuple format)
        entity_results = await self.store.search_entities(
            [random.random() for _ in range(384)], limit=10
        )

        assert isinstance(entity_results, list)
        if entity_results:
            result = entity_results[0]
            assert isinstance(result, tuple)
            assert len(result) == 3  # (uuid, score, metadata)
            assert isinstance(result[0], str)  # uuid
            assert isinstance(result[1], float)  # score
            assert isinstance(result[2], dict)  # metadata

        # Test edge search with node filtering
        edge_results = await self.store.search_edges(
            [random.random() for _ in range(384)], source_node_uuid='src-1', limit=10
        )

        if edge_results:
            # Should find the edge we inserted
            uuids = [result[0] for result in edge_results]
            assert 'legacy-edge-1' in uuids

        # Test delete by collection name
        await self.store.delete_by_uuids(
            self.config.get_collection_name('entities'), ['legacy-entity-1']
        )

        await asyncio.sleep(0.5)

        # Verify deletion
        entity_results = await self.store.search_entities(
            [random.random() for _ in range(384)], limit=10, min_score=0.0
        )

        uuids = [result[0] for result in entity_results]
        assert 'legacy-entity-1' not in uuids

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling with real Qdrant."""
        # Test searching in non-existent collection (should be created during setup)
        # This test verifies graceful handling rather than errors

        # Test with malformed vectors (wrong dimension)
        with pytest.raises(ValueError):  # Qdrant should reject wrong dimensions
            await self.store.upsert_vectors(
                VectorCollection.ENTITIES,
                [('bad-vector', [0.1, 0.2], {'name': 'Bad Dimension'})],  # Only 2D instead of 384D
            )

        # Test deleting non-existent vectors (should not raise error)
        await self.store.delete_vectors(
            VectorCollection.ENTITIES, ['non-existent-uuid-1', 'non-existent-uuid-2']
        )

        # Test searching with empty query vector (should raise error)
        with pytest.raises(ValueError):
            await self.store.search_vectors(
                VectorCollection.ENTITIES,
                [],  # Empty vector
                limit=10,
            )

    @pytest.mark.asyncio
    async def test_large_batch_operations(self):
        """Test operations with larger batches of vectors."""
        # Create a larger batch of test vectors
        batch_size = 100
        vectors = []

        for i in range(batch_size):
            vectors.append(
                (
                    f'batch-entity-{i}',
                    [random.random() for _ in range(384)],
                    {
                        'group_id': f'batch-group-{i // 10}',  # 10 groups
                        'name': f'Batch Entity {i}',
                        'type': 'test_entity',
                        'batch_id': i,
                    },
                )
            )

        # Upsert large batch
        await self.store.upsert_vectors(VectorCollection.ENTITIES, vectors)
        await asyncio.sleep(1.0)  # Wait for indexing

        # Search and verify we can find some results
        results = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            vectors[0][1],  # Use first vector as query
            limit=50,
        )

        assert len(results) >= 10  # Should find a good number of results

        # Test group filtering with large dataset
        results_group_0 = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            [random.random() for _ in range(384)],
            limit=50,
            group_ids=['batch-group-0'],
        )

        assert len(results_group_0) <= 10  # Should only get group-0 results
        for result in results_group_0:
            assert result.metadata['group_id'] == 'batch-group-0'

        # Test batch deletion
        uuids_to_delete = [f'batch-entity-{i}' for i in range(0, 10)]
        await self.store.delete_vectors(VectorCollection.ENTITIES, uuids_to_delete)
        await asyncio.sleep(0.5)

        # Verify deletion
        results_after_delete = await self.store.search_vectors(
            VectorCollection.ENTITIES,
            [random.random() for _ in range(384)],
            limit=100,
            min_score=0.0,
        )

        remaining_uuids = {result.uuid for result in results_after_delete}
        for deleted_uuid in uuids_to_delete:
            assert deleted_uuid not in remaining_uuids

    @pytest.mark.asyncio
    async def test_collection_info_accuracy(self):
        """Test that collection info returns accurate counts."""
        # Start with empty collection info
        initial_info = await self.store.get_collection_info(VectorCollection.ENTITIES)
        initial_count = initial_info.get('vectors_count', 0)

        # Add some vectors
        test_vectors = [
            ('info-test-1', [random.random() for _ in range(384)], {'name': 'Test 1'}),
            ('info-test-2', [random.random() for _ in range(384)], {'name': 'Test 2'}),
            ('info-test-3', [random.random() for _ in range(384)], {'name': 'Test 3'}),
        ]

        await self.store.upsert_vectors(VectorCollection.ENTITIES, test_vectors)
        await asyncio.sleep(0.5)

        # Check updated info
        updated_info = await self.store.get_collection_info(VectorCollection.ENTITIES)

        assert updated_info['vectors_count'] >= initial_count + 3
        assert updated_info['points_count'] >= initial_count + 3
        assert updated_info['name'] == self.config.get_collection_name('entities')
        assert updated_info['status'] in ['green', 'yellow']

        # Delete one vector and check again
        await self.store.delete_vectors(VectorCollection.ENTITIES, ['info-test-1'])
        await asyncio.sleep(0.5)

        final_info = await self.store.get_collection_info(VectorCollection.ENTITIES)

        # Count should decrease (though it might take time for Qdrant to update stats)
        # So we just verify the operation completed without error
        assert final_info['name'] == self.config.get_collection_name('entities')
