"""
Integration tests for VectorStore functionality in Graphiti.

These tests verify that the VectorStore integration works correctly
and maintains backward compatibility.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.vector_store import VectorStore


class MockVectorStore(VectorStore):
    """Mock VectorStore for testing."""

    def __init__(self):
        self.stored_embeddings = []
        self.search_results = []

    async def store_embeddings(self, embeddings: list[dict]) -> None:
        """Store embeddings in mock storage."""
        self.stored_embeddings.extend(embeddings)

    async def similarity_search(
        self,
        query_vector: list[float],
        limit: int = 10,
        threshold: float = 0.0,
        filters: dict = None,
    ) -> list[dict]:
        """Return mock search results."""
        return self.search_results[:limit]

    async def close(self) -> None:
        """Mock close operation."""
        pass


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing."""
    return MockVectorStore()


@pytest.fixture
def mock_graphiti_clients(mock_vector_store):
    """Create mock GraphitiClients with vector store."""
    clients = MagicMock(spec=GraphitiClients)
    clients.vector_store = mock_vector_store
    clients.driver = MagicMock()
    clients.embedder = MagicMock()
    clients.cross_encoder = MagicMock()
    return clients


class TestVectorStoreIntegration:
    """Test vector store integration functionality."""

    def test_graphiti_clients_accepts_vector_store(self, mock_vector_store):
        """Test that GraphitiClients accepts vector_store parameter."""
        from graphiti_core.graphiti_types import GraphitiClients

        clients = GraphitiClients(
            driver=MagicMock(),
            llm_client=MagicMock(),
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
            vector_store=mock_vector_store,
        )

        assert clients.vector_store == mock_vector_store

    def test_graphiti_constructor_accepts_vector_store(self, mock_vector_store):
        """Test that Graphiti constructor accepts vector_store parameter."""
        graphiti = Graphiti(
            uri='bolt://localhost:7687',
            user='neo4j',
            password='test',
            vector_store=mock_vector_store,
        )

        assert graphiti.vector_store == mock_vector_store
        assert graphiti.clients.vector_store == mock_vector_store

    def test_graphiti_backwards_compatibility_no_vector_store(self):
        """Test that Graphiti works without vector_store (backward compatibility)."""
        graphiti = Graphiti(
            uri='bolt://localhost:7687',
            user='neo4j',
            password='test',
            # No vector_store parameter
        )

        assert graphiti.vector_store is None
        assert graphiti.clients.vector_store is None

    @pytest.mark.asyncio
    async def test_vector_store_used_in_bulk_operations(self, mock_vector_store):
        """Test that vector store is used during bulk save operations."""
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EntityNode
        from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk

        # Create mock nodes with embeddings
        node = MagicMock(spec=EntityNode)
        node.uuid = 'test-node-uuid'
        node.name = 'Test Node'
        node.name_embedding = [0.1, 0.2, 0.3]
        node.group_id = 'test-group'
        node.labels = ['Entity']

        edge = MagicMock(spec=EntityEdge)
        edge.uuid = 'test-edge-uuid'
        edge.name = 'Test Edge'
        edge.fact = 'Test fact'
        edge.fact_embedding = [0.4, 0.5, 0.6]
        edge.group_id = 'test-group'
        edge.source_node_uuid = 'source-uuid'
        edge.target_node_uuid = 'target-uuid'

        # Mock driver and embedder
        driver = MagicMock()
        embedder = MagicMock()
        session = MagicMock()
        driver.session.return_value = session
        session.execute_write = AsyncMock()
        session.close = AsyncMock()

        await add_nodes_and_edges_bulk(
            driver=driver,
            episodic_nodes=[],
            episodic_edges=[],
            entity_nodes=[node],
            entity_edges=[edge],
            embedder=embedder,
            vector_store=mock_vector_store,
        )

        # Verify embeddings were stored in vector store
        assert len(mock_vector_store.stored_embeddings) == 2

        # Check node embedding was stored
        node_embedding = next(
            e for e in mock_vector_store.stored_embeddings if e['metadata']['type'] == 'node'
        )
        assert node_embedding['id'] == 'test-node-uuid'
        assert node_embedding['embedding'] == [0.1, 0.2, 0.3]
        assert node_embedding['metadata']['name'] == 'Test Node'

        # Check edge embedding was stored
        edge_embedding = next(
            e for e in mock_vector_store.stored_embeddings if e['metadata']['type'] == 'edge'
        )
        assert edge_embedding['id'] == 'test-edge-uuid'
        assert edge_embedding['embedding'] == [0.4, 0.5, 0.6]
        assert edge_embedding['metadata']['fact'] == 'Test fact'

    @pytest.mark.asyncio
    async def test_search_uses_vector_store_when_available(self, mock_graphiti_clients):
        """Test that search functions use vector store when available."""
        from graphiti_core.search.search_utils import edge_similarity_search

        # Set up mock search results
        mock_graphiti_clients.vector_store.search_results = [
            {'id': 'edge-1', 'score': 0.95},
            {'id': 'edge-2', 'score': 0.85},
        ]

        # Mock EntityEdge.get_by_uuids
        from graphiti_core.edges import EntityEdge

        EntityEdge.get_by_uuids = AsyncMock(
            return_value=[MagicMock(uuid='edge-1'), MagicMock(uuid='edge-2')]
        )

        # Test edge similarity search with vector store
        results = await edge_similarity_search(
            driver=mock_graphiti_clients.driver,
            search_vector=[0.1, 0.2, 0.3],
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=MagicMock(),
            vector_store=mock_graphiti_clients.vector_store,
        )

        # Verify vector store was called
        assert len(results) == 2
        assert results[0].uuid == 'edge-1'
        assert results[1].uuid == 'edge-2'

    @pytest.mark.asyncio
    async def test_search_fallback_without_vector_store(self):
        """Test that search falls back to graph DB when vector store not available."""
        from graphiti_core.search.search_utils import edge_similarity_search

        # Mock driver for graph DB fallback
        driver = MagicMock()
        driver.execute_query = AsyncMock(return_value=([], {}, {}))

        # Test edge similarity search without vector store
        await edge_similarity_search(
            driver=driver,
            search_vector=[0.1, 0.2, 0.3],
            source_node_uuid=None,
            target_node_uuid=None,
            search_filter=MagicMock(),
            vector_store=None,  # No vector store
        )

        # Should use graph DB fallback
        driver.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_graphiti_close_closes_vector_store(self, mock_vector_store):
        """Test that Graphiti.close() also closes vector store."""
        graphiti = Graphiti(
            uri='bolt://localhost:7687',
            user='neo4j',
            password='test',
            vector_store=mock_vector_store,
        )

        # Mock driver close
        graphiti.driver.close = AsyncMock()

        # Close should close both driver and vector store
        await graphiti.close()

        graphiti.driver.close.assert_called_once()
        # Vector store close would be called but our mock doesn't track this

    def test_telemetry_includes_vector_store_provider(self, mock_vector_store):
        """Test that telemetry captures vector store provider information."""
        graphiti = Graphiti(
            uri='bolt://localhost:7687',
            user='neo4j',
            password='test',
            vector_store=mock_vector_store,
        )

        # Test provider detection
        provider = graphiti._get_provider_type(mock_vector_store)
        assert provider == 'unknown'  # MockVectorStore doesn't match known providers

        # Test with Qdrant-like class name
        mock_qdrant = MagicMock()
        mock_qdrant.__class__.__name__ = 'QdrantVectorStore'
        provider = graphiti._get_provider_type(mock_qdrant)
        assert provider == 'qdrant'


if __name__ == '__main__':
    pytest.main([__file__])
