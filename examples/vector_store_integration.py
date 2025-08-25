"""
Example demonstrating VectorStore integration with Graphiti.

This example shows how to use Graphiti with an optional VectorStore
for high-performance vector similarity searches.
"""

import asyncio
from datetime import datetime

from graphiti_core.graphiti import Graphiti
from graphiti_core.vector_store.qdrant_store import QdrantVectorStore


async def example_with_vector_store():
    """Example using Graphiti with QdrantVectorStore for enhanced search performance."""

    # Initialize vector store (optional - improves vector search performance)
    vector_store = QdrantVectorStore(
        host='localhost',
        port=6333,
        collection_name='graphiti_embeddings',
        vector_size=1536,  # OpenAI embedding dimensions
    )

    # Initialize Graphiti with vector store
    graphiti = Graphiti(
        uri='bolt://localhost:7687',
        user='neo4j',
        password='password',
        vector_store=vector_store,  # Optional: enhances search performance
    )

    try:
        # Build indices for optimal performance
        await graphiti.build_indices_and_constraints()

        # Add episodes (embeddings automatically stored in both graph DB and vector store)
        await graphiti.add_episode(
            name='Meeting Summary',
            episode_body='John and Sarah discussed the new product roadmap. They decided to prioritize mobile features for Q2.',
            source_description='team meeting notes',
            reference_time=datetime.now(),
            group_id='project_alpha',
        )

        await graphiti.add_episode(
            name='Customer Feedback',
            episode_body='Customer reported issues with mobile app performance. Slow loading times affecting user experience.',
            source_description='support ticket',
            reference_time=datetime.now(),
            group_id='project_alpha',
        )

        # Search uses vector store automatically for similarity search when available
        # Falls back to graph database if vector store not configured
        results = await graphiti.search_(
            query='mobile app development priorities', group_ids=['project_alpha']
        )

        print(f'Found {len(results.edges)} relevant edges')
        print(f'Found {len(results.nodes)} relevant nodes')

        # Specific similarity search benefits from vector store performance
        for edge in results.edges[:3]:
            print(f'- {edge.fact}')

    finally:
        # Clean close both graph database and vector store connections
        await graphiti.close()


async def example_without_vector_store():
    """Example using Graphiti without vector store (traditional approach)."""

    # Initialize Graphiti without vector store (uses graph DB for all operations)
    graphiti = Graphiti(
        uri='bolt://localhost:7687',
        user='neo4j',
        password='password',
        # vector_store=None by default
    )

    try:
        # Same API - search operations fall back to graph database vector capabilities
        results = await graphiti.search_(
            query='mobile app development', group_ids=['project_alpha']
        )
        print(f'Graph DB search found {len(results.edges)} edges')

    finally:
        await graphiti.close()


if __name__ == '__main__':
    # Run example with vector store for enhanced performance
    print('=== With Vector Store (Enhanced Performance) ===')
    asyncio.run(example_with_vector_store())

    print('\n=== Without Vector Store (Traditional) ===')
    asyncio.run(example_without_vector_store())
