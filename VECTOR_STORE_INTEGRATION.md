# VectorStore Integration for Graphiti

This document describes the VectorStore integration added to Graphiti for high-performance vector similarity searches.

## Overview

The VectorStore integration adds optional support for dedicated vector databases (Qdrant, Pinecone, Chroma, etc.) to enhance search performance while maintaining full backward compatibility.

## Architecture

### Clean Dependency Injection Pattern

The integration follows Graphiti's established dependency injection pattern:

```python
# GraphitiClients now includes optional vector_store
class GraphitiClients(BaseModel):
    driver: GraphDriver
    llm_client: LLMClient
    embedder: EmbedderClient
    cross_encoder: CrossEncoderClient
    vector_store: VectorStore | None = None  # NEW: Optional vector store
    ensure_ascii: bool = False
```

### Main API Integration

```python
# Graphiti constructor accepts optional vector_store
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j", 
    password="password",
    vector_store=qdrant_store  # Optional: enhances search performance
)
```

## Key Features

### 1. Backward Compatibility
- VectorStore is completely optional (`None` by default)
- Existing code works without any changes
- No breaking changes to existing APIs

### 2. Transparent Performance Enhancement
- When VectorStore is configured, similarity searches use vector database
- Falls back to graph database if VectorStore is not available
- Search API remains identical for users

### 3. Automatic Storage
- Embeddings are automatically stored in both graph database and vector store
- Metadata preserved for filtering capabilities
- Consistent data between storage systems

### 4. Enhanced Search Performance
- Vector similarity searches leverage optimized vector databases
- Maintains hybrid search capabilities with BM25 and graph traversal
- Filtering by group_id, entity types, and other metadata

## Usage Examples

### With VectorStore (Enhanced Performance)

```python
from graphiti_core.graphiti import Graphiti
from graphiti_core.vector_store.qdrant_store import QdrantVectorStore

# Initialize vector store
vector_store = QdrantVectorStore(
    host="localhost", 
    port=6333,
    collection_name="graphiti_embeddings",
    vector_size=1536
)

# Initialize Graphiti with vector store
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password", 
    vector_store=vector_store  # Optional enhancement
)

# All operations work identically - enhanced performance is automatic
results = await graphiti.search_("mobile app development")
```

### Without VectorStore (Traditional)

```python
# Traditional usage - no changes needed
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
    # vector_store=None by default
)

# Same API - uses graph database for all operations
results = await graphiti.search_("mobile app development")
```

## Implementation Details

### Search Function Updates

The similarity search functions were enhanced to use VectorStore when available:

```python
async def edge_similarity_search(..., vector_store=None):
    # Use vector store if available for high-performance search
    if vector_store is not None:
        results = await vector_store.similarity_search(...)
        return await EntityEdge.get_by_uuids(driver, edge_uuids)
    
    # Fallback to graph database vector search
    # ... existing graph DB logic ...
```

### Bulk Operations Integration

Embedding storage was enhanced to save to VectorStore during bulk operations:

```python
async def add_nodes_and_edges_bulk_tx(..., vector_store=None):
    # Save to graph database (existing logic)
    await tx.run(entity_edge_save_bulk, entity_edges=edges)
    
    # Also save to vector store if available (new logic)
    if vector_store is not None:
        await vector_store.store_embeddings(node_embeddings)
        await vector_store.store_embeddings(edge_embeddings)
```

## Benefits

### Performance
- Dedicated vector databases optimized for similarity search
- Faster query response times for large knowledge graphs
- Improved scalability for vector operations

### Flexibility
- Choose vector database that fits your infrastructure
- Mix and match storage backends
- Easy to add/remove vector store without code changes

### Consistency
- Single API for all search operations
- Consistent metadata handling
- Graceful degradation when vector store unavailable

## Migration Path

### For Existing Users
1. **No changes required** - existing code continues to work
2. **Optional upgrade** - add vector store for performance benefits
3. **Gradual adoption** - can enable vector store per environment

### For New Users
1. Start with basic Graphiti setup (graph database only)
2. Add VectorStore when performance needs increase
3. Choose vector database that fits infrastructure requirements

## Technical Architecture Benefits

This integration demonstrates several architectural best practices:

1. **Dependency Injection**: VectorStore passed through GraphitiClients
2. **Optional Enhancement**: Core functionality works without VectorStore
3. **Single Responsibility**: VectorStore only handles similarity search
4. **Graceful Fallback**: Automatic fallback to graph database
5. **Clean Separation**: Search logic cleanly separated from storage choice

The pattern can be reused for other optional high-performance components in the future.