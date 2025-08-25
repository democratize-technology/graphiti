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

from pydantic import BaseModel, Field


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store"""

    url: str = Field(
        default_factory=lambda: os.getenv('QDRANT_URL', 'localhost'),
        description='Qdrant server URL or hostname',
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv('QDRANT_PORT', '6333')),
        description='Qdrant server port',
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv('QDRANT_API_KEY'),
        description='API key for Qdrant Cloud or secured instances',
    )
    use_memory: bool = Field(
        default_factory=lambda: os.getenv('QDRANT_USE_MEMORY', 'false').lower() == 'true',
        description='Use in-memory storage for testing (overrides url/port)',
    )
    collection_prefix: str = Field(
        default_factory=lambda: os.getenv('QDRANT_COLLECTION_PREFIX', 'graphiti'),
        description='Prefix for collection names to avoid conflicts',
    )
    embedding_dim: int = Field(
        default=1024,
        description='Dimension of embeddings to store',
    )
    timeout: float = Field(
        default=30.0,
        description='Request timeout in seconds',
    )

    class Config:
        # Allow Field default_factory to access environment variables
        arbitrary_types_allowed = True

    def get_collection_name(self, collection_type: str) -> str:
        """Get full collection name with prefix"""
        return f'{self.collection_prefix}_{collection_type}'
