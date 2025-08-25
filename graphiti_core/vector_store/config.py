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
import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator


def _parse_port_env() -> int:
    """Parse port from environment with validation"""
    port_str = os.getenv('QDRANT_PORT', '6333')
    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ValueError(f'Port must be between 1 and 65535, got {port}')
        return port
    except ValueError as e:
        raise ValueError(f'Invalid QDRANT_PORT environment variable: {e}') from e


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store"""

    url: str = Field(
        default_factory=lambda: os.getenv('QDRANT_URL', 'localhost'),
        description='Qdrant server URL or hostname',
    )
    port: int = Field(
        default_factory=lambda: _parse_port_env(),
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

    @validator('url')
    def validate_url(cls, v):
        """Validate URL for security and correctness"""
        if not v or not isinstance(v, str):
            raise ValueError('URL must be a non-empty string')
        
        # Allow localhost and IP addresses for common use cases
        if v in ('localhost', '127.0.0.1', '::1'):
            return v
            
        # For other URLs, validate more strictly
        if '://' in v:
            parsed = urlparse(v)
            if parsed.scheme not in ('http', 'https', 'grpc'):
                raise ValueError('URL scheme must be http, https, or grpc')
            if not parsed.netloc:
                raise ValueError('URL must have a valid hostname')
        else:
            # Hostname/IP validation
            if not re.match(r'^[a-zA-Z0-9.-]+$', v):
                raise ValueError('Hostname contains invalid characters')
            if '..' in v:
                raise ValueError('Hostname cannot contain consecutive dots')
                
        return v
    
    @validator('port')  
    def validate_port(cls, v):
        """Validate port is in valid range"""
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v
        
    @validator('collection_prefix')
    def validate_collection_prefix(cls, v):
        """Validate collection prefix for safety"""
        if not v or not isinstance(v, str):
            raise ValueError('Collection prefix must be a non-empty string')
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Collection prefix can only contain letters, numbers, underscores, and hyphens')
        if len(v) > 50:
            raise ValueError('Collection prefix must be 50 characters or less')
        return v
        
    @validator('timeout')
    def validate_timeout(cls, v):
        """Validate timeout is reasonable"""
        if not (0.1 <= v <= 300.0):
            raise ValueError('Timeout must be between 0.1 and 300 seconds')
        return v
        
    @validator('embedding_dim')
    def validate_embedding_dim(cls, v):
        """Validate embedding dimension"""
        if not (1 <= v <= 4096):
            raise ValueError('Embedding dimension must be between 1 and 4096')
        return v

    class Config:
        # Allow Field default_factory to access environment variables
        arbitrary_types_allowed = True

    def get_collection_name(self, collection_type: str) -> str:
        """Get full collection name with prefix"""
        return f'{self.collection_prefix}_{collection_type}'
