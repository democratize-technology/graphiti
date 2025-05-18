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

import json
import logging
from collections.abc import Iterable
from typing import Any, Dict, List, Union

import boto3
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

# Default embedding model for Bedrock
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"


class BedrockEmbedderConfig(EmbedderConfig):
    """Configuration for the Bedrock embedder."""
    region_name: str = Field(default="us-east-1")
    model_id: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None


class BedrockEmbedder(EmbedderClient):
    """
    AWS Bedrock Embedder Client
    """

    def __init__(self, config: BedrockEmbedderConfig | None = None):
        """
        Initialize the BedrockEmbedder with the provided configuration.

        Args:
            config (BedrockEmbedderConfig | None): The configuration for the embedder.
        """
        if config is None:
            config = BedrockEmbedderConfig()
        self.config = config

        # Set up the AWS Bedrock client
        # If AWS credentials are provided, use them; otherwise, rely on environment variables or IAM role
        session_kwargs = {"region_name": config.region_name}
        
        if config.aws_access_key_id and config.aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": config.aws_access_key_id,
                "aws_secret_access_key": config.aws_secret_access_key,
            })
            
        self.client = boto3.client("bedrock-runtime", **session_kwargs)
        self.model_id = config.model_id

    async def create(
        self, input_data: Union[str, List[str], Iterable[int], Iterable[Iterable[int]]]
    ) -> List[float]:
        """
        Create embeddings for the given input data using AWS Bedrock embedding models.

        Args:
            input_data: The input data to create embeddings for. Can be a string, list of strings,
                       or an iterable of integers or iterables of integers.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Convert input to string if not already
        if not isinstance(input_data, str):
            if isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
                # Join list of strings with spaces
                input_data = " ".join(input_data)
            else:
                # Convert any other type to string
                input_data = str(input_data)

        # Check which model we're using and format request accordingly
        model_id = self.model_id or DEFAULT_EMBEDDING_MODEL
        
        if "amazon.titan" in model_id:
            return await self._create_titan_embedding(input_data)
        elif "cohere.embed" in model_id:
            return await self._create_cohere_embedding(input_data)
        else:
            raise ValueError(f"Unsupported embedding model: {model_id}")

    async def _create_titan_embedding(self, input_text: str) -> List[float]:
        """
        Create embeddings using Amazon Titan embedding models.
        
        Args:
            input_text: The text to create embeddings for.
            
        Returns:
            A list of floats representing the embedding vector.
        """
        request_body = {
            "inputText": input_text,
        }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        
        response_body = json.loads(response["body"].read().decode("utf-8"))
        embedding = response_body.get("embedding", [])
        
        return embedding

    async def _create_cohere_embedding(self, input_text: str) -> List[float]:
        """
        Create embeddings using Cohere embedding models on Bedrock.
        
        Args:
            input_text: The text to create embeddings for.
            
        Returns:
            A list of floats representing the embedding vector.
        """
        request_body = {
            "texts": [input_text],
            "input_type": "search_document",
        }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        
        response_body = json.loads(response["body"].read().decode("utf-8"))
        embeddings = response_body.get("embeddings", [[]])
        
        return embeddings[0]

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of input texts.
        
        Args:
            input_data_list: A list of strings to create embeddings for.
            
        Returns:
            A list of embedding vectors.
        """
        # Check which model we're using and format request accordingly
        model_id = self.model_id or DEFAULT_EMBEDDING_MODEL
        
        if "amazon.titan" in model_id:
            # Titan doesn't support batch embeddings natively, so process serially
            embeddings = []
            for text in input_data_list:
                embedding = await self._create_titan_embedding(text)
                embeddings.append(embedding)
            return embeddings
        elif "cohere.embed" in model_id:
            # Cohere supports batch embeddings
            request_body = {
                "texts": input_data_list,
                "input_type": "search_document",
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            
            response_body = json.loads(response["body"].read().decode("utf-8"))
            embeddings = response_body.get("embeddings", [])
            
            return embeddings
        else:
            raise ValueError(f"Unsupported embedding model: {model_id}")
