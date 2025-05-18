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
import typing

import boto3
from pydantic import BaseModel, Field

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError

logger = logging.getLogger(__name__)

# Default model for Anthropic Claude on Bedrock
DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


class BedrockLLMConfig(LLMConfig):
    """Configuration for the Bedrock LLM client."""
    region_name: str = Field(default="us-east-1")
    model_id: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None


class BedrockClient(LLMClient):
    """
    BedrockClient is a client class for interacting with AWS Bedrock language models.

    This class extends the LLMClient and provides methods to initialize the client
    and generate responses from the AWS Bedrock language models (primarily Anthropic Claude).

    Attributes:
        model (str): The model name/ID to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        client: The boto3 bedrock-runtime client.

    Methods:
        __init__(config: BedrockLLMConfig | None = None, cache: bool = False):
            Initializes the BedrockClient with the provided configuration and cache setting.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    def __init__(
        self,
        config: BedrockLLMConfig | None = None,
        cache: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the BedrockClient with the provided configuration and cache setting.

        Args:
            config (BedrockLLMConfig | None): The configuration for the LLM client, including AWS credentials, region, model, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            max_tokens (int): The maximum number of tokens to generate in a response.
        """
        if config is None:
            config = BedrockLLMConfig()

        super().__init__(config, cache)

        # Use model_id if provided, otherwise use the model from LLMConfig base class
        self.model = config.model_id or config.model or DEFAULT_MODEL
        self.max_tokens = max_tokens

        # Set up the AWS Bedrock client
        # If AWS credentials are provided, use them; otherwise, rely on environment variables or IAM role
        session_kwargs = {"region_name": config.region_name}
        
        if config.aws_access_key_id and config.aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": config.aws_access_key_id,
                "aws_secret_access_key": config.aws_secret_access_key,
            })
            
        self.client = boto3.client("bedrock-runtime", **session_kwargs)

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the AWS Bedrock language model.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int): The maximum number of tokens to generate in the response.
            model_size (ModelSize): The size of the model to use (small, medium, large).

        Returns:
            dict[str, typing.Any]: The response from the language model.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            Exception: If there is an error generating the response.
        """
        try:
            # Determine if we're using an Anthropic Claude model
            is_claude_model = "anthropic.claude" in self.model

            # Process messages based on model type
            if is_claude_model:
                return await self._generate_claude_response(messages, response_model, max_tokens)
            else:
                # For other model types, implement as needed
                raise NotImplementedError(f"Model {self.model} is not currently supported in BedrockClient")
                
        except Exception as e:
            # Check if it's a rate limit error
            if 'rate limit' in str(e).lower() or 'quota' in str(e).lower() or 'throttle' in str(e).lower():
                raise RateLimitError from e
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def _generate_claude_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from an Anthropic Claude model on AWS Bedrock.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int): The maximum number of tokens to generate in the response.

        Returns:
            dict[str, typing.Any]: The response from the language model.
        """
        # Format messages for Claude's message API
        claude_messages = []
        system_prompt = ""

        # Extract system message if present
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content
            messages = messages[1:]

        # Format the response model schema if provided
        if response_model is not None:
            # Get the schema from the Pydantic model
            pydantic_schema = response_model.model_json_schema()
            
            # Add to system prompt
            schema_instructions = (
                f"Output ONLY valid JSON matching this schema: {json.dumps(pydantic_schema)}.\n"
                "Do not include any explanatory text before or after the JSON.\n\n"
            )
            
            system_prompt = f"{system_prompt}\n\n{schema_instructions}" if system_prompt else schema_instructions

        # Convert messages to Claude's format
        for m in messages:
            m.content = self._clean_input(m.content)
            claude_messages.append({
                "role": "user" if m.role == "user" else "assistant",
                "content": m.content
            })

        # Prepare the request body for Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "messages": claude_messages,
        }
        
        # Add system prompt if present
        if system_prompt:
            request_body["system"] = system_prompt

        # Invoke the Bedrock model
        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )

        # Parse the response
        response_body = json.loads(response["body"].read().decode("utf-8"))
        response_content = response_body.get("content", [{"text": ""}])[0].get("text", "")

        # If this was a structured output request, parse the response into the Pydantic model
        if response_model is not None:
            try:
                # Try to parse the response as JSON
                response_json = json.loads(response_content)
                validated_model = response_model.model_validate(response_json)

                # Return as a dictionary for API consistency
                return validated_model.model_dump()
            except Exception as e:
                raise Exception(f"Failed to parse structured response: {e}") from e

        # Otherwise, return the response text as a dictionary
        return {"content": response_content}
