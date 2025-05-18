"""
Example of using AWS Bedrock with Graphiti.

This example demonstrates how to initialize Graphiti with AWS Bedrock clients for both
LLM inference and embeddings.
"""

import asyncio
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.embedder.bedrock import BedrockEmbedder, BedrockEmbedderConfig
from graphiti_core.llm_client.bedrock_client import BedrockClient, BedrockLLMConfig
from graphiti_core.nodes import EpisodeType

# Load environment variables
load_dotenv()

# AWS Bedrock configuration (Either set these in .env or directly here)
region_name = os.getenv("AWS_REGION", "us-east-1")
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Model IDs - you can change these to any Bedrock model you have access to
llm_model_id = os.getenv("BEDROCK_LLM_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
embedder_model_id = os.getenv("BEDROCK_EMBEDDER_MODEL_ID", "amazon.titan-embed-text-v2:0")


async def main():
    """Main function to demonstrate AWS Bedrock integration with Graphiti."""
    print("Initializing Graphiti with AWS Bedrock...")

    # Initialize LLM config
    llm_config = BedrockLLMConfig(
        region_name=region_name,
        model_id=llm_model_id,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        temperature=0.1,  # Lower temperature for more consistent outputs
    )

    # Initialize Embedder config
    embedder_config = BedrockEmbedderConfig(
        region_name=region_name,
        model_id=embedder_model_id,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    # Initialize clients
    llm_client = BedrockClient(config=llm_config)
    embedder = BedrockEmbedder(config=embedder_config)

    # Initialize Graphiti with AWS Bedrock clients
    graphiti = Graphiti(
        "bolt://localhost:7687",  # Neo4j connection URI
        "neo4j",  # Neo4j username
        "password",  # Neo4j password
        llm_client=llm_client,
        embedder=embedder,
    )

    # Initialize the graph database with Graphiti's indices (only needs to be done once)
    await graphiti.build_indices_and_constraints()
    print("Indices built successfully")

    # Add some example episodes
    episodes = [
        "John works as a software engineer at Google.",
        "John enjoys playing basketball on weekends.",
        "John has a pet dog named Max.",
        "John likes to travel and visited Japan last summer.",
    ]

    print("Adding episodes...")
    episode_ids = []
    for i, episode in enumerate(episodes):
        # Use current time plus a small offset for each episode to maintain order
        timestamp = datetime.now(timezone.utc).replace(microsecond=i * 1000)
        
        # Add the episode to Graphiti
        episode_id = await graphiti.add_episode(
            content=episode,
            episode_type=EpisodeType.TEXT,
            timestamp=timestamp,
            source="example",
        )
        episode_ids.append(episode_id)
        print(f"Added episode {i+1}/{len(episodes)}")

    # Process any pending jobs
    await graphiti.jobs_processor.wait_for_jobs()
    print("All episodes processed")

    # Perform a search
    print("\nSearching for information about John...")
    edges = await graphiti.search("What are John's hobbies?", num_results=3)
    
    print("\nSearch results:")
    for edge in edges:
        print(f"- {edge.source.name} {edge.relationship_type} {edge.target.name} "
              f"(confidence: {edge.confidence:.2f})")
        
        # Print validity dates
        if edge.valid_at:
            print(f"  Valid from: {edge.valid_at}")
        if edge.invalid_at:
            print(f"  Valid until: {edge.invalid_at}")


if __name__ == "__main__":
    asyncio.run(main())
