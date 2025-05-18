# AWS Bedrock Example for Graphiti

This example demonstrates how to use Graphiti with AWS Bedrock for both LLM inference and embeddings.

## Prerequisites

- Python 3.10+
- Neo4j database (5.26+) running locally or accessible
- AWS account with access to Bedrock
- Permissions to use the required Bedrock models

## Setup

1. Install the necessary dependencies:

```bash
# Install graphiti-core with AWS Bedrock support
pip install 'graphiti-core[bedrock]'
```

2. Set up your AWS credentials. You can use any of the standard AWS credential methods:

   - Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
   - AWS credentials file (~/.aws/credentials)
   - IAM role (for EC2/ECS/Lambda environments)

3. Configure the example with your specific settings:

Create a `.env` file in the same directory as the example script with the following variables:

```
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Bedrock Model IDs
BEDROCK_LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_EMBEDDER_MODEL_ID=amazon.titan-embed-text-v2:0
```

## Available Bedrock Models

### LLM Models (for inference)

- `anthropic.claude-3-haiku-20240307-v1:0` - Fast & cost-effective
- `anthropic.claude-3-sonnet-20240229-v1:0` - Good balance of speed/quality
- `anthropic.claude-3-opus-20240229-v1:0` - Most capable

### Embedding Models

- `amazon.titan-embed-text-v2:0` - Amazon Titan text embeddings
- `cohere.embed-english-v3:0` - Cohere English embeddings 
- `cohere.embed-multilingual-v3:0` - Cohere multilingual embeddings

## Running the Example

```bash
python bedrock_example.py
```

## Troubleshooting

- **AccessDeniedException**: Ensure your AWS credentials have access to Bedrock and the specified models
- **ValidationException**: Check that you're using valid model IDs that you have access to
- **Connection errors to Neo4j**: Verify your Neo4j database is running and accessible

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Graphiti Documentation](https://help.getzep.com/graphiti/graphiti/overview)
- [Neo4j Documentation](https://neo4j.com/docs/)

