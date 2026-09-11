"""Formatting utilities for Graphiti MCP Server."""

from datetime import date, datetime, time
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from neo4j.time import Date as Neo4jDate
from neo4j.time import DateTime as Neo4jDateTime
from neo4j.time import Duration as Neo4jDuration
from neo4j.time import Time as Neo4jTime


def _json_safe(value: Any) -> Any:
    """Recursively convert neo4j temporal types into JSON-serializable values.

    graphiti_core's `EntityEdge`/`EntityNode.attributes` dict holds LLM-extracted
    entity attributes copied verbatim from the raw Neo4j record. graphiti_core only
    converts its own known temporal fields (created_at/valid_at/...); any extracted
    attribute that happens to be a date/time value comes back as a `neo4j.time.*`
    object and makes `model_dump(mode='json')` raise
    "Unable to serialize unknown type" (unfixed as of graphiti-core 0.30.2).
    """
    if isinstance(value, (Neo4jDateTime, Neo4jDate, Neo4jTime)):
        return value.to_native().isoformat()
    if isinstance(value, Neo4jDuration):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def format_node_result(node: EntityNode) -> dict[str, Any]:
    """Format an entity node into a readable result.

    Since EntityNode is a Pydantic BaseModel, we can use its built-in serialization capabilities.
    Excludes embedding vectors to reduce payload size and avoid exposing internal representations.
    Dumps in 'python' mode (rather than 'json') and sanitizes manually since the
    `attributes` dict can carry raw neo4j temporal types pydantic doesn't recognize.

    Args:
        node: The EntityNode to format

    Returns:
        A dictionary representation of the node with serialized dates and excluded embeddings
    """
    result = _json_safe(
        node.model_dump(
            mode='python',
            exclude={
                'name_embedding',
            },
        )
    )
    # Remove any embedding that might be in attributes
    result.get('attributes', {}).pop('name_embedding', None)
    return result


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.
    Dumps in 'python' mode (rather than 'json') and sanitizes manually since the
    `attributes` dict can carry raw neo4j temporal types pydantic doesn't recognize.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = _json_safe(
        edge.model_dump(
            mode='python',
            exclude={
                'fact_embedding',
            },
        )
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result
