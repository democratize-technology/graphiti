"""Formatting utilities for Graphiti MCP Server."""

from datetime import date, datetime, time
from typing import Any

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from models.response_types import EdgeResult, NodeResult
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


def to_node_result(node: EntityNode) -> NodeResult:
    """Build a NodeResult TypedDict from an EntityNode, dropping embeddings.

    Attributes are sanitized through _json_safe for the same reason as
    format_node_result: LLM-extracted attributes can carry raw neo4j temporal
    types that aren't JSON-serializable.
    """
    attrs = node.attributes if node.attributes else {}
    attrs = {k: v for k, v in attrs.items() if 'embedding' not in k.lower()}
    attrs = _json_safe(attrs)
    return NodeResult(
        uuid=node.uuid,
        name=node.name,
        labels=node.labels if node.labels else [],
        created_at=node.created_at.isoformat() if node.created_at else None,
        summary=node.summary,
        group_id=node.group_id,
        attributes=attrs,
    )


def to_edge_result(edge: EntityEdge) -> EdgeResult:
    """Build an EdgeResult TypedDict from an EntityEdge."""
    return EdgeResult(
        uuid=edge.uuid,
        name=edge.name,
        fact=edge.fact,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        group_id=edge.group_id,
        created_at=edge.created_at.isoformat() if edge.created_at else None,
        valid_at=edge.valid_at.isoformat() if edge.valid_at else None,
        invalid_at=edge.invalid_at.isoformat() if edge.invalid_at else None,
    )


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
