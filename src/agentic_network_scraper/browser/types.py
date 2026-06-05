"""Pydantic models for the browser capture layer."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class BodyKind(StrEnum):
    """Recognized response body formats."""

    JSON = "json"
    NDJSON = "ndjson"
    JSONP = "jsonp"
    SSE = "sse"
    BINARY = "binary"
    EMPTY = "empty"
    UNPARSEABLE = "unparseable"


class JsonShape(BaseModel):
    """Structural summary of a JSON value used by the agent to understand response shapes without seeing raw data."""

    root_type: str  # "object" | "array" | "scalar" | "null"
    keys: list[str] | None = None
    array_length: int | None = None
    item_shape: "JsonShape | None" = None
    fields: "dict[str, JsonShape] | None" = None  # nested shapes for object keys whose value is itself a dict/array
    truncated: bool = False


JsonShape.model_rebuild()


class CapturedRequest(BaseModel):
    """HTTP request metadata recorded during a browser session."""

    method: str
    url: str
    url_template: str


class CapturedResponse(BaseModel):
    """Full metadata for one captured HTTP response, including body kind and parse outcome."""

    response_id: str
    request: CapturedRequest
    status: int
    body_kind: BodyKind
    size_bytes: int
    parse_error: str | None = None


class ResponseDigest(BaseModel):
    """LLM-safe view of a captured response - no raw bodies, no sensitive headers."""

    response_id: str
    method: str
    url_template: str
    status: int
    body_kind: BodyKind
    size_bytes: int
    seen_count: int = 1
    parse_error: str | None = None  # set when the body could not be parsed, so the agent can skip it
    json_shape: JsonShape | None = None


class ResponseSample(BaseModel):
    """Actual values from a captured response - the full body if small, else a clipped representative slice."""

    response_id: str
    truncated: bool
    body: Any
