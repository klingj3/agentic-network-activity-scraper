"""Pydantic models for the agent pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_network_scraper.browser.types import CapturedRequest


class VariableSpec(BaseModel):
    """A discovered URL variable - name, location, example values, and human description."""

    name: str = Field(
        description="Placeholder name as it appears in url_template, without braces (e.g. 'page' for {page})."
    )
    location: Literal["query", "path"] = Field(
        description="Where the variable lives in the URL: a query-string parameter or a path segment."
    )
    description: str = Field(
        description="Human-readable explanation of what the variable controls and how changing it affects the response."
    )
    example_values: list[str] = Field(
        default_factory=list,
        description="Concrete values observed to produce valid responses; the first is a sane default.",
    )
    required: bool = Field(
        default=False, description="True if the endpoint returns no usable data unless this variable is supplied."
    )


class EndpointNote(BaseModel):
    """Agent annotation for one captured endpoint - its relevance and a one-line summary recorded during exploration."""

    response_id: str = Field(
        description="ID of the captured response this note refers to, as returned by list_responses."
    )
    url_template: str = Field(
        description="Normalized endpoint URL (with {id}-style path placeholders) the response came from."
    )
    summary: str = Field(
        description="One-line description of what the endpoint returns and why it does or does not fit the goal."
    )
    relevant: bool = Field(
        description="True if this endpoint plausibly contains data matching the goal and is worth pursuing."
    )


class ExtractionState(BaseModel):
    """Mutable agent memory for a run: the goal plus the per-endpoint relevance notes recorded during exploration."""

    goal: str = Field(description="The natural-language extraction goal driving this run; set once at startup.")
    endpoint_notes: dict[str, EndpointNote] = Field(
        default_factory=dict, description="Per-endpoint notes keyed by response_id, summarizing relevance found so far."
    )


class StatePatch(BaseModel):
    """Partial update applied by update_state - omitted fields are unchanged."""

    endpoint_notes: dict[str, EndpointNote] | None = Field(
        default=None,
        description="If set, replaces the entire endpoint_notes map; prune stale entries by omitting them here.",
    )


class Synthesize(BaseModel):
    """Signal returned when the agent is confident enough to attempt extraction and validation."""

    kind: Literal["synthesize"] = "synthesize"
    source_response_ids: dict[str, str] = Field(
        description=(
            "Maps the short names used in the expression to the response_ids they resolve to, "
            "e.g. {'events': '<id>'}."
        ),
    )
    expression: str = Field(
        description=(
            "JMESPath evaluated as jmespath.search(expression, responses), where responses is keyed "
            "by source_response_ids; must yield a list of objects matching the target schema."
        ),
    )
    description: str = Field(
        description="Human-readable summary of what the extraction returns and where it comes from."
    )
    url_template: str | None = Field(
        default=None,
        description=(
            "Source endpoint URL with {name} placeholders for each entry in variable_specs; "
            "None if not parameterizable."
        ),
    )
    variable_specs: list[VariableSpec] = Field(
        default_factory=list,
        description=(
            "One spec per URL variable observed to affect the response; empty when the endpoint "
            "takes no useful parameters."
        ),
    )


class Abandon(BaseModel):
    """Signal returned when the agent determines no suitable resource exists in the captured responses."""

    kind: Literal["abandon"] = "abandon"
    reason: str = Field(description="Explanation of why no captured response can satisfy the goal.")
    responses_considered: list[str] = Field(description="response_ids the agent evaluated before giving up.")


class ParseAttempt(BaseModel):
    """Result of an attempt_parse tool call - success flag plus sample output or structured error."""

    success: bool = Field(description="True if the expression both ran and produced output matching the target schema.")
    sample_output: Any | None = Field(
        default=None, description="On success, the first few validated items; None on failure."
    )
    error: str | None = Field(
        default=None,
        description="On failure, the error message (JMESPath error, validation report, or missing-id detail).",
    )
    error_type: Literal["jmespath", "validation", "missing", "empty"] | None = Field(
        default=None,
        description=(
            "Category of failure: 'jmespath' (expression error), 'validation' (output mismatched "
            "schema), 'missing' (unknown response_id), or 'empty' (ran and validated but matched no items)."
        ),
    )


class ExtractionBlueprint(BaseModel):
    """Verified extraction result containing the source endpoints, JMESPath expression, and sample output."""

    target_url: str = Field(description="The page URL that was scraped to capture these endpoints.")
    goal: str = Field(description="The extraction goal this blueprint satisfies.")
    endpoints: dict[str, CapturedRequest] = Field(
        description="The captured source request(s) keyed by the short name the expression uses for each."
    )
    expression: str = Field(
        description="Verified JMESPath that transforms the source responses into the target schema."
    )
    target_schema: dict[str, Any] = Field(description="JSON Schema of the target item type the output conforms to.")
    sample_output: Any = Field(description="A few validated items produced by the expression, for confirmation.")
    description: str = Field(description="Human-readable summary of what the blueprint extracts.")
    url_template: str | None = Field(
        default=None, description="Source endpoint URL with {name} placeholders for each variable in variable_specs."
    )
    variable_specs: list[VariableSpec] = Field(
        default_factory=list, description="Parameterizable URL variables callers can vary to re-fetch the endpoint."
    )


class NoResource(BaseModel):
    """Returned when the pipeline cannot find extractable data matching the goal."""

    reason: str = Field(
        description="Why extraction failed - no matching data, exhausted budget, verification failure, etc."
    )
    responses_considered: list[str] = Field(description="response_ids (or endpoint notes) evaluated before failing.")
