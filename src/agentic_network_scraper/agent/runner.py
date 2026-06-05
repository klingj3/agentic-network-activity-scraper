"""Execute a stored ExtractionBlueprint against live endpoints - the library's read side.

A blueprint is a self-contained pydantic model, so persistence is left to the caller: serialize with
blueprint.model_dump_json() and reload with ExtractionBlueprint.model_validate_json(), storing that JSON in
a file, a Postgres column, or anywhere else. BlueprintRunner only fetches and extracts.
"""

from __future__ import annotations

from typing import Any, cast

import httpx
from pydantic import BaseModel

from agentic_network_scraper.browser.engine import detect
from agentic_network_scraper.browser.types import BodyKind
from agentic_network_scraper.common.urls import merge_query

from .extraction import run_extract, validate_extraction
from .types import ExtractionBlueprint


class BlueprintError(RuntimeError):
    """Raised when a blueprint's endpoints cannot be fetched or parsed, or its expression yields nothing."""


class BlueprintRunner:
    """Fetch a blueprint's source endpoints and apply its JMESPath expression to reproduce the extraction."""

    def __init__(self, blueprint: ExtractionBlueprint, client: httpx.Client | None = None) -> None:
        """Bind a blueprint to an httpx client; a default client is created and owned if none is supplied."""
        self.blueprint = blueprint
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)
        self._owns_client = client is None

    @classmethod
    def from_json(cls, data: str | bytes, client: httpx.Client | None = None) -> BlueprintRunner:
        """Build a runner from serialized blueprint JSON - file contents, a DB column, an API payload."""
        return cls(ExtractionBlueprint.model_validate_json(data), client)

    def fetch_responses(self, variables: dict[str, str] | None = None) -> dict[str, Any]:
        """Fetch every source endpoint and return parsed bodies keyed by the expression's source names.

        Provided variables are merged into each endpoint's query string, covering the common
        pagination/filter case; path-segment variables fall back to the URL captured at blueprint time.
        Bodies are parsed with the same detector used during capture, so replay matches the original.
        """
        responses: dict[str, Any] = {}
        for name, req in self.blueprint.endpoints.items():
            url = merge_query(req.url, variables) if variables else req.url
            try:
                resp = self._client.request(req.method, url)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise BlueprintError(f"fetch failed for {name!r} ({url}): {e}") from e
            kind, parsed, err = detect(resp.text)
            if err or kind in (BodyKind.EMPTY, BodyKind.UNPARSEABLE):
                raise BlueprintError(f"could not parse response for {name!r} ({url}): {err or kind}")
            responses[name] = parsed
        return responses

    def run(self, variables: dict[str, str] | None = None) -> list[Any]:
        """Fetch and extract, returning the raw JMESPath result (a list of records)."""
        result, err = run_extract(self.blueprint.expression, self.fetch_responses(variables))
        if err:
            raise BlueprintError(err)
        return cast("list[Any]", result)

    def run_validated(self, model: type[BaseModel], variables: dict[str, str] | None = None) -> list[BaseModel]:
        """Fetch, extract, and validate each record against model, returning typed instances.

        The blueprint stores only the target JSON Schema, not the original Python type, so callers that
        want typed objects supply the model class here.
        """
        validated, err, _ = validate_extraction(self.blueprint.expression, self.fetch_responses(variables), model)
        if err:
            raise BlueprintError(err)
        assert validated is not None  # validate_extraction returns a list whenever err is None
        return validated

    def close(self) -> None:
        """Close the underlying client if this runner created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> BlueprintRunner:
        """Enter a context manager that closes an owned client on exit."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close an owned client when leaving the context."""
        self.close()
