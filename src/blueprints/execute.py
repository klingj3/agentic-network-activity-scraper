"""Run a stored blueprint against live endpoints and get back a fully typed result.

A blueprint persists only its target JSON Schema, not the pydantic class it was built from - a model class
does not round-trip through JSON. The caller therefore re-supplies that class here. We assert its schema
matches the one the blueprint was verified against, so a drifted type fails loudly instead of silently
dropping or coercing fields, then return an instance of target_type. For the common many-records case
target_type is a RootModel[list[Item]], so the result is that root model and its records are result.root.
This is the typed front door over BlueprintRunner, which does the untyped fetch-and-extract work.
"""

from __future__ import annotations

from typing import TypeVar

import httpx
from pydantic import BaseModel

from agent.runner import BlueprintRunner
from agent.types import ExtractionBlueprint

T = TypeVar("T", bound=BaseModel)


class BlueprintTypeMismatch(RuntimeError):
    """Raised when target_type's JSON Schema differs from the schema stored in the blueprint."""


def run_blueprint(  # noqa: UP047
    blueprint: ExtractionBlueprint | str | bytes,
    target_type: type[T],
    variables: dict[str, str] | None = None,
    client: httpx.Client | None = None,
) -> T:
    """Fetch and extract a blueprint into a target_type instance, asserting the type matches its schema.

    Accepts a parsed ExtractionBlueprint or its serialized JSON (file contents, a DB column, an API payload).
    For the usual list extraction pass a RootModel[list[Item]] and read the records off result.root.
    variables are merged into each endpoint's query string for pagination/filtering, exactly as
    BlueprintRunner.run_validated handles them. Raises BlueprintTypeMismatch before any network call if
    target_type has drifted from the schema the blueprint was built and verified against.
    """
    if isinstance(blueprint, (str, bytes)):
        blueprint = ExtractionBlueprint.model_validate_json(blueprint)
    if target_type.model_json_schema() != blueprint.target_schema:
        raise BlueprintTypeMismatch(
            f"{target_type.__name__} schema does not match the schema this blueprint was verified against"
        )
    with BlueprintRunner(blueprint, client) as runner:
        records = runner.run_validated(target_type, variables)
        return target_type.model_validate(records)
