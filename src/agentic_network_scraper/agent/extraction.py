"""JMESPath extraction runner and target-schema validation for LLM-generated expressions."""

from collections import Counter
from typing import Any, Literal, get_args, get_origin

import jmespath
import jmespath.exceptions
from pydantic import BaseModel, RootModel, TypeAdapter, ValidationError

# Failure categories validate_extraction can report. run_extract only surfaces JMESPath-domain
# errors, so its failures are always "jmespath"; the schema/empty cases arise here.
ErrorType = Literal["jmespath", "validation", "empty"]


def run_extract(expression: str, responses: dict[str, Any]) -> tuple[Any, str | None]:
    """Evaluate a JMESPath expression against responses keyed by short name. Returns (result, error)."""
    try:
        result = jmespath.search(expression, responses)
    except jmespath.exceptions.JMESPathError as e:
        return None, f"jmespath: {e}"
    if result is None:
        return None, "jmespath: expression matched nothing"
    return result, None


def summarize_validation_error(exc: ValidationError) -> str:
    """Dedupe a list-of-models ValidationError, which repeats the same error per item.

    A 40-item result yields kilobytes of identical text; collapsing to one counted line per
    distinct (field, message) keeps the signal small enough to ride into retries. The trailing
    "produced keys" line names what the expression actually emitted, so the model can spot an
    output-key mismatch (e.g. emitting `start_date` when the schema field is `date`) directly.
    """
    counts: Counter[tuple[str, str]] = Counter()
    produced: set[str] = set()
    for err in exc.errors():
        counts[(".".join(str(p) for p in err["loc"][1:]) or "(item)", err["msg"])] += 1
        if isinstance(err.get("input"), dict):
            produced.update(err["input"])
    lines = [f"  {field}: {msg} (x{n})" for (field, msg), n in counts.items()]
    tail = f"\nexpression produced keys: {', '.join(sorted(produced))}" if produced else ""
    return f"{len(exc.errors())} validation errors:\n" + "\n".join(lines) + tail


def item_type(target_type: type[BaseModel]) -> Any:
    """The element type each extracted record is validated against.

    A RootModel[list[X]] target declares the list shape explicitly, so its element is X; any other
    model is itself the element (the expression yields a list of it). Either way validation runs over
    a list, since every extraction produces a collection of records.
    """
    if issubclass(target_type, RootModel):
        root = target_type.model_fields["root"].annotation
        if get_origin(root) is list:
            return get_args(root)[0]
    return target_type


def validate_extraction(
    expression: str,
    responses: dict[str, Any],
    target_type: type[BaseModel],
) -> tuple[list[Any] | None, str | None, ErrorType | None]:
    """Run an expression and validate the result as a list of target_type's item_type.

    Returns (validated_items, error, error_type). On success error is None and validated_items is a
    non-empty list; on failure validated_items is None and error is a retry-friendly message.
    error_type is "jmespath" (expression error), "validation" (output mismatched the schema), or
    "empty" (the expression ran and validated but matched no items).
    """
    result, err = run_extract(expression, responses)
    if err:
        return None, err, "jmespath"
    try:
        element = item_type(target_type)
        ta: TypeAdapter[Any] = TypeAdapter(list[element])  # type: ignore[valid-type]
        validated = ta.validate_python(result)
    except ValidationError as e:
        return None, summarize_validation_error(e), "validation"
    if not validated:
        return None, "expression matched zero items - it ran and validated but produced an empty list", "empty"
    return validated, None, None
