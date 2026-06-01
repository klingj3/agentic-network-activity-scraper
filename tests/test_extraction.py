"""Unit tests for the JMESPath extraction runner and schema validation - the self-verifying core."""

from pydantic import BaseModel

from agent.extraction import validate_extraction


class Event(BaseModel):
    title: str
    date: str


RESPONSES = {"events": {"data": [{"title": "Concert", "date": "2026-01-01"}]}}


def test_good_expression_validates():
    validated, err, error_type = validate_extraction("events.data[*].{title: title, date: date}", RESPONSES, Event)
    assert err is None
    assert error_type is None
    assert validated is not None and len(validated) == 1
    assert validated[0].title == "Concert"


def test_invalid_syntax_is_jmespath_error():
    validated, err, error_type = validate_extraction("events.data[*].{title: title", RESPONSES, Event)
    assert validated is None
    assert error_type == "jmespath"


def test_schema_mismatch_is_validation_error():
    validated, err, error_type = validate_extraction("events.data[*].{name: title}", RESPONSES, Event)
    assert validated is None
    assert error_type == "validation"


def test_no_matches_is_empty():
    validated, err, error_type = validate_extraction(
        "events.data[?title == 'nope'].{title: title, date: date}", RESPONSES, Event
    )
    assert validated is None
    assert error_type == "empty"


def test_validation_error_is_deduped_across_items():
    responses = {"events": {"data": [{"title": f"E{i}"} for i in range(3)]}}
    _, err, error_type = validate_extraction("events.data[*].{name: title}", responses, Event)
    assert error_type == "validation"
    # Three items, each missing the same fields, collapse to counted lines naming the produced key.
    assert "(x3)" in err
    assert "produced keys: name" in err
