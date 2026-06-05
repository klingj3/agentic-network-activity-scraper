"""Tests for BlueprintRunner using httpx.MockTransport - real HTTP plumbing, canned responses."""

import httpx
import pytest
from pydantic import BaseModel

from agentic_network_scraper.agent.runner import BlueprintError, BlueprintRunner
from agentic_network_scraper.agent.types import ExtractionBlueprint
from agentic_network_scraper.browser.types import CapturedRequest


class Event(BaseModel):
    title: str
    date: str


def make_blueprint(url: str = "https://api.test/events?page=1") -> ExtractionBlueprint:
    return ExtractionBlueprint(
        target_url="https://site.test",
        goal="upcoming events",
        endpoints={"events": CapturedRequest(method="GET", url=url, url_template=url)},
        expression="events.data[*].{title: title, date: date}",
        target_schema={},
        sample_output=[],
        description="events from the feed",
        url_template=url,
        variable_specs=[],
    )


def runner_with(handler) -> BlueprintRunner:
    """Build a BlueprintRunner whose client is backed by a MockTransport handler."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return BlueprintRunner(make_blueprint(), client=client)


PAYLOAD = {"data": [{"title": "Concert", "date": "2026-01-01"}, {"title": "Talk", "date": "2026-02-01"}]}


def test_run_fetches_and_applies_expression():
    runner = runner_with(lambda req: httpx.Response(200, json=PAYLOAD))
    assert runner.run() == [
        {"title": "Concert", "date": "2026-01-01"},
        {"title": "Talk", "date": "2026-02-01"},
    ]


def test_run_validated_returns_typed_models():
    runner = runner_with(lambda req: httpx.Response(200, json=PAYLOAD))
    events = runner.run_validated(Event)
    assert all(isinstance(e, Event) for e in events)
    assert events[0].title == "Concert"


def test_variables_merge_into_query_string():
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json=PAYLOAD)

    runner_with(handler).run(variables={"page": "2"})
    assert "page=2" in seen["url"]


def test_http_error_becomes_blueprint_error():
    runner = runner_with(lambda req: httpx.Response(500, json={}))
    with pytest.raises(BlueprintError):
        runner.run()


def test_from_json_round_trips():
    runner = BlueprintRunner.from_json(make_blueprint().model_dump_json())
    assert runner.blueprint.expression.startswith("events.data")
