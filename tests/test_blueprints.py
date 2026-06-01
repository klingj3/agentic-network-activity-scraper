"""Tests for blueprints.run_blueprint - typed blueprint execution with a schema-match guard."""

import httpx
import pytest
from pydantic import BaseModel, RootModel

from agent.types import ExtractionBlueprint
from blueprints import BlueprintTypeMismatch, run_blueprint
from browser.types import CapturedRequest


class Event(BaseModel):
    title: str
    date: str


class EventList(RootModel[list[Event]]):
    pass


class Drifted(BaseModel):
    title: str
    when: str


class DriftedList(RootModel[list[Drifted]]):
    pass


def make_blueprint(target_type: type[BaseModel] = EventList) -> ExtractionBlueprint:
    url = "https://api.test/events?page=1"
    return ExtractionBlueprint(
        target_url="https://site.test",
        goal="upcoming events",
        endpoints={"events": CapturedRequest(method="GET", url=url, url_template=url)},
        expression="events.data[*].{title: title, date: date}",
        target_schema=target_type.model_json_schema(),
        sample_output=[],
        description="events from the feed",
        url_template=url,
        variable_specs=[],
    )


PAYLOAD = {"data": [{"title": "Concert", "date": "2026-01-01"}, {"title": "Talk", "date": "2026-02-01"}]}


def mock_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=PAYLOAD)))


def test_returns_the_root_model_shape():
    events = run_blueprint(make_blueprint(), EventList, client=mock_client())
    assert isinstance(events, EventList)
    assert all(isinstance(e, Event) for e in events.root)
    assert events.root[0].title == "Concert"


def test_accepts_serialized_blueprint_json():
    events = run_blueprint(make_blueprint().model_dump_json(), EventList, client=mock_client())
    assert [e.date for e in events.root] == ["2026-01-01", "2026-02-01"]


def test_mismatched_type_raises_before_any_fetch():
    # DriftedList's schema differs from the blueprint's, so this must fail without touching the network.
    def explode(req: httpx.Request) -> httpx.Response:
        raise AssertionError("fetch must not happen on a schema mismatch")

    client = httpx.Client(transport=httpx.MockTransport(explode))
    with pytest.raises(BlueprintTypeMismatch):
        run_blueprint(make_blueprint(), DriftedList, client=client)
