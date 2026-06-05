"""Unit tests for the capture engine's body classification, URL masking, and dedup."""

from agentic_network_scraper.browser.engine import CaptureEngine, _mask_url, _sample, _shape, detect
from agentic_network_scraper.browser.events import JsonNetworkEvent
from agentic_network_scraper.browser.types import BodyKind


def test_detect_json():
    kind, parsed, err = detect('{"a": 1}')
    assert kind == BodyKind.JSON
    assert parsed == {"a": 1}
    assert err is None


def test_detect_ndjson():
    kind, parsed, _ = detect('{"a": 1}\n{"a": 2}')
    assert kind == BodyKind.NDJSON
    assert parsed == [{"a": 1}, {"a": 2}]


def test_detect_sse():
    kind, parsed, _ = detect('data: {"x": 1}\n\ndata: [DONE]')
    assert kind == BodyKind.SSE
    assert parsed == [{"x": 1}]


def test_detect_jsonp():
    kind, parsed, _ = detect('callback({"a": 1});')
    assert kind == BodyKind.JSONP
    assert parsed == {"a": 1}


def test_detect_empty():
    kind, parsed, _ = detect("   ")
    assert kind == BodyKind.EMPTY
    assert parsed is None


def test_detect_unparseable():
    kind, parsed, err = detect("{not json")
    assert kind == BodyKind.UNPARSEABLE
    assert err is not None


def test_mask_url_replaces_ids():
    masked = _mask_url("https://x.test/users/12345/posts?id=98765&q=concert")
    assert "/users/{id}/posts" in masked
    assert "12345" not in masked and "98765" not in masked
    assert "q=concert" in masked


def test_shape_describes_nested_structure():
    shape = _shape({"meta": {"total": 1}, "data": [{"id": 1, "title": "x"}]})
    assert shape.root_type == "object"
    assert set(shape.keys) == {"meta", "data"}
    item = shape.fields["data"].item_shape
    assert set(item.keys) == {"id", "title"}


def test_sample_clips_large_payload():
    body, truncated = _sample({"data": [{"i": n} for n in range(200)]})
    assert truncated
    assert len(body["data"]) < 200


def _event(url: str, body: str) -> JsonNetworkEvent:
    return JsonNetworkEvent(
        url=url,
        method="GET",
        status=200,
        content=body,
    )


def test_ingest_dedupes_identical_responses():
    engine = CaptureEngine()
    engine.ingest([_event("https://x.test/a", '{"a": 1}'), _event("https://x.test/a", '{"a": 1}')])
    digests = engine.digests()
    assert len(digests) == 1
    assert next(iter(digests.values())).seen_count == 2
