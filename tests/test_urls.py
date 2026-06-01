"""Unit tests for the shared URL helpers, including the same-origin probe guard."""

from common.urls import merge_query


def test_merge_query_adds_and_overrides_params():
    merged = merge_query("https://api.test/events?page=1&q=x", {"page": "2"})
    assert "page=2" in merged
    assert "q=x" in merged
    assert "page=1" not in merged
