"""Live end-to-end test against the real Goethe Institut events page.

Hits the network and the Anthropic API, so it is excluded from the default run.
Run it explicitly with: pytest -m integration
"""

import os

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from agentic_network_scraper.agent.scraping_agent import run_extraction
from agentic_network_scraper.agent.types import ExtractionBlueprint

load_dotenv()

URL = "https://www.goethe.de/ins/us/en/ver.cfm"
GOAL = (
    "Get all upcoming events. For each event extract: title, start date, city, "
    "event type or category, and registration/detail URL."
)


class Event(BaseModel):
    title: str
    date: str
    city: str | None = None
    event_type: str | None = None
    url: str | None = None


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY")
async def test_identifies_fetchevents_endpoint():
    result = await run_extraction(url=URL, goal=GOAL, target_type=Event)

    assert isinstance(result, ExtractionBlueprint), getattr(result, "reason", result)
    endpoint_urls = [ep.url for ep in result.endpoints.values()]
    assert any("fetchEvents" in url for url in endpoint_urls), endpoint_urls
    assert result.sample_output
