"""Extract upcoming events from the Goethe Institut USA events page into a typed list[Event]."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, RootModel
from rich.console import Console

from agent.scraping_agent import run_extraction
from agent.types import NoResource
from blueprints import run_blueprint

# Load ANTHROPIC_API_KEY (and any other settings) from a local .env, as documented in the README.
load_dotenv()

console = Console()

BLUEPRINT_PATH = Path(__file__).parent / "goethe_events_blueprint.json"
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


# The target is an explicit list-of-events shape: a RootModel makes "many events" a first-class,
# named pydantic type, so the agent's verified output and the blueprint's stored schema are an array.
class EventList(RootModel[list[Event]]):
    pass


async def main() -> None:
    # Passing EventList tells the pipeline the result is a list of events; run_extraction only returns
    # a blueprint once the agent's output validates against that exact shape.
    blueprint = await run_extraction(url=URL, goal=GOAL, target_type=EventList, headless=False)

    if isinstance(blueprint, NoResource):
        console.print(f"[red]No result:[/red] {blueprint.reason}")
        return

    console.print(f"[bold green]Success[/bold green] [dim]expression:[/dim] {blueprint.expression}")
    for name, ep in blueprint.endpoints.items():
        console.print(f"[dim]endpoint {name}:[/dim] {ep.url}")
    for vs in blueprint.variable_specs:
        console.print(f"[dim]var {vs.name} ({vs.location}):[/dim] {vs.description}")

    # Persist the blueprint as JSON - this is the durable artifact you can commit, store, and replay later
    # without ever calling a model again. A checked-in copy lives next to this script for reference.
    BLUEPRINT_PATH.write_text(blueprint.model_dump_json(indent=2))
    console.print(f"[dim]wrote blueprint to[/dim] {BLUEPRINT_PATH}")

    # A blueprint is a plain pydantic model - run it directly (or reload that JSON with run_blueprint later).
    # run_blueprint re-checks EventList against the blueprint's schema, then returns a typed EventList.
    events = run_blueprint(blueprint, EventList)
    console.print(f"\n[bold]{len(events.root)} events[/bold]")
    for event in events.root:
        parts = [event.date, event.title, event.city or "", event.event_type or ""]
        console.print("  " + " | ".join(p for p in parts if p))


if __name__ == "__main__":
    asyncio.run(main())
