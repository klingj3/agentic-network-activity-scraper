# agentic-network-scraper

## Background

While computer use continues to expand, and vision models continue to improve their abilities to translate information
contained in images of text (e.g. websites) into coherent forms, for any kind of repeated scraping this is not a
cost effective or reliably repeatable approach.

Observing the fairly well-trodden landscape of AI-driven web scrapers, nothing seemed to do what I needed: scrape
once into a repeatable form for cost-free re-use, and lean on a page's network activity for more reliable, API-driven
scraping rather than re-reading rendered content on every run.

This project fills those two gaps for a project of my own -- maybe not an especially common use case, but putting it
up in case anyone else has similar needs.

## Overview

It generally operates as follows - a Pydantic-AI based agent:
* Is given a URL to open, a Pydantic shape for the output, and a goal. E.g. "Extract all the events on this website"
* Opens the URL in a browser (and waits a few seconds for network activity to settle)
* Evaluates the JSON-producing network calls, glancing at some top-level features for promising items (schema, arguments, etc)
* Deliberately introspects some of these deeper if their surface level features seem promising
* Experiments with extra args or parameters to find the best set
* If the surfaced calls look incomplete, it can fall back to a vision pass: screenshot the page, identify the
  interactable elements (tabs, filters, load-more buttons), and click one to try to trigger the additional API
  calls hiding behind it, then re-inspect whatever new responses that surfaces
* Produces an `ExtractionBlueprint` object, containing the source API endpoint(s), the URL variables it
discovered (name, location, examples, and whether each is required), and most crucially a **JMESPath
expression** that maps the responses to your target Pydantic schema.

From this, you now have a durable and cost-free method of running / re-running the driver of scraped web material for
use as you see fit!

Of course, many websites have server-side rendering for which this approach is useless, but this project serves as a
reasonable first pass before falling back to more traditional scraping methodology.

## Differentiation

There are plenty of good AI-driven scrapers around, and the constituent ideas here are not new -- but each of the
adjacent tools I looked at takes a different path and stops short of what I wanted:

* [Firecrawl](https://www.firecrawl.dev/) renders the page to clean markdown/HTML and runs an LLM extraction pass on
  every call. It's excellent at that, but it returns *data* and re-pays the model each run -- there's no durable
  artifact to replay for free.
* [scrapegraph-ai](https://github.com/ScrapeGraphAI/Scrapegraph-ai) feeds the rendered *DOM* to a model in a graph
  pipeline. Its `ScriptCreatorGraph` can even emit a reusable scraper, which is the closest thing to what I'm after
  -- but it's generated HTML-scraping code, which is exactly the "execute AI-written code" path I deliberately avoid
  (a JMESPath blueprint is declarative data, not code to run).
* [browser-use](https://github.com/browser-use/browser-use) gives an agent vision-and-DOM control of a browser to
  carry out a task. It's powerful and general, but it's computer-use at its core -- the model is in the loop on every
  step of every run, which is the per-run cost and reliability problem this project is reacting to.
* Reverse-engineering a site's JSON API by hand in the devtools Network tab is the well-known *manual* version of
  this. That's really the technique this project automates -- and then freezes into a declarative, model-free
  blueprint you can re-run.

So the niche this fills is narrow but, as far as I can tell, unoccupied: have an agent discover the underlying JSON
endpoint, probe its parameters, and compile a one-time **blueprint** (endpoint + variables + JMESPath) that replays
deterministically with no further model calls and no executing of generated code.

## Setup

```bash
uv sync --dev

cp .env.sample .env
# add your ANTHROPIC_API_KEY to .env
```

## Running the example

```bash
uv run python scripts/examples/basic_run.py
```

Extracts upcoming events from the Goethe Institut USA events page. It opens a visible browser to capture network traffic, and on a successful run prints the discovered API endpoint(s), the JMESPath expression, and the discovered URL variables. It writes the resulting blueprint to `scripts/examples/goethe_events_blueprint.json`, then replays that blueprint with `run_blueprint` to fetch the events live and prints them, demonstrating the durable, model-free re-use the blueprint enables.

A checked-in copy of that artifact lives at [`scripts/examples/goethe_events_blueprint.json`](scripts/examples/goethe_events_blueprint.json) so you can see the endpoint, variables, and JMESPath expression a run produces without running it yourself.

## Writing your own extraction

```python
import asyncio
from pydantic import BaseModel, RootModel
from agent.scraping_agent import run_extraction
from agent.types import ExtractionBlueprint, NoResource

class Product(BaseModel):
    name: str
    price: float
    sku: str | None = None

# Pass the full result shape. A RootModel makes "a list of products" a first-class pydantic type,
# so the agent's verified output and the blueprint's stored schema are an array of Product.
class ProductList(RootModel[list[Product]]):
    pass

result = asyncio.run(run_extraction(
    url="https://example.com/products",
    goal="Get all products with name, price, and SKU.",
    target_type=ProductList,
))

if isinstance(result, ExtractionBlueprint):
    print(result.expression)      # reusable JMESPath expression
    print(result.endpoints)       # source request(s) keyed by the names the expression uses
    print(result.variable_specs)  # discovered URL variables you can vary on re-fetch
    print(result.sample_output)   # validated sample rows
```

## Re-running a blueprint

A blueprint is a plain pydantic model, so it can be persisted with `model_dump_json()` and reloaded later -
no model calls, no browser. `run_blueprint` re-fetches the endpoints and applies the stored expression,
returning a typed instance of the same shape you extracted with:

```python
from blueprints import run_blueprint

# `blueprint` is the ExtractionBlueprint from above, or its JSON reloaded from disk / a database.
products = run_blueprint(blueprint, ProductList)
for product in products.root:
    print(product.name, product.price)

# Pass `variables` to drive discovered query parameters - e.g. pagination or filters.
page_2 = run_blueprint(blueprint, ProductList, variables={"page": "2"})
```

`run_blueprint` re-checks the target type against the schema the blueprint was verified with, so a drifted
model fails loudly instead of silently dropping fields. For untyped access (raw JMESPath output) or to share
a single `httpx.Client`, drop down to `BlueprintRunner` in `agent/runner.py`.

## Project layout

```
src/
  agent/
    scraping_agent.py    — agent definition and run_extraction() entry point
    tools.py             — all agent tools + AgentDeps + vision helper
    types.py             — shared Pydantic models (ExtractionBlueprint, NoResource, …) and agent signals
    extraction.py        — JMESPath extraction runner + target-schema validation
    runner.py            — BlueprintRunner: re-fetch a saved blueprint's endpoints and re-extract (untyped)
    constants.py         — model tier config (Sonnet for the extraction loop, Opus for vision)
  blueprints/
    execute.py           — run_blueprint(): typed, schema-checked front door over BlueprintRunner
  browser/
    session.py           — Playwright session: network recording, lazy-load scrolling, interactable annotation
    engine.py            — CaptureEngine: deduplication, parsing, shape inference
    annotate.py          — discover visible interactables and paint them onto screenshots for the vision pass
    overlay.py           — in-page status panel that mirrors the console log onto the live browser
    events.py            — JsonNetworkEvent dataclass
    types.py             — Pydantic models for the capture layer
  common/
    log.py               — rich console logger + log signal, shared by both layers (depends on neither)
    urls.py              — query-string merge helper shared by capture and blueprint replay
scripts/
  examples/
    basic_run.py         — Goethe Institut events: extract a blueprint, then replay it with run_blueprint
```

## Tests

```bash
uv run pytest                  # unit tests: extraction, urls, engine, runner, blueprints
uv run pytest -m integration   # live end-to-end run (hits the network + Anthropic API, needs a key)
```

The integration test is marked and excluded from the default run, so the unit suite stays fast and
offline; it exercises the real Goethe Institut page when you opt in.
