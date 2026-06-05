"""Network scraping agent and end-to-end extraction pipeline."""

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from agentic_network_scraper.browser.engine import CaptureEngine
from agentic_network_scraper.browser.session import BrowserSession
from agentic_network_scraper.common import log

from .constants import ModelTier, get_model
from .extraction import validate_extraction
from .tools import (
    AgentDeps,
    attempt_parse,
    click_element,
    get_interactables,
    inspect,
    list_responses,
    probe_endpoint,
    sample,
    update_state,
    with_reason,
)
from .types import Abandon, ExtractionBlueprint, ExtractionState, NoResource, Synthesize

_SYSTEM_PROMPT = """
You are an expert at extracting structured data from network API responses captured in a browser session.

Your job: find responses containing data that matches the goal, understand their structure, produce
a JMESPath expression that transforms those responses into the target schema, and document the URL
variables that allow callers to parameterize the endpoint.

## Every tool call must justify itself
Every tool takes a required `reason` argument. Set it to one short sentence explaining why you are
making this specific call right now, given the current state and goal (e.g. "events feed is the only
candidate whose shape matches the schema, so inspect it next"). Be concrete, not generic.

## Workflow
The inspection tools form a cost ladder - spend the least that answers your question, climbing only
as needed:
1. `list_responses` - metadata for every captured response (no shapes, no values). Always start here.
2. `inspect` - the nested key/array structure of one response. No values. Use to learn field names.
3. `sample` - the actual values of one response (full body if small, else a clipped slice). Use when
   you need to see real data - formats, example values - before writing an expression.
4. `attempt_parse` - test a JMESPath expression against the most promising response.
   **Limit to 3 attempts per response.** If all fail, call `update_state` with what you learned
   and move on to the next.
5. **As soon as `attempt_parse` succeeds, output `Synthesize` immediately.** Do not probe further,
   do not call more tools. A working expression is the finish line.
6. `update_state` - record a short per-endpoint relevance note (what each response is and whether it
   fits the goal) for the responses you've evaluated. These become the final report if no response
   matches. Prune stale notes freely; call it after inspecting a response and after any attempt_parse
   failures.
7. `probe_endpoint` - only use this after you have a confirmed working expression, to discover
   pagination or filter variables for the url_template. One probe per parameter set.
8. `get_interactables` + `click_element` - expensive; use only in edge cases. If you have a likely
   partial result and suspect more data sits behind a tab, filter, or load-more control, click it
   once to surface the additional API calls, then re-inspect the new responses.
9. Output `Abandon` if no captured response contains data matching the goal.

## Decision rule
- attempt_parse succeeds → output Synthesize (stop all other work)
- attempt_parse fails 3× on a response → update_state, move to the next one
- all responses exhausted → try get_interactables once, then Abandon if still nothing

## JMESPath expression contract
Your expression is evaluated as:

  jmespath.search(expression, responses)

where `responses` is a dict keyed by the short names you assign in `source_response_ids`.
The expression must produce a list of objects matching the target schema.

Examples:
  Single source:    products.data[*].{name: name, price: price}
  Multiple sources: {items: catalog.data[*].{name: name, sku: sku}, total: meta.pagination.total}
  Filtered:         events[?status == 'active'].{id: id, title: title}

## URL variable discovery contract
In your `Synthesize` output:
- Set `url_template` to the endpoint URL with {name} placeholders for each discovered variable.
- Set `variable_specs` to one entry per useful variable with name, location ("query" or "path"),
  description, example values, and required flag.
- Only include variables that observably affect the response.
- Path variables already in the captured url_template ({id}, {n}) count as "path" variables.
"""

_agent: Agent[AgentDeps, Synthesize | Abandon] = Agent(
    deps_type=AgentDeps,
    output_type=[Synthesize, Abandon],
    instructions=_SYSTEM_PROMPT,
    name="scraping_agent",
    tools=[
        with_reason(t)
        for t in (
            list_responses,
            inspect,
            sample,
            attempt_parse,
            update_state,
            get_interactables,
            click_element,
            probe_endpoint,
        )
    ],
)


# Hard ceiling on vision discovery passes per run. get_interactables runs a full-page screenshot
# through the vision model - the agent's most expensive operation - so this caps how many times it
# can surface interactables before falling back to the already-captured data.
_MAX_INTERACT_LOOPS = 3


def _write_trace(trace_dir: Path, url: str, messages_json: bytes) -> None:
    """Write the full pydantic-ai message log for a run to trace_dir as pretty JSON."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    slug = url.split("//")[-1].split("/")[0].replace(".", "-")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = trace_dir / f"{ts}-{slug}.json"
    path.write_text(json.dumps(json.loads(messages_json), indent=2))
    log.info(f"trace → {path}")


def _build_blueprint(
    signal: Synthesize,
    deps: AgentDeps,
    url: str,
    goal: str,
) -> ExtractionBlueprint | NoResource:
    """Verify the agent's expression against the captured responses and assemble the blueprint."""
    considered = list(deps.state.endpoint_notes)
    log.step("SYNTHESIZE", f"verifying {list(signal.source_response_ids.keys())}")
    try:
        responses = {name: deps.engine.store.get_parsed(rid) for name, rid in signal.source_response_ids.items()}
    except KeyError as e:
        log.err(f"unknown response_id: {e}")
        return NoResource(reason=f"Agent referenced an unknown response_id: {e}", responses_considered=considered)

    validated, err, _ = validate_extraction(signal.expression, responses, deps.target_type)
    if err:
        log.err(err.splitlines()[0][:120])
        return NoResource(reason=f"Expression failed verification: {err}", responses_considered=considered)

    assert validated is not None  # validate_extraction returns a list whenever err is None
    log.ok(f"{len(validated)} items from {len(signal.source_response_ids)} response(s)")
    return ExtractionBlueprint(
        target_url=url,
        goal=goal,
        endpoints={name: deps.engine.store.captured(rid).request for name, rid in signal.source_response_ids.items()},
        expression=signal.expression,
        target_schema=deps.target_type.model_json_schema(),
        sample_output=[m.model_dump() if isinstance(m, BaseModel) else m for m in validated[:5]],
        description=signal.description,
        url_template=signal.url_template,
        variable_specs=signal.variable_specs,
    )


async def run_extraction(
    url: str,
    goal: str,
    target_type: type[BaseModel],
    headless: bool = True,
    request_limit: int = 30,
    output_tokens_limit: int | None = 8192,
    total_tokens_limit: int | None = None,
    trace_dir: Path | None = None,
) -> ExtractionBlueprint | NoResource:
    """Capture a page's network traffic and let the agent find and verify a JMESPath extraction for the goal.

    A single agent run, bounded by pydantic-ai usage limits. Returns an ExtractionBlueprint on success or a
    NoResource otherwise; callers that want to retry should re-invoke this themselves.

    Cost controls (all cumulative across the run; pass None to disable):
    - `request_limit`: max model round-trips (tool-call budget).
    - `output_tokens_limit`: max generated tokens.
    - `total_tokens_limit`: max input + output tokens combined - the closest proxy for spend, since the
      captured JSON bodies dominate the input side.

    Pass `trace_dir` to dump the full pydantic-ai message log for the run there as JSON; tracing is
    off by default so the entry point has no implicit filesystem side effects.
    """
    log.step("RUN", f"{url}  goal={goal!r}  target={target_type.__name__}  request_limit={request_limit}")

    async with BrowserSession(url, headless=headless, wait_ms=6000) as session:
        deps = AgentDeps(
            engine=CaptureEngine(),
            session=session,
            state=ExtractionState(goal=goal),
            target_type=target_type,
            max_interact_loops=_MAX_INTERACT_LOOPS,
        )
        deps.engine.ingest(await session.submit(session.collect_json))
        log.step("OPEN", f"{len(deps.engine.digests())} responses captured")
        if not deps.engine.digests():
            return NoResource(reason="No JSON responses captured.", responses_considered=[])

        prompt = (
            f"Goal: {goal}\n\nIdentify the captured response(s) that satisfy this goal "
            "and produce a verified JMESPath extraction."
        )
        log.block("PROMPT", "bright_black", prompt)
        try:
            result = await _agent.run(
                prompt,
                model=get_model(ModelTier.LOW),
                deps=deps,
                usage_limits=UsageLimits(
                    request_limit=request_limit,
                    output_tokens_limit=output_tokens_limit,
                    total_tokens_limit=total_tokens_limit,
                ),
            )
        except UsageLimitExceeded as e:
            log.warn(f"request budget exhausted: {e}")
            return NoResource(
                reason=f"Request budget of {request_limit} exhausted.",
                responses_considered=list(deps.state.endpoint_notes),
            )

        u = result.usage
        log.info(
            f"usage: requests={u.requests}  tokens={u.total_tokens} (req={u.request_tokens} resp={u.response_tokens})"
        )
        log.dump_messages(result.all_messages())
        if trace_dir is not None:
            _write_trace(trace_dir, url, result.all_messages_json())

        signal = result.output
        log.step("EXPLORE", f"→ {signal.kind}")
        if isinstance(signal, Abandon):
            log.warn(signal.reason)
            return NoResource(reason=signal.reason, responses_considered=signal.responses_considered)
        return _build_blueprint(signal, deps, url, goal)
