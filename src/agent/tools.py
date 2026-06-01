"""Tool functions and shared AgentDeps for the network scraping agent."""

import functools
import inspect as _inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.usage import UsageLimits

from browser.engine import CaptureEngine
from browser.events import JsonNetworkEvent
from browser.session import BrowserSession
from browser.types import ResponseDigest, ResponseSample
from common import log
from common.urls import merge_query

from .constants import ModelTier, get_model
from .extraction import validate_extraction
from .types import ExtractionState, ParseAttempt, StatePatch


def with_reason(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool so it takes a required `reason` argument, logged before the tool body runs.

    Inserts a `reason: str` parameter into the tool's schema without editing each tool's own
    signature, so every call the model makes carries its one-line justification for making it.
    """

    @functools.wraps(fn)
    async def wrapper(ctx: RunContext[AgentDeps], reason: str, *args: Any, **kwargs: Any) -> Any:
        log.reason(fn.__name__, reason)
        return await fn(ctx, *args, **kwargs)

    sig = _inspect.signature(fn)
    params = list(sig.parameters.values())
    reason_param = _inspect.Parameter("reason", _inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
    wrapper.__signature__ = sig.replace(parameters=[params[0], reason_param, *params[1:]])  # type: ignore[attr-defined]
    wrapper.__annotations__ = {"reason": str, **fn.__annotations__}
    return wrapper


@dataclass
class AgentDeps:
    """Runtime state shared across all tool calls within a single agent run."""

    engine: CaptureEngine
    session: BrowserSession
    state: ExtractionState
    target_type: type[BaseModel]
    interact_count: int = 0
    max_interact_loops: int = 3


class ClickCandidate(BaseModel):
    """An element identified by the vision model as likely to expose relevant API data when clicked."""

    selector: str
    label: str
    justification: str


VISION_PROMPT = """\
You are analyzing an annotated web page screenshot to identify elements worth clicking.
The screenshot has colored numbered boxes on every interactable element.

Return only elements that, when clicked, are likely to trigger API calls exposing data
relevant to the goal - pagination, load-more, tabs, filters, expandable sections.
Ignore auth, navigation, ads, and UI chrome unrelated to the data.

Use the labeled element list to populate selector and label fields exactly as shown.
"""

_vision_agent: Agent[None, list[ClickCandidate]] = Agent(
    output_type=list[ClickCandidate],
    instructions=VISION_PROMPT,
    name="vision",
)


async def find_clickables(session: BrowserSession, goal: str) -> list[ClickCandidate]:
    """Paint the page, ask a vision model which elements are worth clicking, return structured candidates."""
    screenshot, elements = await session.submit(session.paint_interactable)
    if not elements:
        return []
    element_list = "\n".join(f"[{el.index}] <{el.tag}> selector={el.selector!r} text={el.text!r}" for el in elements)
    prompt = f"Goal: {goal}\n\nLabeled elements:\n{element_list}"
    result = await _vision_agent.run(
        [BinaryContent(data=screenshot, media_type="image/png"), prompt],
        model=get_model(ModelTier.HIGH),
        usage_limits=UsageLimits(response_tokens_limit=2048),
    )
    candidates = result.output
    log.tool("find_clickables", f"{len(candidates)} candidates from {len(elements)} elements")
    return candidates


async def list_responses(ctx: RunContext[AgentDeps]) -> list[ResponseDigest]:
    """Return all captured responses as lightweight digests (metadata only, no shapes or values). Call this first."""
    digests = list(ctx.deps.engine.digests().values())
    log.tool("list_responses", f"{len(digests)} digests")
    return digests


async def inspect(ctx: RunContext[AgentDeps], response_id: str) -> ResponseDigest | str:
    """Return metadata + nested JSON shape skeleton (max depth 4, 40 keys per object). Structure only, no values.

    Objects list every key in `keys`; nested objects/arrays are expanded under `fields`, so the
    shape of records inside arrays (e.g. the keys of each item in a results list) is visible here.
    Use these key names directly in your JMESPath expression - do not guess field names.
    Call `sample` next if you need to see the actual values behind these keys.
    """
    d = ctx.deps.engine.digests().get(response_id)
    if d is None:
        return f"unknown response_id: {response_id!r}"
    shape = ctx.deps.engine.store.shape(response_id)
    log.tool("inspect", f"{d.url_template}  {shape.root_type} keys={shape.keys or shape.array_length}")
    return d.model_copy(update={"json_shape": shape})


async def sample(ctx: RunContext[AgentDeps], response_id: str) -> ResponseSample | str:
    """Return actual values from one response: the full body if small, else a clipped representative slice.

    Use after `inspect`, when you need to see real data - field formats, example values, how nested
    records are populated - before writing an expression. Long arrays are reduced to their first few
    items and long strings clipped; `truncated` flags whether anything was dropped.
    """
    d = ctx.deps.engine.digests().get(response_id)
    if d is None:
        return f"unknown response_id: {response_id!r}"
    body, truncated = ctx.deps.engine.store.sample(response_id)
    log.tool("sample", f"{d.url_template}  truncated={truncated}")
    return ResponseSample(response_id=response_id, truncated=truncated, body=body)


async def attempt_parse(
    ctx: RunContext[AgentDeps],
    source_ids: dict[str, str],
    expression: str,
) -> ParseAttempt:
    """Test a JMESPath expression against captured responses. Returns success+sample or structured error.

    source_ids maps a short name to a response_id, e.g. {"events": "<id>"}.
    Non-terminal - call update_state to persist what you learn.
    """
    log.tool("attempt_parse", f"testing {list(source_ids.keys())}")
    try:
        responses = {name: ctx.deps.engine.store.get_parsed(rid) for name, rid in source_ids.items()}
    except KeyError as e:
        return ParseAttempt(success=False, error=f"unknown response_id: {e}", error_type="missing")

    validated, err, error_type = validate_extraction(expression, responses, ctx.deps.target_type)
    if err:
        log.tool("attempt_parse", f"failed ({error_type}): {err.splitlines()[0][:100]}")
        return ParseAttempt(success=False, error=err, error_type=error_type)

    assert validated is not None  # validate_extraction returns a list whenever err is None
    sample = [m.model_dump() if isinstance(m, BaseModel) else m for m in validated[:3]]
    log.tool("attempt_parse", f"ok - {len(validated)} items, keys={list(sample[0].keys()) if sample else []}")
    return ParseAttempt(success=True, sample_output=sample)


async def update_state(ctx: RunContext[AgentDeps], patch: StatePatch) -> str:
    """Record your running relevance notes about each response; these are summarized into the result if nothing matches.

    Omit endpoint_notes to leave it unchanged. You may prune stale entries by omitting them.
    """
    s = ctx.deps.state
    if patch.endpoint_notes is None:
        log.state("nothing changed")
        return "Updated: nothing changed"
    s.endpoint_notes = patch.endpoint_notes
    for note in patch.endpoint_notes.values():
        log.state(f"  {'[OK]' if note.relevant else '[--]'} {note.url_template}  {note.summary}")
    summary = f"endpoint_notes({len(patch.endpoint_notes)})"
    log.state(summary)
    return f"Updated: {summary}"


async def get_interactables(ctx: RunContext[AgentDeps]) -> list[ClickCandidate] | str:
    """Identify page elements likely to expose new API data when clicked.

    Returns a ranked list of candidates with selectors and justifications.
    Pass a selector to `click_element` to act on a candidate.
    """
    if ctx.deps.interact_count >= ctx.deps.max_interact_loops:
        return f"Interact budget exhausted ({ctx.deps.max_interact_loops} discovery passes used)."
    # Count this against the budget: find_clickables runs a full-page screenshot through the vision
    # model, the single most expensive operation in a run, so it must be capped like clicking is.
    ctx.deps.interact_count += 1
    candidates = await find_clickables(ctx.deps.session, ctx.deps.state.goal)
    if not candidates:
        return "No actionable elements found on the current page."
    return candidates


async def probe_endpoint(
    ctx: RunContext[AgentDeps],
    url: str,
    params: dict[str, str],
) -> str:
    """Fetch url with additional query params, ingest the response, and return a status summary.

    Merges params into the existing query string. The new response is ingested into the engine
    and will appear in list_responses. Call this once per parameter set you want to test.
    Use to discover which query params meaningfully change the response before populating
    url_template and variable_specs in your Synthesize output.
    """
    target = merge_query(url, params)
    log.tool("probe_endpoint", f"{target}")

    status, body = await ctx.deps.session.submit(ctx.deps.session.fetch_url, target)
    if not body or status == 0:
        return f"probe failed: status={status} error={body or 'no response'}"

    event = JsonNetworkEvent(
        url=target,
        method="GET",
        status=status,
        content=body,
    )
    ctx.deps.engine.ingest([event])
    log.tool("probe_endpoint", f"status={status}  size={len(body)}B ingested")
    return f"probed {target!r}: status={status} size={len(body)}B - call list_responses to see result"


def _sync_click(session: BrowserSession, selector: str) -> list[JsonNetworkEvent]:
    """Sync: clear buffer, click, wait for network idle, scroll, collect new events."""
    session.clear()
    # Hide the status overlay across the click so it can never intercept Playwright's pointer
    # hit-testing if it happens to sit over the target element.
    session.set_overlay_visible(False)
    try:
        session.page.click(selector)
    finally:
        session.set_overlay_visible(True)
    session.page.wait_for_load_state("networkidle", timeout=5000)
    session.scroll_to_bottom_and_back()
    return session.collect_json()


async def click_element(ctx: RunContext[AgentDeps], selector: str) -> str:
    """Click an element by CSS selector, wait for network activity, and ingest any new JSON responses.

    Get valid selectors from get_interactables. Call list_responses after to see new responses.
    """
    if ctx.deps.interact_count >= ctx.deps.max_interact_loops:
        return f"Interact budget exhausted ({ctx.deps.max_interact_loops} discovery passes used)."
    try:
        events = await ctx.deps.session.submit(_sync_click, ctx.deps.session, selector)
    except Exception as e:
        return f"Click on {selector!r} failed: {e}"
    ctx.deps.engine.ingest(events)
    log.tool("click_element", f"{selector!r} → {len(events)} new responses")
    if not events:
        return f"Clicked {selector!r} - no new JSON responses captured."
    return f"Clicked {selector!r} - {len(events)} new JSON responses ingested. " f"Call list_responses to see them."
