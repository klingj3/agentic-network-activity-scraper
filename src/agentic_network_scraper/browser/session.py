"""Playwright browser session: network recording, scroll-triggered lazy loading, and interactable annotation."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from playwright.sync_api import Page, Response, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from agentic_network_scraper.common.log import LogEvent, log_signal

from .annotate import InteractableElement, annotate, discover_interactables
from .events import JsonNetworkEvent
from .overlay import Overlay

# A current desktop-Chrome identity. Sending a realistic UA plus the matching
# client-hint / fetch-metadata headers below keeps us out of the trivially-blocked
# "headless automation" bucket. This layer does not rate-limit or consult robots.txt;
# honoring each site's terms of service and crawl limits is the caller's responsibility.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
_DEFAULT_HEADERS = {
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9," "image/avif,image/webp,image/apng,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
}
# Chromium exposes navigator.webdriver=true under automation; many bot filters key
# off it, so we restore the default-browser value before any page script runs.
_STEALTH_INIT_JS = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"


class BrowserSession:
    """A browser session exposing the Playwright page alongside network recording and inspection utilities.

    Construction is cheap; the browser is brought up on a dedicated single thread when the session is
    entered as an async context manager, so every Playwright call shares one thread for its lifetime::

        async with BrowserSession(url, headless=False) as session:
            events = await session.submit(session.collect_json)

    Blocking methods (``collect_json``, ``paint_interactable``, ``fetch_url`` …) run on that thread; call
    them through ``submit`` from async code.
    """

    def __init__(self, url: str, headless: bool = True, wait_ms: int = 5000) -> None:
        self._url = url
        self._headless = headless
        self._wait_ms = wait_ms
        self._json_events: list[JsonNetworkEvent] = []
        self._executor: ThreadPoolExecutor | None = None
        self._overlay_pump: asyncio.Task[None] | None = None

    @property
    def base_url(self) -> str:
        """The URL this session was opened on; used to confine model-driven probes to the same origin."""
        return self._url

    async def __aenter__(self) -> BrowserSession:
        """Bring the browser up on its own thread; in headed mode, mirror console logs onto the page overlay."""
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._start)
        if not self._headless:
            # blinker holds the receiver weakly; we also disconnect explicitly on exit.
            log_signal.connect(self._on_log_event)
            self._overlay_pump = asyncio.create_task(self._pump_overlay())
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Stop the overlay mirror and tear the browser down on its own thread."""
        if self._overlay_pump is not None:
            log_signal.disconnect(self._on_log_event)
            self._overlay_pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._overlay_pump
        if self._executor is not None:
            await self.submit(self._teardown)
            self._executor.shutdown(wait=True)

    async def submit(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking session method on the dedicated browser thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

    def _start(self) -> None:
        """Blocking bring-up (browser thread): launch Chromium, navigate, and scroll to trigger lazy content."""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._context = self._browser.new_context(
            user_agent=_USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1280, "height": 800},
            extra_http_headers=_DEFAULT_HEADERS,
        )
        self._context.add_init_script(_STEALTH_INIT_JS)
        self.page: Page = self._context.new_page()
        self.page.on("response", self._on_response)
        self._overlay = Overlay(self.page)
        self.page.goto(self._url)
        try:
            self.page.wait_for_load_state("networkidle", timeout=self._wait_ms)
        except PlaywrightTimeoutError:
            # Busy pages may never reach network idle; fall back to a fixed settle wait. A real
            # navigation failure (page crash, etc.) is not a timeout and is left to propagate.
            self.page.wait_for_timeout(self._wait_ms)
        self.scroll_to_bottom_and_back()

    def _teardown(self) -> None:
        """Blocking teardown (browser thread)."""
        self._browser.close()
        self._playwright.stop()

    def _on_response(self, response: Response) -> None:
        """Capture the body of any JSON response into the event log."""
        headers = dict(response.headers)
        if "application/json" not in headers.get("content-type", ""):
            return
        try:
            text = response.text()
            json.loads(text)
        except Exception:
            return
        self._json_events.append(
            JsonNetworkEvent(
                url=response.url,
                method=response.request.method,
                status=response.status,
                content=text,
            )
        )

    def collect_json(self) -> list[JsonNetworkEvent]:
        """Snapshot of every captured JSON API response with its body."""
        return list(self._json_events)

    def clear(self) -> None:
        """Discard recorded JSON events (e.g. before a click that should surface only new traffic)."""
        self._json_events.clear()

    def scroll_to_bottom_and_back(self, step_px: int = 400, step_ms: int = 150, max_steps: int = 100) -> None:
        """Scroll to the page bottom in increments to trigger lazy loading, then return to top.

        Capped at max_steps increments so an infinite-scroll page (whose height keeps growing as
        content loads) can never spin this loop forever.
        """
        for _ in range(max_steps):
            if not self.page.evaluate("window.scrollY + window.innerHeight < document.body.scrollHeight"):
                break
            self.page.evaluate(f"window.scrollBy(0, {step_px})")
            self.page.wait_for_timeout(step_ms)
        self.page.evaluate("window.scrollTo(0, 0)")
        self.page.wait_for_timeout(500)

    def fetch_url(self, url: str) -> tuple[int, str | None]:
        """Fetch url via the session's context (shares cookies). Returns (status, body_text); status 0 on error."""
        try:
            resp = self._context.request.get(url, headers={"Accept": "application/json"})
            return resp.status, resp.text()
        except Exception as e:
            return 0, str(e)

    def paint_interactable(self) -> tuple[bytes, list[InteractableElement]]:
        """Screenshot the page with all interactables annotated, hiding the overlay so it can't leak into vision."""
        ratio, elements = discover_interactables(self.page)
        self.set_overlay_visible(False)
        try:
            screenshot = self.page.screenshot(full_page=True)
        finally:
            self.set_overlay_visible(True)
        return annotate(screenshot, elements, ratio), elements

    def set_overlay_visible(self, visible: bool) -> None:
        """Toggle the status overlay; kept hidden during screenshots and clicks so it can't leak or intercept."""
        self._overlay.set_visible(visible)

    def _on_log_event(self, event: LogEvent) -> None:
        """Listener for the console log signal: buffer each line for the overlay pump (thread-safe)."""
        self._overlay.record(event.color, event.tag, event.msg[:600])

    async def _pump_overlay(self, interval: float = 0.25) -> None:
        """Render buffered log lines onto the live page whenever they change, from the browser thread."""
        last: tuple[tuple[str, str, str], ...] | None = None
        while True:
            await asyncio.sleep(interval)
            lines = self._overlay.snapshot()
            if lines and lines != last:
                last = lines
                await self.submit(self._overlay.render, lines)


async def _demo() -> None:
    """Smoke test: open a page, capture its JSON traffic, and show the annotated screenshot."""
    import io

    from PIL import Image

    url = "https://www.goethe.de/ins/us/en/ver.cfm"
    print(f"Opening {url} ...")
    async with BrowserSession(url, headless=True) as session:
        png, elements = await session.submit(session.paint_interactable)
        events = await session.submit(session.collect_json)
        print(f"Page loaded - {len(elements)} interactables painted, {len(events)} JSON responses captured.")
        for ev in events:
            print(f"  {ev.method} {ev.status} {ev.url}")
        Image.open(io.BytesIO(png)).show()
        print("\nScreenshot displayed.")


if __name__ == "__main__":
    asyncio.run(_demo())
