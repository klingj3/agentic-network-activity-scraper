"""In-page status overlay that mirrors the console log onto the live browser page."""

from __future__ import annotations

from collections import deque

from playwright.sync_api import Page

# Maps the logger's rich colour names to CSS so the in-page status overlay matches
# the console output. Unknown names pass through unchanged (CSS accepts most of them).
_CSS_STATUS_COLORS = {
    "cyan": "#42d4f4",
    "blue": "#6fa8ff",
    "magenta": "#f032e6",
    "green": "#3cb44b",
    "yellow": "#ffe119",
    "bright_yellow": "#ffe14d",
    "red": "#e6194b",
    "white": "#f5f5f5",
    "bright_black": "#9aa0a6",
}

# Creates the overlay once and re-renders it as a scrollable log panel mirroring the console:
# one row per recent line, a coloured tag chip beside its full (un-clamped) message. The panel is
# capped at a fraction of the viewport and scrolls natively; each render auto-follows the newest
# line unless the viewer has scrolled up to read history. pointer-events:auto lets the viewer
# scroll it (it is hidden via visibility during screenshots and clicks, so it never intercepts
# either); a maximal z-index keeps it on top. It is appended to documentElement so it survives body
# re-renders, is left out of the interactable set, and uses textContent so logged data can never
# inject markup.
_OVERLAY_JS = """(p) => {
    let el = document.getElementById('__agent_overlay__');
    if (!el) {
        el = document.createElement('div');
        el.id = '__agent_overlay__';
        el.style.cssText = 'position:fixed;top:14px;left:14px;max-width:46vw;max-height:82vh;'
            + 'overflow-y:auto;z-index:2147483647;pointer-events:auto;border-radius:10px;'
            + 'padding:10px 14px;background:rgba(17,18,23,0.82);'
            + 'backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);'
            + 'box-shadow:0 4px 20px rgba(0,0,0,0.5);'
            + 'display:flex;flex-direction:column;gap:5px;'
            + 'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
            + 'scrollbar-width:thin;scrollbar-color:rgba(255,255,255,0.25) transparent;';
        document.documentElement.appendChild(el);
    }
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    el.replaceChildren();
    for (const line of p.lines) {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;gap:8px;align-items:baseline;';
        const tag = document.createElement('span');
        tag.style.cssText = 'flex:none;min-width:96px;font-size:9.5px;font-weight:700;'
            + 'letter-spacing:0.08em;text-transform:uppercase;';
        const sep = line.tag.indexOf(':');
        if (sep !== -1) {
            const pre = document.createElement('span');
            pre.textContent = line.tag.slice(0, sep + 1);
            pre.style.color = 'rgba(255,255,255,0.35)';
            const name = document.createElement('span');
            name.textContent = line.tag.slice(sep + 1);
            name.style.color = line.color;
            tag.appendChild(pre);
            tag.appendChild(name);
        } else {
            tag.textContent = line.tag;
            tag.style.color = line.color;
        }
        const msg = document.createElement('span');
        msg.textContent = line.text;
        msg.style.cssText = 'font-size:11.5px;line-height:1.4;color:#e8e8ea;'
            + 'white-space:pre-wrap;word-break:break-word;';
        row.appendChild(tag);
        row.appendChild(msg);
        el.appendChild(row);
    }
    if (atBottom) el.scrollTop = el.scrollHeight;
    el.style.visibility = 'visible';
}"""

# Toggles overlay visibility without destroying it, to keep it out of screenshots.
_OVERLAY_VIS_JS = """(visible) => {
    const el = document.getElementById('__agent_overlay__');
    if (el) el.style.visibility = visible ? 'visible' : 'hidden';
}"""


def _css_status(color: str) -> str:
    """Translate a logger colour name to a CSS colour for the overlay."""
    return _CSS_STATUS_COLORS.get(color, color)


class Overlay:
    """Buffers (color, tag, text) log lines and renders them as a live panel on the page."""

    def __init__(self, page: Page, maxlen: int = 40) -> None:
        """Bind the overlay to a page, keeping the most recent maxlen lines."""
        self._page = page
        self._lines: deque[tuple[str, str, str]] = deque(maxlen=maxlen)

    def record(self, color: str, tag: str, text: str) -> None:
        """Thread-safe: buffer a log line for the next render."""
        self._lines.append((color, tag, text))

    def snapshot(self) -> tuple[tuple[str, str, str], ...]:
        """Return an immutable copy of the buffered lines for change detection."""
        return tuple(self._lines)

    def render(self, lines: tuple[tuple[str, str, str], ...]) -> None:
        """Render the given log lines onto the page; runs on the browser thread."""
        try:
            payload = [{"color": _css_status(c), "tag": t, "text": x} for c, t, x in lines]
            self._page.evaluate(_OVERLAY_JS, {"lines": payload})
        except Exception:
            # Best-effort cosmetic overlay: the page may be mid-navigation or closed when this
            # fires, so a failed render must never disrupt the actual scrape. Intentionally swallowed.
            pass

    def set_visible(self, visible: bool) -> None:
        """Toggle the overlay's visibility without destroying it, to keep it out of screenshots."""
        self._page.evaluate(_OVERLAY_VIS_JS, visible)
