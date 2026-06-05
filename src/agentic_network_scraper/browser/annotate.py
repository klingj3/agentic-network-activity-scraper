"""Discovery and screenshot annotation of visible interactable page elements."""

from __future__ import annotations

import io
from dataclasses import dataclass
from itertools import cycle

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page

# A 20-colour qualitative palette chosen for mutual contrast
PAINT_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
]

# Elements a user would click to trigger an API call: links, buttons, clickable
# form controls, explicit click handlers, and the ARIA roles that imply the same.
# Deliberately excludes text entry (text inputs, textarea, contenteditable) and
# media controls (audio/video) - clicking those does not surface new data.
_INTERACTABLE_SELECTOR = ", ".join(
    [
        "a[href]",
        "button:not([disabled])",
        "input[type='checkbox']:not([disabled])",
        "input[type='radio']:not([disabled])",
        "input[type='button']:not([disabled])",
        "input[type='submit']:not([disabled])",
        "select:not([disabled])",
        "details > summary",
        "[onclick]",
        "[tabindex]:not([tabindex='-1'])",
        *(
            f"[role='{role}']"
            for role in (
                "button",
                "link",
                "checkbox",
                "radio",
                "tab",
                "menuitem",
                "menuitemcheckbox",
                "menuitemradio",
                "option",
                "combobox",
                "listbox",
                "switch",
                "treeitem",
                "gridcell",
            )
        ),
    ]
)

# Discovers visible interactables, assigns a stable id where one is missing, and
# returns geometry in CSS pixels plus the device pixel ratio for screenshot scaling.
_DISCOVER_JS = """(selector) => {
    const visible = (el) => {
        const s = getComputedStyle(el);
        return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0'
            && el.offsetWidth > 0 && el.offsetHeight > 0;
    };
    const label = (el) => {
        const raw = el.tagName === 'INPUT'
            ? (el.value || el.placeholder || el.getAttribute('aria-label') || el.type)
            : (el.getAttribute('aria-label') || el.textContent || '');
        return raw.replace(/\\s+/g, ' ').trim().slice(0, 60);
    };
    const elements = [...document.querySelectorAll(selector)].filter(visible);
    return {
        ratio: window.devicePixelRatio || 1,
        elements: elements.map((el, i) => {
            if (!el.id) el.id = 'paint-' + i;
            const r = el.getBoundingClientRect();
            return {
                id: el.id,
                tag: el.tagName.toLowerCase(),
                text: label(el),
                box: [r.left + scrollX, r.top + scrollY, r.width, r.height],
            };
        }),
    };
}"""

_LABEL_FONT = ImageFont.load_default(size=13)


@dataclass
class InteractableElement:
    """One visible interactable element with its assigned paint colour and CSS-pixel geometry."""

    index: int
    id: str
    tag: str
    text: str
    color: str
    box: tuple[float, float, float, float]  # x, y, width, height in CSS pixels

    @property
    def selector(self) -> str:
        """CSS selector that uniquely targets this element for later actions."""
        return f"#{self.id}"


def discover_interactables(page: Page) -> tuple[float, list[InteractableElement]]:
    """Find visible interactables on the page, returning the device pixel ratio and elements."""
    discovered = page.evaluate(_DISCOVER_JS, _INTERACTABLE_SELECTOR)
    elements = [
        InteractableElement(
            index=i,
            color=color,
            box=tuple(item["box"]),
            id=item["id"],
            tag=item["tag"],
            text=item["text"],
        )
        for i, (item, color) in enumerate(zip(discovered["elements"], cycle(PAINT_COLORS)))
    ]
    return discovered["ratio"], elements


def annotate(screenshot: bytes, elements: list[InteractableElement], ratio: float) -> bytes:
    """Draw each element's bordered box and index label onto the screenshot, scaling CSS pixels to device pixels."""
    image = Image.open(io.BytesIO(screenshot)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for el in elements:
        x, y, w, h = (v * ratio for v in el.box)
        draw.rectangle((x, y, x + w, y + h), outline=el.color, width=2)
        caption = str(el.index)
        _, _, tw, th = draw.textbbox((0, 0), caption, font=_LABEL_FONT)
        draw.rectangle((x, y, x + tw + 4, y + th + 2), fill=el.color)
        draw.text((x + 2, y + 1), caption, fill="white", font=_LABEL_FONT)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
