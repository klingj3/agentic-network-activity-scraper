"""CaptureEngine: ingest JsonNetworkEvent objects, dedupe by content hash, and expose a RawBodyStore for the agent."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from .events import JsonNetworkEvent
from .types import (
    BodyKind,
    CapturedRequest,
    CapturedResponse,
    JsonShape,
    ResponseDigest,
)

_ID_RE = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\d+|[0-9a-f]{16,})$",
    re.I,
)

_JSONP_RE = re.compile(r"^\s*[\w$]+\s*\(", re.S)


def _mask_url(url: str) -> str:
    """Replace ID-like path segments and query values with placeholders."""
    p = urlparse(url)
    path = "/".join("{id}" if _ID_RE.match(seg) else seg for seg in p.path.split("/"))
    qs = {
        k: ["{n}" if _ID_RE.match(v) else v for v in vs] for k, vs in parse_qs(p.query, keep_blank_values=True).items()
    }
    return urlunparse((p.scheme, p.netloc, path, p.params, urlencode(qs, doseq=True), ""))


def _response_id(method: str, url_template: str, body: bytes) -> str:
    """Derive a stable 16-char hex ID from method, URL template, and body hash."""
    key = f"{method}:{url_template}:{hashlib.sha256(body).hexdigest()[:16]}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def detect(raw: str) -> tuple[BodyKind, Any, str | None]:
    """Classify body format and parse it; returns (kind, parsed_value, error_string)."""
    stripped = raw.strip()
    if not stripped:
        return BodyKind.EMPTY, None, None

    if stripped.startswith("data:"):
        frames = []
        for line in stripped.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload in ("", "[DONE]"):
                continue
            try:
                frames.append(json.loads(payload))
            except json.JSONDecodeError:
                frames.append(payload)
        return BodyKind.SSE, frames or None, None

    if _JSONP_RE.match(stripped):
        inner = re.sub(r"^\s*[\w$]+\s*\((.*)\)\s*;?\s*$", r"\1", stripped, flags=re.S)
        try:
            return BodyKind.JSONP, json.loads(inner), None
        except json.JSONDecodeError as e:
            return BodyKind.UNPARSEABLE, None, str(e)

    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) > 1:
        parsed: list[Any] = []
        for line in lines:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                break
        else:
            return BodyKind.NDJSON, parsed, None

    try:
        return BodyKind.JSON, json.loads(stripped), None
    except json.JSONDecodeError as e:
        return BodyKind.UNPARSEABLE, None, str(e)


def _shape(data: Any, depth: int = 0, max_depth: int = 4, max_keys: int = 40) -> JsonShape:
    """Recursively summarize JSON structure up to max_depth levels.

    Objects list every key in `keys`; keys whose value is itself a non-empty dict or
    array are expanded in `fields` so the agent can see nested record shapes without
    fetching raw bodies. Scalar leaves are named in `keys` but not repeated in `fields`.
    """
    if data is None:
        return JsonShape(root_type="null")
    if isinstance(data, dict):
        keys = list(data.keys())
        fields = None
        if depth < max_depth:
            nested = {
                k: _shape(v, depth + 1, max_depth, max_keys)
                for k, v in list(data.items())[:max_keys]
                if isinstance(v, (dict, list)) and v
            }
            fields = nested or None
        return JsonShape(
            root_type="object",
            keys=keys[:max_keys],
            fields=fields,
            truncated=len(keys) > max_keys,
        )
    if isinstance(data, list):
        item = _shape(data[0], depth + 1, max_depth, max_keys) if data and depth < max_depth else None
        return JsonShape(root_type="array", array_length=len(data), item_shape=item)
    return JsonShape(root_type="scalar")


def _clip(data: Any, head: int, max_str: int) -> Any:
    """Recursively keep the first `head` items of every array and clip long strings."""
    if isinstance(data, list):
        clipped = [_clip(x, head, max_str) for x in data[:head]]
        if len(data) > head:
            clipped.append(f"... +{len(data) - head} more items")
        return clipped
    if isinstance(data, dict):
        return {k: _clip(v, head, max_str) for k, v in data.items()}
    if isinstance(data, str) and len(data) > max_str:
        return data[:max_str] + "..."
    return data


def _sample(data: Any, max_chars: int = 2000, head: int = 2, max_str: int = 300) -> tuple[Any, bool]:
    """Return a size-bounded view of data: the whole value if it serializes small, else a clipped slice.

    Returns (sample, truncated). When the full body exceeds max_chars, arrays are reduced to their
    first `head` items and long strings are clipped, so the model sees representative real values
    (field formats, example contents) without ingesting the entire body.
    """
    if len(json.dumps(data, ensure_ascii=False, default=str)) <= max_chars:
        return data, False
    return _clip(data, head, max_str), True


class RawBodyStore:
    """In-process store for parsed JSON bodies. Never serialized into model context."""

    def __init__(self) -> None:
        """Initialize empty parsed-body and metadata maps."""
        self._parsed: dict[str, Any] = {}
        self._meta: dict[str, CapturedResponse] = {}

    def put(self, cap: CapturedResponse, parsed: Any) -> None:
        """Store a parsed response body alongside its metadata."""
        self._parsed[cap.response_id] = parsed
        self._meta[cap.response_id] = cap

    def get_parsed(self, response_id: str) -> Any:
        """Return the parsed body for a given response_id."""
        return self._parsed[response_id]

    def shape(self, response_id: str) -> JsonShape:
        """Return the structural shape summary for a given response_id."""
        return _shape(self._parsed[response_id])

    def sample(self, response_id: str) -> tuple[Any, bool]:
        """Return a size-bounded representative view of a parsed body and whether it was clipped."""
        return _sample(self._parsed[response_id])

    def captured(self, response_id: str) -> CapturedResponse:
        """Return the CapturedResponse metadata for a given response_id."""
        return self._meta[response_id]


class CaptureEngine:
    """Ingests JsonNetworkEvent objects from a BrowserSession into a RawBodyStore."""

    def __init__(self) -> None:
        """Initialize an empty store and deduplication index."""
        self.store = RawBodyStore()
        self._digests: dict[str, ResponseDigest] = {}

    def ingest(self, events: list[JsonNetworkEvent]) -> None:
        """Process a list of JsonNetworkEvent objects, skipping duplicates."""
        for ev in events:
            self._ingest_one(ev)

    def _ingest_one(self, ev: JsonNetworkEvent) -> None:
        """Parse, fingerprint, and store a single network event; increment seen_count for duplicates."""
        url_template = _mask_url(ev.url)
        body_kind, parsed, parse_error = detect(ev.content)
        raw_bytes = ev.content.encode()
        rid = _response_id(ev.method, url_template, raw_bytes)

        if rid in self._digests:
            d = self._digests[rid]
            self._digests[rid] = d.model_copy(update={"seen_count": d.seen_count + 1})
            return

        self.store.put(
            CapturedResponse(
                response_id=rid,
                request=CapturedRequest(
                    method=ev.method,
                    url=ev.url,
                    url_template=url_template,
                ),
                status=ev.status,
                body_kind=body_kind,
                size_bytes=len(raw_bytes),
                parse_error=parse_error,
            ),
            parsed,
        )
        self._digests[rid] = ResponseDigest(
            response_id=rid,
            method=ev.method,
            url_template=url_template,
            status=ev.status,
            body_kind=body_kind,
            size_bytes=len(raw_bytes),
            parse_error=parse_error,
        )

    def digests(self) -> dict[str, ResponseDigest]:
        """Return a snapshot of all ingested response digests keyed by response_id."""
        return dict(self._digests)
