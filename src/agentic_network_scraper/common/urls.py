"""Small URL helpers shared between the capture agent and the blueprint runner."""

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse


def merge_query(url: str, extra: dict[str, str]) -> str:
    """Merge extra query params into url's existing query string, returning the rebuilt URL."""
    p = urlparse(url)
    qs = {k: vs[-1] for k, vs in parse_qs(p.query, keep_blank_values=True).items()}
    qs.update(extra)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(qs), ""))
