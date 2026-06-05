"""JsonNetworkEvent dataclass representing a single captured JSON API response from the browser."""

from dataclasses import dataclass


@dataclass
class JsonNetworkEvent:
    """A captured JSON API response: request/response metadata plus the raw body text."""

    url: str
    method: str
    status: int
    content: str
