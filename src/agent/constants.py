"""AI model tier constants mapping HIGH/LOW tiers to Anthropic model IDs."""

import os
from enum import Enum
from functools import cache

from pydantic_ai.models.anthropic import AnthropicModel


class ModelTier(Enum):
    """Model tiers mapped to Anthropic model IDs: HIGH for vision, LOW for the extraction loop."""

    HIGH = "claude-opus-4-8"
    LOW = "claude-sonnet-4-6"


@cache
def get_model(tier: ModelTier) -> AnthropicModel:
    """Return a cached AnthropicModel for the given tier, building it on first use."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return AnthropicModel(tier.value)
