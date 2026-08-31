"""Camada de providers de LLM (Fases 2–4 da migração Multi-LLM)."""

from app.services.llm.base import (
    LLMError,
    LLMProvider,
    LLMResult,
    Message,
)
from app.services.llm.registry import available_providers, get_provider
from app.services.llm.router import RoutingDecision, route
from app.services.llm.telemetry import (
    clear_context,
    get_context,
    record_error,
    record_result,
    set_context,
    update_context,
)

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMResult",
    "Message",
    "available_providers",
    "get_provider",
    "route",
    "RoutingDecision",
    "set_context",
    "update_context",
    "get_context",
    "clear_context",
    "record_result",
    "record_error",
]
