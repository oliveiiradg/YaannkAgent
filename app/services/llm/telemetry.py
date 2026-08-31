"""Observabilidade de LLM (Fase 4) — uma linha em `llm_calls` por chamada.

Mede Ollama e Kimi **separadamente**: provider, modelo, tokens, latência e
custo estimado. Ligada por `LLM_TELEMETRY` (default true). Falha de escrita
nunca derruba o pipeline.

O `request_id` e o `intent` da mensagem em curso vêm de um `ContextVar` setado
no início de `receive_webhook` — evita passar esses campos por 4 camadas até o
ponto de chamada do modelo. O `block` (`bloco1` / `bloco2` / `bloco3`) é local
e passado explicitamente por quem chama.
"""

import logging
import sqlite3
from contextvars import ContextVar
from dataclasses import dataclass, replace

from app.config import settings
from app.services.db import get_connection
from app.services.llm.base import LLMResult
from app.services.llm.pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMContext:
    request_id: str
    intent: str | None = None
    complexity: int | None = None
    tier: int | None = None
    rag_enabled: bool | None = None
    retrieved_documents: int | None = None


_ctx: ContextVar[LLMContext | None] = ContextVar("llm_ctx", default=None)


def set_context(request_id: str, **kw) -> None:
    _ctx.set(LLMContext(request_id=request_id, **kw))


def update_context(**kw) -> None:
    """Preenche campos que só ficam conhecidos mais tarde (intent, nº de docs)."""
    current = _ctx.get()
    if current is None:
        return
    _ctx.set(replace(current, **kw))


def get_context() -> LLMContext | None:
    return _ctx.get()


def clear_context() -> None:
    _ctx.set(None)


def _conn() -> sqlite3.Connection:
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            ts TEXT NOT NULL DEFAULT (datetime('now')),
            block TEXT,
            intent TEXT,
            complexity INTEGER,
            tier INTEGER,
            provider TEXT NOT NULL,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            latency_ms INTEGER,
            cost_usd REAL,
            rag_enabled INTEGER,
            retrieved_documents INTEGER,
            fallback INTEGER NOT NULL DEFAULT 0,
            fallback_reason TEXT,
            success INTEGER NOT NULL,
            error TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_calls_prov_ts "
        "ON llm_calls(provider, ts)"
    )
    return conn


def _write(row: dict) -> None:
    if not settings.llm_telemetry:
        return
    try:
        conn = _conn()
        try:
            cols = ", ".join(row)
            marks = ", ".join("?" for _ in row)
            conn.execute(
                f"INSERT INTO llm_calls ({cols}) VALUES ({marks})",
                tuple(row.values()),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("Telemetria LLM: falha ao gravar (%r)", exc)


def record_result(
    result: LLMResult, block: str, *,
    fallback: bool = False, fallback_reason: str | None = None,
) -> None:
    """Grava uma chamada bem-sucedida a partir do `LLMResult`."""
    ctx = get_context()
    cost = estimate_cost_usd(
        result.provider, result.model, result.input_tokens, result.output_tokens
    )
    _write(
        {
            "request_id": ctx.request_id if ctx else None,
            "block": block,
            "intent": ctx.intent if ctx else None,
            "complexity": ctx.complexity if ctx else None,
            "tier": ctx.tier if ctx else None,
            "provider": result.provider,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "cost_usd": cost,
            "rag_enabled": _as_int(ctx.rag_enabled) if ctx else None,
            "retrieved_documents": ctx.retrieved_documents if ctx else None,
            "fallback": int(fallback),
            "fallback_reason": fallback_reason,
            "success": 1,
        }
    )
    logger.info(
        "Telemetria: %s/%s block=%s in=%s out=%s %sms $%.6f%s",
        result.provider, result.model, block,
        result.input_tokens, result.output_tokens, result.latency_ms, cost,
        f" (fallback: {fallback_reason})" if fallback else "",
    )


def record_error(
    provider: str, block: str, error: str, model: str | None = None, *,
    fallback: bool = False, fallback_reason: str | None = None,
) -> None:
    """Grava uma chamada que falhou (o `LLMResult` não existe)."""
    ctx = get_context()
    _write(
        {
            "request_id": ctx.request_id if ctx else None,
            "block": block,
            "intent": ctx.intent if ctx else None,
            "provider": provider,
            "model": model,
            "cost_usd": 0.0,
            "fallback": int(fallback),
            "fallback_reason": fallback_reason,
            "success": 0,
            "error": error[:500],
        }
    )


def _as_int(value: bool | None) -> int | None:
    return None if value is None else int(value)
