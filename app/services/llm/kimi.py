"""Adapter Kimi via OpenRouter — Tier 1 (K2.6, default) e Tier 2 (heavy).

API compatível com OpenAI; usa a lib `openai` (`AsyncOpenAI`) apontada para o
`KIMI_BASE_URL` (OpenRouter: `https://openrouter.ai/api/v1`). Um provider por
tier, cada um fixado num modelo:

- `KimiProvider(heavy=False)` → `KIMI_MODEL_TIER1` (ex: `moonshotai/kimi-k2.6`)
- `KimiProvider(heavy=True)`  → `KIMI_MODEL_TIER2` (ex: `moonshotai/kimi-k2.7-code`)

O OpenRouter pede os headers `HTTP-Referer` e `X-Title` para identificar a app
(usados no ranking/analytics do OpenRouter). Enviados como `default_headers`.

Registrado só quando `KIMI_API_KEY` está presente (ver `registry.py`). O Yaannk
não roteia para cá ainda — só entra em uso quando o LLM Router (Fase 5) existir
ou via `LLM_DEFAULT_PROVIDER=kimi`.
"""

import logging
import time
from dataclasses import dataclass

from app.config import settings
from app.services.llm.base import LLMError, LLMProvider, LLMResult, Message
from app.services.llm.telemetry import record_result

logger = logging.getLogger(__name__)

# Identificação da app no OpenRouter (headers obrigatórios).
_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/yaannk-agent",
    "X-Title": "Yaannk",
}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON cru, como o modelo gerou


@dataclass(frozen=True)
class ToolTurn:
    """Um turno do modelo com ferramentas. `message` é a mensagem do
    assistente pronta pra voltar no histórico — inclui `reasoning_details`,
    que o OpenRouter exige de volta pra manter o raciocínio entre chamadas."""

    message: dict
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    latency_ms: int


class KimiProvider(LLMProvider):
    def __init__(self, *, heavy: bool = False) -> None:
        self._heavy = heavy
        self.name = "kimi-heavy" if heavy else "kimi"
        self._model = (
            settings.kimi_model_tier2 if heavy else settings.kimi_model_tier1
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 300,
        temperature: float | None = None,
        timeout: float = 220.0,
    ) -> LLMResult:
        if not settings.kimi_api_key:
            raise LLMError(f"{self.name}: KIMI_API_KEY não configurado")

        # Import tardio — mantém a lib fora do caminho de quem só usa Ollama.
        from openai import AsyncOpenAI, OpenAIError

        client = AsyncOpenAI(
            api_key=settings.kimi_api_key,
            base_url=settings.kimi_base_url,
            timeout=timeout,
            max_retries=1,
            default_headers=_OPENROUTER_HEADERS,
        )

        # K2.6/K2.7 raciocinam por padrão no OpenRouter e gastam centenas de
        # tokens de reasoning antes de responder — numa consulta RAG simples
        # isso é ~40-60s a mais e trunca a resposta dentro do max_tokens.
        # Desligamos o reasoning: resposta direta em ~2s. (OpenRouter:
        # `reasoning.enabled=false`.)
        effective_max = max(max_tokens, settings.kimi_min_max_tokens)

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": effective_max,
            "extra_body": {"reasoning": {"enabled": False}},
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        logger.info(
            "LLM[%s]: %s (max_tokens=%d, pedido=%d)",
            self.name, self._model, effective_max, max_tokens,
        )

        def _extract(resp) -> str:
            try:
                return (resp.choices[0].message.content or "").strip()
            except (AttributeError, IndexError, TypeError) as exc:
                raise LLMError(
                    f"{self.name}: resposta em formato inesperado ({exc!r})"
                ) from exc

        started = time.monotonic()
        try:
            resp = await client.chat.completions.create(**kwargs)
            text = _extract(resp)
            # K2.6 (raciocínio) às vezes volta `content` vazio de forma
            # transiente — 1 retry idêntico antes de deixar a cadeia de
            # escalonamento cair no K2.7-code, que é pior em prosa.
            if not text:
                logger.warning("LLM[%s]: resposta vazia — retry (1/1)", self.name)
                resp = await client.chat.completions.create(**kwargs)
                text = _extract(resp)
        except OpenAIError as exc:
            raise LLMError(f"{self.name}: falha na chamada ({exc!r})") from exc
        finally:
            await client.close()

        latency_ms = int((time.monotonic() - started) * 1000)

        if not text:
            raise LLMError(
                f"{self.name}: resposta vazia após retry (modelo {self._model})"
            )

        usage = getattr(resp, "usage", None)
        return LLMResult(
            text=text,
            provider=self.name,
            model=getattr(resp, "model", self._model),
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            latency_ms=latency_ms,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )

    async def chat_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        max_tokens: int = 4000,
        timeout: float = 60.0,
        reasoning: str = "low",
        tool_choice: str = "auto",
    ) -> ToolTurn:
        """Chamada com tool calling nativo (`tools` da API OpenAI) — usada pelo
        agente (D-11). Diferente de `generate()`, não tenta de novo em resposta
        vazia: quem decide o que fazer com o turno é o loop do agente.

        `reasoning`: `off` | `on` | `low` | `medium` | `high` (esforço)."""
        if not settings.kimi_api_key:
            raise LLMError(f"{self.name}: KIMI_API_KEY não configurado")

        from openai import AsyncOpenAI, OpenAIError

        client = AsyncOpenAI(
            api_key=settings.kimi_api_key,
            base_url=settings.kimi_base_url,
            timeout=timeout,
            max_retries=1,
            default_headers=_OPENROUTER_HEADERS,
        )
        if reasoning in ("off", "on"):
            reasoning_body = {"enabled": reasoning == "on"}
        else:
            reasoning_body = {"effort": reasoning}
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": {"reasoning": reasoning_body},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        started = time.monotonic()
        try:
            resp = await client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise LLMError(f"{self.name}: falha na chamada com ferramentas ({exc!r})") from exc
        finally:
            await client.close()
        latency_ms = int((time.monotonic() - started) * 1000)

        try:
            choice = resp.choices[0]
            msg = choice.message
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name}: resposta em formato inesperado ({exc!r})") from exc

        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        record_result(
            LLMResult(
                text=msg.content or "",
                provider=self.name,
                model=getattr(resp, "model", self._model),
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                latency_ms=latency_ms,
            ),
            "agent",
        )
        return ToolTurn(
            message=msg.model_dump(exclude_none=True),
            content=(msg.content or "").strip(),
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
        )
