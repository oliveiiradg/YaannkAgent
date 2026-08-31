"""Adapter Ollama — Tier 0 (Qwen local).

Embrulha a chamada `/api/chat` do Ollama, já com o roteador de nó
(`ollama_router`: desktop com GPU → fallback notebook CPU). É o provider
default e o único que não depende de chave de API — o Yaannk tem que
continuar 100% funcional só com ele (princípio da spec: nenhum provider
obrigatório além do local).
"""

import logging
import time

import httpx

from app.config import settings
from app.services.llm.base import LLMError, LLMProvider, LLMResult, Message
from app.services.ollama_router import get_ollama_url

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    async def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 300,
        temperature: float | None = None,
        timeout: float = 220.0,
    ) -> LLMResult:
        base_url = await get_ollama_url()
        model = settings.ollama_model
        logger.info("LLM[ollama]: %s em %s (max_tokens=%d)", model, base_url, max_tokens)

        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature

        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{base_url}/api/chat", json=body)
                response.raise_for_status()
                payload = response.json()
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            raise LLMError(f"ollama: falha na chamada ({exc!r})") from exc
        except ValueError as exc:  # JSON inválido
            raise LLMError(f"ollama: resposta não-JSON ({exc!r})") from exc

        latency_ms = int((time.monotonic() - started) * 1000)

        try:
            text = payload["message"]["content"].strip()
        except (KeyError, TypeError, AttributeError) as exc:
            raise LLMError(f"ollama: resposta em formato inesperado ({exc!r})") from exc

        return LLMResult(
            text=text,
            provider=self.name,
            model=model,
            input_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
            latency_ms=latency_ms,
            raw=payload,
        )
