"""Registro de providers — resolve um nome (`"ollama"`, `"kimi"`, ...) para a
instância concreta do `LLMProvider`.

Fase 2: só `ollama`. A Fase 3 adiciona `kimi` (K2.6 default de API, K3 heavy);
Anthropic fica para depois do benchmark. Enquanto o LLM Router (Fase 6) não
existe, todo o pipeline usa `get_provider()` sem argumento → o provider default
de `settings.llm_default_provider` (hoje `ollama`).
"""

import logging

from app.config import settings
from app.services.llm.base import LLMError, LLMProvider
from app.services.llm.ollama import OllamaProvider

logger = logging.getLogger(__name__)

_INSTANCES: dict[str, LLMProvider] = {}


def _build(name: str) -> LLMProvider:
    if name == "ollama":
        return OllamaProvider()
    if name in ("kimi", "kimi-heavy"):
        if not settings.kimi_api_key:
            raise LLMError(f"provider {name!r}: KIMI_API_KEY não configurado")
        from app.services.llm.kimi import KimiProvider

        return KimiProvider(heavy=(name == "kimi-heavy"))
    # Anthropic entra aqui só depois do benchmark (decisão 28/08).
    raise LLMError(
        f"provider {name!r} desconhecido ou não habilitado "
        f"(disponíveis: {', '.join(available_providers())})"
    )


def get_provider(name: str | None = None) -> LLMProvider:
    """Provider pelo nome; sem nome, o default de configuração.

    Instâncias são reaproveitadas (os providers são stateless — só carregam
    config; abrem o cliente HTTP por chamada).
    """
    resolved = name or settings.llm_default_provider
    if resolved not in _INSTANCES:
        _INSTANCES[resolved] = _build(resolved)
        logger.info("LLM registry: provider %r instanciado", resolved)
    return _INSTANCES[resolved]


def available_providers() -> list[str]:
    """Nomes que `get_provider` consegue resolver com a config atual."""
    names = ["ollama"]
    if settings.kimi_api_key:
        names += ["kimi", "kimi-heavy"]
    return names
