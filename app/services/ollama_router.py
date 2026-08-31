import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Nó primário (desktop com GPU) e fallback (o próprio notebook, onde o
# Gateway roda). O fallback usa settings.ollama_url, que já é o endpoint
# local configurado via .env (http://localhost:11434 por padrão).
# `OLLAMA_DESKTOP_URL` vazio (ou placeholder) → sem nó primário, vai direto
# para o fallback local.
_FALLBACK_URL = settings.ollama_url
_DESKTOP_URL = settings.ollama_desktop_url
if not _DESKTOP_URL or "DESKTOP_LAN_IP" in _DESKTOP_URL:
    _DESKTOP_URL = ""

_HEALTH_CHECK_TIMEOUT = 3.0

# Cache curto para não repetir o health-check (que gera 1 token de verdade,
# não é só um ping TCP) a cada chamada do pipeline — uma única mensagem do
# WhatsApp já dispara até 3 chamadas ao Ollama (Bloco 1, 2 e 3).
_CACHE_TTL = 30.0

_cached_url: str | None = None
_cached_at: float = 0.0


async def _is_healthy(url: str) -> bool:
    """Health check real: gera 1 token de fato, não apenas testa conectividade."""
    body = {
        "model": settings.ollama_model,
        "prompt": "ping",
        "stream": False,
        "options": {"num_predict": 1},
    }
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
            response = await client.post(f"{url}/api/generate", json=body)
            response.raise_for_status()
            return True
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.warning("Health check falhou para %s: %r", url, exc)
        return False


async def get_ollama_url() -> str:
    """Retorna a URL do nó Ollama a usar, com desktop como primário.

    Tenta o desktop primeiro (health check real, timeout de 3s). Se falhar
    ou não responder a tempo, cai para o fallback (o próprio notebook). O
    resultado fica em cache por _CACHE_TTL segundos para não pagar o custo
    de um health-check completo a cada chamada do pipeline.
    """
    global _cached_url, _cached_at

    if not _DESKTOP_URL:
        return _FALLBACK_URL

    now = time.monotonic()
    if _cached_url is not None and (now - _cached_at) < _CACHE_TTL:
        return _cached_url

    if await _is_healthy(_DESKTOP_URL):
        logger.info("Roteador Ollama: nó escolhido = desktop (%s)", _DESKTOP_URL)
        _cached_url, _cached_at = _DESKTOP_URL, now
        return _DESKTOP_URL

    logger.warning(
        "Roteador Ollama: desktop (%s) indisponível, usando fallback notebook (%s)",
        _DESKTOP_URL,
        _FALLBACK_URL,
    )
    _cached_url, _cached_at = _FALLBACK_URL, now
    return _FALLBACK_URL
