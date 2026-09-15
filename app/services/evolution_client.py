from typing import Literal

import httpx

from app.config import settings

Destination = Literal["grupo", "douglas", "bia"]


async def send_message(number: str, text: str) -> None:
    url = f"{settings.evolution_api_url}/message/sendText/{settings.evolution_instance_name}"
    headers = {
        "Content-Type": "application/json",
        "apikey": settings.evolution_api_key,
    }
    body = {"number": number, "text": text}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()


def resolve_jid(destination: Destination) -> str:
    """JID completo (`GROUP_JID`/`DOUGLAS_JID`/`BIA_JID`) para o destino
    lógico pedido (V5). Levanta `ValueError` se a variável correspondente
    não estiver configurada — falha explícita em vez de mandar mensagem
    pra `number=""`."""
    jid_por_destino = {
        "grupo": settings.group_jid,
        "douglas": settings.douglas_jid,
        "bia": settings.bia_jid,
    }
    jid = jid_por_destino.get(destination)
    if not jid:
        raise ValueError(f"destino {destination!r} sem JID configurado no .env")
    return jid


async def send_to(destination: Destination, text: str) -> None:
    """Envia `text` pro destino lógico (`"grupo"`/`"douglas"`/`"bia"`),
    resolvendo o JID via `resolve_jid()`. Quem decide o destino é a camada de
    domínio (Agent Vida/endpoints proativos) — esta função só resolve o
    transporte."""
    jid = resolve_jid(destination)
    await send_message(jid, text)
