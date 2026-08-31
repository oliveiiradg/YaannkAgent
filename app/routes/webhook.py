import hmac
import logging
import re
import time
import uuid
from collections import deque

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.models.webhook_payload import WebhookPayload, jid_to_number
from app.services import pipeline
from app.services.conversation_store import add_message, clear_history
from app.services.evolution_client import send_message
from app.services.vault_writer import VaultWriteError, detect_save_intent, save_to_vault

logger = logging.getLogger(__name__)

router = APIRouter()

if not settings.webhook_secret:
    logger.warning(
        "WEBHOOK_SECRET não configurado — requisições do webhook não são autenticadas"
    )

_RESET_COMMANDS = {"esquece tudo", "/reset"}

# Nos grupos (chat_id `...@g.us`) o Yaannk só responde quando é acionado:
#   1. texto começando com um dos gatilhos (case-insensitive); ou
#   2. menção via contato do WhatsApp — o Baileys entrega isso em
#      `contextInfo.mentionedJid` (o texto vem com `@<numero>`, não `@yaannk`).
# O gatilho/menção é removido antes de processar. Em conversa individual ele
# responde tudo, como antes.
_GROUP_TRIGGERS = ("@yaannk",)

# Nome de quem enviou, para carimbar o registro no vault e personalizar a
# resposta. Mapa número→nome vem do .env (KNOWN_NAMES=numero:Nome,...).
_KNOWN_NAMES = settings.known_names


_TRIGGER_TRIM = " :,-––—"


def _activate_in_group(
    text: str, mentioned_numbers: set[str], yaannk_ids: set[str]
) -> str | None:
    """Decide se a mensagem de grupo aciona o Yaannk e devolve o texto já sem
    o gatilho; devolve None se não aciona (mensagem a ignorar).

    `yaannk_ids` = identificadores do próprio Yaannk que podem aparecer em
    `mentionedJid` (número e/ou LID, só dígitos).
    """
    stripped = text.lstrip()
    low = stripped.lower()
    for trigger in _GROUP_TRIGGERS:
        if low.startswith(trigger):
            return stripped[len(trigger):].lstrip(_TRIGGER_TRIM).strip()

    hit = yaannk_ids & mentioned_numbers
    if hit:
        # Menção via contato: o texto traz `@<numero-ou-lid>` no lugar do nome.
        for ident in hit:
            text = re.sub(rf"@{re.escape(ident)}\b", " ", text)
        return text.lstrip(_TRIGGER_TRIM).strip()

    return None

# Em memória, por processo — reseta ao reiniciar o gateway, o que é aceitável
# (só precisa sobreviver enquanto uma resposta pode estar em voo).
_latest_request: dict[str, float] = {}

# Dedupe por message_id (`data.key.id`): a Evolution reenvia o mesmo evento
# quando o webhook demora a responder (retry). Guardamos os últimos IDs vistos
# e ignoramos repetições. Só precisa sobreviver ao processo — janela curta.
_DEDUPE_MAXLEN = 500
_processed_ids: deque[str] = deque(maxlen=_DEDUPE_MAXLEN)
_processed_set: set[str] = set()


def _already_processed(message_id: str) -> bool:
    """True se `message_id` já foi visto; caso contrário registra e devolve False."""
    if message_id in _processed_set:
        return True
    if len(_processed_ids) == _DEDUPE_MAXLEN:
        _processed_set.discard(_processed_ids[0])  # append evicta o mais antigo
    _processed_ids.append(message_id)
    _processed_set.add(message_id)
    return False


@router.post("/webhook")
async def receive_webhook(payload: WebhookPayload, request: Request) -> dict:
    if settings.webhook_secret:
        got = request.headers.get("x-webhook-secret") or ""
        if not hmac.compare_digest(got, settings.webhook_secret):
            client = request.client.host if request.client else "?"
            logger.warning(
                "Webhook rejeitado: X-Webhook-Secret inválido ou ausente (de %s)", client
            )
            raise HTTPException(status_code=401, detail="invalid webhook secret")

    if payload.event != "messages.upsert":
        return {"status": "ignored", "reason": "event not handled"}

    if payload.data.key.from_me:
        return {"status": "ignored", "reason": "own message"}

    # Dedupe: a Evolution reenvia o mesmo evento em retry (comum quando o
    # webhook demora a responder). Processa cada message_id uma única vez.
    message_id = payload.data.key.id
    if message_id:
        if _already_processed(message_id):
            logger.info("Webhook duplicado ignorado (message_id=%s)", message_id)
            return {"status": "duplicate"}
    else:
        logger.warning("Webhook sem data.key.id — dedupe não aplicável")

    remote_jid = payload.data.key.remote_jid
    is_group = remote_jid.endswith("@g.us")

    # Em grupo, quem enviou vem em `key.participant` (ou `data.participant`);
    # `remote_jid` é o JID do grupo. Em conversa individual, é o próprio JID.
    if is_group:
        origin = payload.data.key.participant or payload.data.participant or ""
    else:
        origin = remote_jid

    # Contas novas do WhatsApp entregam o remetente do grupo como `<lid>@lid`
    # em vez de `<numero>@s.whatsapp.net`. Resolve via LID_MAP; se não houver
    # mapa, ainda aceita se o lid estiver em ALLOWED_LIDS (aí `quem` fica sem
    # nome conhecido, cai no pushName).
    raw_id = origin.split("@")[0].split(":")[0]
    is_lid = origin.endswith("@lid")
    sender_number = settings.lid_map.get(raw_id, raw_id) if is_lid else raw_id

    allowed = sender_number in settings.allowed_numbers or (
        is_lid and raw_id in settings.allowed_lids
    )
    if not allowed:
        if is_lid:
            logger.info(
                "Mensagem de grupo bloqueada — LID não autorizado: %s "
                "(mapeie em LID_MAP=%s:<numero> ou libere em ALLOWED_LIDS)",
                raw_id, raw_id,
            )
        else:
            logger.info(
                "Mensagem bloqueada (remetente fora da allowlist): %s (grupo=%s)",
                sender_number, is_group,
            )
        return {"status": "ignored", "reason": "sender not allowed"}

    text = payload.data.message.text
    if not text:
        return {"status": "ignored", "reason": "no text content"}

    # Ativação nos grupos: `@yaannk` literal ou menção via contato.
    if is_group:
        mentioned = payload.data.mentioned_jids
        mentioned_numbers = {jid_to_number(j) for j in mentioned}
        # Identificadores do próprio Yaannk que podem aparecer em mentionedJid:
        # o número (YAANNK_NUMBER / campo `sender`) e o LID (YAANNK_LID) — em
        # grupo a menção costuma vir como `<lid>@lid`, não como o número.
        yaannk_ids = {
            settings.yaannk_number or jid_to_number(payload.sender),
            settings.yaannk_lid,
        } - {""}

        triggered = _activate_in_group(text, mentioned_numbers, yaannk_ids)
        if triggered is None:
            return {"status": "ignored", "reason": "group message without @yaannk"}
        if not triggered:
            await send_message(remote_jid, "Oi! Manda a pergunta depois do @yaannk.")
            return {"status": "ok", "reason": "empty group trigger"}
        text = triggered

    # Chave de histórico e destino da resposta: o grupo é uma conversa só
    # (contexto compartilhado do casal); a individual continua por número.
    conv_key = remote_jid if is_group else sender_number
    reply_to = remote_jid if is_group else sender_number
    quem = _KNOWN_NAMES.get(sender_number) or payload.data.push_name or sender_number
    # Só personaliza a saudação ("Anotado, Alice!") quando o número está no
    # KNOWN_NAMES — pushName pode ser apelido/emoji e não serve pro vault.
    nome = _KNOWN_NAMES.get(sender_number)
    voc = f", {nome}" if nome else ""

    logger.info("Mensagem de %s (%s chars)", quem, len(text))
    logger.debug("Mensagem de %s (%s): %s", quem, conv_key, text)

    if text.strip().lower().rstrip("!.?") in _RESET_COMMANDS:
        clear_history(conv_key)
        reply = f"Prontinho{voc}, esqueci nosso histórico de conversa."
        logger.info("Histórico resetado para %s", conv_key)
        await send_message(reply_to, reply)
        return {"status": "ok", "reason": "history reset"}

    # Intenção de registro ("anota", "gastei", "lembra que"...) — salva direto
    # na nota certa de `03 - Vida/` e responde a confirmação, sem rodar o RAG.
    save_tipo = detect_save_intent(text)
    if save_tipo:
        logger.info("Intenção de registro detectada: %s", save_tipo)
        try:
            confirmation = await save_to_vault(
                save_tipo, text, quem=quem, voc=voc, chat_id=conv_key
            )
            add_message(conv_key, "user", text)
            add_message(conv_key, "assistant", confirmation)
        except VaultWriteError as exc:
            logger.warning("save_to_vault falhou: %r", exc)
            confirmation = (
                "Não consegui salvar no vault agora (Obsidian pode estar fora "
                "do ar). Tenta de novo daqui a pouco."
            )
        await send_message(reply_to, confirmation)
        return {"status": "ok", "reason": f"saved to vault ({save_tipo})"}

    request_token = time.time()
    _latest_request[conv_key] = request_token

    # request_id único por mensagem — o pipeline seta o ContextVar da telemetria.
    request_id = uuid.uuid4().hex[:12]
    logger.info("request_id=%s (%s)", request_id, conv_key)

    # Núcleo do pipeline (app/services/pipeline.py): fast-path financeiro →
    # Bloco 1/2 → RAG → Router → Bloco 3.
    result = await pipeline.answer(
        text,
        conv_key=conv_key,
        autor=nome,
        request_id=request_id,
    )

    # Proteção contra resposta obsoleta: se uma mensagem mais nova dessa conversa
    # chegou enquanto o pipeline rodava, descarta silenciosamente.
    if _latest_request.get(conv_key) != request_token:
        logger.info(
            "Resposta descartada por obsolescência (mensagem mais nova chegou) para %s",
            conv_key,
        )
        return {"status": "ok", "reason": "stale, discarded"}

    if result.succeeded:
        add_message(conv_key, "user", text)
        add_message(conv_key, "assistant", result.reply)

    logger.info(
        "Resposta enviada a %s (%s chars, source=%s)",
        conv_key, len(result.reply), result.source,
    )
    logger.debug("Resposta enviada a %s: %s", conv_key, result.reply)
    await send_message(reply_to, result.reply)

    return {"status": "ok", "reason": result.source}
