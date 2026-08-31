import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class MessageKey(_CamelModel):
    remote_jid: str = Field(alias="remoteJid")
    from_me: bool = Field(alias="fromMe")
    id: str
    # Presente só em mensagens de grupo: o JID de quem realmente enviou
    # (remote_jid, nesse caso, é o JID do grupo `...@g.us`).
    participant: Optional[str] = None


class ContextInfo(_CamelModel):
    # Baileys/Evolution: JIDs mencionados na mensagem (@contato). O texto vem
    # com `@<numero>` (não `@yaannk`); o nome exibido não chega no payload.
    mentioned_jid: list[str] = Field(default_factory=list, alias="mentionedJid")


class ExtendedTextMessage(_CamelModel):
    text: str = ""
    context_info: Optional[ContextInfo] = Field(default=None, alias="contextInfo")


class MessageContent(_CamelModel):
    conversation: Optional[str] = None
    extended_text_message: Optional[ExtendedTextMessage] = Field(
        default=None, alias="extendedTextMessage"
    )

    @property
    def text(self) -> Optional[str]:
        if self.conversation:
            return self.conversation
        if self.extended_text_message:
            return self.extended_text_message.text
        return None

    @property
    def mentioned_jids(self) -> list[str]:
        etm = self.extended_text_message
        if etm and etm.context_info:
            return etm.context_info.mentioned_jid
        return []


class MessageData(_CamelModel):
    key: MessageKey
    participant: Optional[str] = None
    push_name: Optional[str] = Field(default=None, alias="pushName")
    message_type: Optional[str] = Field(default=None, alias="messageType")
    message: MessageContent
    context_info: Optional[ContextInfo] = Field(default=None, alias="contextInfo")
    message_timestamp: Optional[int] = Field(default=None, alias="messageTimestamp")

    @property
    def mentioned_jids(self) -> list[str]:
        # A menção pode vir dentro de extendedTextMessage.contextInfo ou,
        # dependendo da versão da Evolution, num contextInfo no nível de data.
        jids = list(self.message.mentioned_jids)
        if self.context_info:
            jids.extend(self.context_info.mentioned_jid)
        return jids


class WebhookPayload(_CamelModel):
    event: str
    instance: str
    # Evolution costuma expor aqui o JID dono da instância (o próprio Yaannk).
    sender: Optional[str] = None
    data: MessageData


def jid_to_number(jid: Optional[str]) -> str:
    """`5521999999999@s.whatsapp.net` / `...:12@...` / `<lid>@lid` -> `5521999999999`."""
    if not jid:
        return ""
    return re.split(r"[@:]", jid, maxsplit=1)[0].strip()
