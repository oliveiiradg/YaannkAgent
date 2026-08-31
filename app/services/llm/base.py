"""Abstração de provider de LLM — Fase 2 da migração Multi-LLM.

O Yaannk trata modelos como providers intercambiáveis. Toda geração passa por
um `LLMProvider`; trocar Ollama por Kimi por Anthropic é configuração, não
reescrita. Ver `01 - Projetos/Pessoal/YaannkAgent/YaannkAgent - Fase 1
Auditoria Multi-LLM.md` no vault.

Fase 2 entrega só a ABC + o adapter Ollama, sem mudança de comportamento.
`structured_generate` fica declarado mas sem implementação — será adicionado
quando surgir o primeiro consumidor (extração estruturada), conforme a
auditoria (§7.3.7).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Mensagem no formato chat (role ∈ {system, user, assistant}).
Message = dict[str, str]


class LLMError(Exception):
    """Falha de um provider de LLM. Quem chama decide o fallback (Fase 7)."""


@dataclass(frozen=True)
class LLMResult:
    """Resultado de uma geração, com os campos que a observabilidade (Fase 3)
    e o fallback (Fase 7) vão precisar."""

    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    # Corpo cru da resposta do provider, para depuração — fora do repr.
    raw: dict = field(default_factory=dict, repr=False)


class LLMProvider(ABC):
    """Contrato mínimo de um provider. Implementações concretas em
    `app/services/llm/<provider>.py` e registradas em `registry.py`."""

    #: Nome curto e estável do provider (`"ollama"`, `"kimi"`, ...).
    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 300,
        temperature: float | None = None,
        timeout: float = 220.0,
    ) -> LLMResult:
        """Gera uma resposta de texto a partir da lista de mensagens.

        Levanta `LLMError` em qualquer falha (rede, timeout, protocolo,
        resposta vazia) — nunca deixa vazar a exceção do transporte.
        """

    async def structured_generate(
        self,
        messages: list[Message],
        schema: dict,
        *,
        max_tokens: int = 512,
        timeout: float = 220.0,
    ) -> dict:
        """Gera saída estruturada validada contra `schema`.

        Sem implementação na Fase 2 — declarado para fechar o contrato da ABC.
        """
        raise NotImplementedError(
            "structured_generate será implementado quando houver o primeiro "
            "consumidor (ver auditoria Fase 1, §7.3.7)"
        )
