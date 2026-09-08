"""Contrato dos Agentes Executores (Fase B) — ver
`Especificação - Arquitetura Multi-Agentes/design.md` no vault.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.services.llm.base import Message
from app.services.orchestrator import OrchestratorDecision


@dataclass(frozen=True)
class AgentResponse:
    text: str
    source: str        # "fast_path" | "rag" | "llm"
    agent: str          # nome do agente que respondeu (ex.: "vida")
    latency_ms: int

    # --- metadados de retrieval (Sessão 20, Fases C/D) ---------------------
    # Só preenchidos por agentes que passam pelo RAG. Existem porque o
    # `PipelineResult` (e portanto o benchmark: `rag_contains`, `tier`,
    # `n_docs`, `intent`) precisa deles — sem isso, despachar o caminho lento
    # por um agente apagaria todos os checks de retrieval do gate duro.
    decision: Any = None                # llm.RoutingDecision | None
    rag_files: list[str] = field(default_factory=list)
    vault_context: str = ""
    decomposed: bool = False
    n_docs: int = 0
    intent: str | None = None
    # False quando o Bloco 3 falhou e `text` é a mensagem de desculpa — o
    # webhook usa isso pra NÃO gravar o erro no histórico da conversa.
    succeeded: bool = True


class AgentExecutor(ABC):
    @abstractmethod
    async def handle(
        self,
        message: str,
        routing: OrchestratorDecision,
        history: list[Message],
        sender: str,
        *,
        conv_key: str,
    ) -> AgentResponse:
        """Processa a mensagem já roteada pro domínio deste agente."""
