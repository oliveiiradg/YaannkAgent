"""Registro de Agentes Executores — resolve `routing.agent` (do orquestrador)
para a instância concreta do `AgentExecutor`.

Fase B: "vida". Fase C: "yaannk". Fase D: "conhecimento" e "carreira".
`pipeline.py` decide se despacha pro registry ou segue o caminho genérico
quando o `agent` não tem registro aqui (ex.: "default").
"""

from app.agents.agent_vida import AgentVida
from app.agents.base import AgentExecutor
from app.agents.rag_agent import AgentCarreira, AgentConhecimento, AgentYaannk

_INSTANCES: dict[str, AgentExecutor] = {
    "vida": AgentVida(),
    "yaannk": AgentYaannk(),
    "conhecimento": AgentConhecimento(),
    "carreira": AgentCarreira(),
}


def get_agent(name: str) -> AgentExecutor | None:
    """Instância do agente pelo nome; `None` se ainda não existe registro
    pra esse `agent` (chamador decide o fallback)."""
    return _INSTANCES.get(name)
