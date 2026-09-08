"""Agente RAG genérico — base das Fases C e D.

Agent YaannkAgent (C), Agent Conhecimento e Agent Carreira (D) fazem
exatamente a mesma coisa, mudando só a skill (e portanto a `priority_folder`,
o `prompt_injection` e as `excluded_subfolders`, todos já parametrizados em
`app.services.skills.SKILLS`):

    Bloco 1 (contexto estrutural, Ollama)
    → Bloco 2 (pasta prioritária)
    → RAG restrito à pasta da skill (+ Agentic RAG se `AGENTIC_RAG_ENABLED`)
    → Bloco 3 (Kimi, com o `prompt_injection` da skill)

Isso é literalmente o que `pipeline.retrieve()` + `pipeline._block3()` já
fazem quando recebem um `skill_name` fixo — então o agente é um wrapper fino
sobre eles, não uma reimplementação. Reimplementar significaria duplicar
Bloco 1/2, decomposição, Agentic RAG, Router e Bloco 3.

Diferente do Agent Vida, estes não têm fast-path determinístico: o domínio
deles não tem operação estruturada (não há "tabela de gastos" equivalente),
é consulta a documentação em prosa.
"""

import logging
import time

from app.agents.base import AgentExecutor, AgentResponse
from app.services.llm.base import Message
from app.services.orchestrator import OrchestratorDecision

logger = logging.getLogger(__name__)


class RagAgent(AgentExecutor):
    """Agente de consulta documental restrito à pasta de uma skill.

    `skill_name` amarra o agente à entrada correspondente de `SKILLS` —
    é ela que define `priority_folder`, `prompt_injection` e
    `excluded_subfolders` (Bug 10). `agent_name` é só o rótulo que aparece
    em `AgentResponse.agent` e na telemetria.
    """

    skill_name: str = "default"
    agent_name: str = "rag"

    async def handle(
        self,
        message: str,
        routing: OrchestratorDecision,
        history: list[Message],
        sender: str,
        *,
        conv_key: str,
    ) -> AgentResponse:
        # Import tardio: `pipeline.py` importa o registry (que importa este
        # módulo) e este módulo precisa de `retrieve()`/`_block3()` de volta —
        # ciclo se fosse import de topo. Mesmo padrão do `agent_vida.py`.
        from app.services.pipeline import _block3, retrieve
        from app.services.skills import get_skill

        started = time.monotonic()
        r = await retrieve(message, conv_key=conv_key, skill_name=self.skill_name)
        text, succeeded = await _block3(message, r, get_skill(self.skill_name), history)
        latency_ms = int((time.monotonic() - started) * 1000)

        logger.info(
            "Agent %s: resposta via RAG (skill=%s, %d docs, %dms, ok=%s)",
            self.agent_name, self.skill_name, r.n_docs, latency_ms, succeeded,
        )
        return AgentResponse(
            text=text, source="rag", agent=self.agent_name, latency_ms=latency_ms,
            # Metadados que o `PipelineResult`/benchmark precisam — sem eles
            # os checks de retrieval (rag_contains, tier, n_docs) morreriam.
            decision=r.decision, rag_files=r.rag_files,
            vault_context=r.vault_context, decomposed=r.decomposed,
            n_docs=r.n_docs, intent=r.decision.intent, succeeded=succeeded,
        )


class AgentYaannk(RagAgent):
    """Fase C — consulta técnica sobre o próprio projeto YaannkAgent.

    Agentic RAG entra sozinho quando `AGENTIC_RAG_ENABLED=true` (é
    `pipeline._run_rag()` quem decide), restrito a
    `01 - Projetos/Pessoal/YaannkAgent/` menos as `excluded_subfolders`
    (Especificação/, Sessões/, SDD fantasma — Bug 10)."""

    skill_name = "yaannk_tecnico"
    agent_name = "yaannk"


class AgentConhecimento(RagAgent):
    """Fase D — faculdade, cursos, certificações, hackathon."""

    skill_name = "conhecimento"
    agent_name = "conhecimento"


class AgentCarreira(RagAgent):
    """Fase D — perfil profissional, currículo, competências, trajetória."""

    skill_name = "carreira"
    agent_name = "carreira"
