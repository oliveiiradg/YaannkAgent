"""Orquestrador (Fase A) — roteamento de intenção via Kimi em linguagem
natural, com fallback determinístico para o keyword matching atual
(`app.services.intent_classifier.classify_intent`).

Decisão de design (Sessão 16/17, `Especificação - Arquitetura
Multi-Agentes/design.md`): o Kimi é perguntado pela **skill** — a mesma
taxonomia de `app.services.skills.SKILLS` que o pipeline já usa hoje — em vez
de perguntar direto pelo `agent` dos futuros Agentes Executores. Isso
preserva `pessoas` e `pendencias` (que hoje têm `priority_folder` e
`prompt_injection` próprios, sem agente dedicado até a Fase D) sem nenhuma
regressão na Fase A. `agent` é só um campo derivado de `skill_name` via
`_SKILL_TO_AGENT`, calculado tanto no caminho Kimi quanto no fallback.

A flag `ORCHESTRATOR_ENABLED` NÃO é lida aqui — é o `pipeline.py` quem decide
se chama `route()` ou vai direto para `classify_intent()`. Isso mantém
`via_fallback` reservado exclusivamente para falha real do Kimi em runtime
(timeout, erro de rede, JSON malformado/inválido), nunca para "está
desligado por configuração" (decisão Q1, sessão 17).
"""

import asyncio
import json
import logging
from dataclasses import dataclass

from app.config import settings
from app.services.intent_classifier import classify_intent
from app.services.llm.base import LLMError, Message
from app.services.llm.registry import get_provider
from app.services.skills import SKILLS

logger = logging.getLogger(__name__)

# Skill -> agente do futuro Agent Executor (Fases B/C/D). `pessoas` e
# `pendencias` não têm agente próprio ainda — caem em "default" sem perder o
# `skill_name`, que é o que o pipeline realmente usa até a Fase D existir.
_SKILL_TO_AGENT: dict[str, str] = {
    "vida_casal": "vida",
    "yaannk_tecnico": "yaannk",
    "conhecimento": "conhecimento",   # Fase D (Sessão 20)
    "carreira": "carreira",           # Fase D (Sessão 20)
    "pessoas": "default",
    "pendencias": "default",
    "default": "default",
}

_VALID_SKILLS = frozenset(SKILLS.keys())

_MAX_TOKENS = 200

_DOMAIN_DESCRIPTIONS = (
    "- yaannk_tecnico: perguntas técnicas sobre o próprio projeto YaannkAgent "
    "(gateway, RAG, pipeline, reranker, arquitetura, bugs, infraestrutura).\n"
    "- vida_casal: gastos, contas, mercado, finanças, listas de compras, "
    "datas importantes, lembretes do casal.\n"
    "- conhecimento: faculdade, semestre, matérias, aulas, provas, trabalhos "
    "acadêmicos, hackathon, cursos e certificações.\n"
    "- carreira: perfil profissional, currículo, vagas, entrevistas, "
    "competências, trajetória e posicionamento profissional.\n"
    "- pessoas: perguntas sobre quem é alguém, contato, responsável por algo.\n"
    "- pendencias: o que está pendente, próximos passos, status de tarefas.\n"
    "- default: qualquer outra coisa — conversa geral, o que não casa com os "
    "domínios acima."
)

_SYSTEM_PROMPT = (
    "Responda APENAS em JSON. NÃO escreva nenhum texto fora do JSON — sem "
    "saudação, explicação ou comentário antes ou depois.\n\n"
    "Você é o orquestrador do Yaannk. Classifique a mensagem do usuário em "
    "UMA das skills abaixo e devolva um JSON válido, exatamente no formato:\n"
    '{"skill_name": "<skill>", "intent": "<intenção em texto livre>", '
    '"context_hint": "<dica curta de contexto>", "confidence": <0.0-1.0>}\n\n'
    "Exemplo de resposta válida (é a resposta INTEIRA — nada mais no output):\n"
    '{"skill_name": "vida_casal", "intent": "consulta de gasto do mês", '
    '"context_hint": "gastos de setembro", "confidence": 0.9}\n\n'
    f"Skills disponíveis:\n{_DOMAIN_DESCRIPTIONS}"
)


@dataclass(frozen=True)
class OrchestratorDecision:
    """Renomeado de `RoutingDecision` (nome do `design.md`) para não colidir
    com `app.services.llm.RoutingDecision` (Router do Bloco 3) — os dois são
    importados no mesmo `pipeline.py`."""

    agent: str          # "vida" | "yaannk" | "conhecimento" | "carreira" | "default"
    intent: str         # intenção em texto livre (Kimi) ou skill_name (fallback)
    context_hint: str   # dica de contexto pro agente; "" no fallback
    confidence: float   # 0.0-1.0; 1.0 no fallback determinístico
    skill_name: str     # compatível com app.services.skills.SKILLS
    via_fallback: bool  # True só em falha real do Kimi — nunca por config

    def as_dict(self) -> dict:
        return {
            "agent": self.agent,
            "intent": self.intent,
            "context_hint": self.context_hint,
            "confidence": self.confidence,
            "skill_name": self.skill_name,
            "via_fallback": self.via_fallback,
        }


def _build_prompt(text: str, history: list[Message] | None) -> list[Message]:
    messages: list[Message] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    # Últimas trocas ajudam a desambiguar ("e o dela?" após pergunta de
    # gasto) sem inflar o prompt de roteamento.
    for msg in (history or [])[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": text})
    return messages


def _parse_response(raw: str) -> OrchestratorDecision:
    """Levanta `ValueError`/`KeyError`/`TypeError` em qualquer desvio do
    contrato — `route()` trata todos esses como falha e cai no fallback."""
    data = json.loads(raw)
    skill_name = data["skill_name"]
    if skill_name not in _VALID_SKILLS:
        raise ValueError(f"skill_name inválido devolvido pelo Kimi: {skill_name!r}")
    confidence = float(data.get("confidence", 0.5))
    return OrchestratorDecision(
        agent=_SKILL_TO_AGENT[skill_name],
        intent=str(data.get("intent", "")),
        context_hint=str(data.get("context_hint", "")),
        confidence=confidence,
        skill_name=skill_name,
        via_fallback=False,
    )


def _fallback(text: str) -> OrchestratorDecision:
    skill_name = classify_intent(text)
    return OrchestratorDecision(
        agent=_SKILL_TO_AGENT.get(skill_name, "default"),
        intent=skill_name,
        context_hint="",
        confidence=1.0,
        skill_name=skill_name,
        via_fallback=True,
    )


async def route(
    text: str, history: list[Message] | None = None
) -> OrchestratorDecision:
    """Chama Kimi para classificar a mensagem; qualquer falha (timeout, erro
    de rede/provider, JSON malformado, campo inválido) cai em `_fallback()`
    com `via_fallback=True`.

    NÃO verifica `settings.orchestrator_enabled` — quem decide se este ponto
    é alcançado é o `pipeline.py` (ver docstring do módulo).
    """
    try:
        provider = get_provider("kimi")
        messages = _build_prompt(text, history)
        result = await asyncio.wait_for(
            provider.generate(
                messages, max_tokens=_MAX_TOKENS, timeout=settings.orchestrator_timeout_s
            ),
            timeout=settings.orchestrator_timeout_s,
        )
        return _parse_response(result.text)
    except (
        TimeoutError,
        LLMError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        logger.warning(
            "Orquestrador: Kimi indisponível/resposta inválida, caindo no "
            "keyword matching (%r)", exc,
        )
        return _fallback(text)
