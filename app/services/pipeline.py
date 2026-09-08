"""Núcleo do pipeline de resposta — Fase 8.

Extraído de `receive_webhook` para ser exercitável fora do WhatsApp (benchmark
da Fase 8, calibração da Fase 9). O webhook cuida de identidade, ativação, ack,
persistência de histórico e envio; aqui fica a cadeia fast-path → Bloco 1/2 →
RAG → Router → Bloco 3.

Duas entradas públicas:
- `retrieve(text, conv_key)` — vai até o Router (sem Bloco 3). Barato, sem Kimi.
- `answer(text, conv_key, autor, on_slow_path)` — pipeline completo, com a
  resposta final. É o que o webhook chama.

O branch de **registro de gasto** (`save_to_vault`) NÃO passa por aqui — é write
path e fica no webhook.
"""

import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from app.agents.registry import get_agent
from app.config import settings
from app.services.orchestrator import _SKILL_TO_AGENT, OrchestratorDecision
from app.services.agentic_rag import refine
from app.services.conversation_store import get_recent_messages
from app.services.intent_classifier import classify_intent
from app.services.llm import (
    LLMError,
    RoutingDecision,
    route,
    set_context,
    update_context,
)
from app.services.llm.base import Message
from app.services.ollama_client import generate_response
from app.services.orchestrator import route as orchestrator_route
from app.services.query_decomposer import (
    build_decomposed_context,
    decompose_query,
    multi_search,
)
from app.services.skills import get_skill
from app.services.vault_search import identify_relevant_folder
from app.services.vault_search_semantic import (
    build_context_blocks,
    search_vault_hybrid_ranked,
)
from app.services.vault_structure import build_block3_context, get_structural_context

logger = logging.getLogger(__name__)

# Blocos de contexto do RAG começam com `### <caminho>.md`; o snippet abaixo pode
# ter seus próprios `### subtítulo` — daí o `.md$` para pegar só os caminhos.
_RAG_FILE_RE = re.compile(r"^### (.+\.md)$", re.MULTILINE)
_NET_ERRORS = (httpx.TimeoutException, httpx.HTTPError, LLMError)


@dataclass
class RetrievalResult:
    skill_name: str
    decision: RoutingDecision
    structural_context: str
    priority_folder: str | None
    vault_context: str
    rag_files: list[str]
    decomposed: bool
    n_docs: int
    sub_queries: list[str]
    # Agentic RAG (só no ramo não decomposto; 1 busca = loop desligado ou
    # contexto suficiente de primeira).
    agentic_searches: int = 1
    agentic_queries: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    reply: str
    source: str                       # "sql" | "vault" | "llm" (fast-paths legados)
                                       # ou "fast_path" | "llm" | "rag" (AgentResponse.source, Fase B)
    succeeded: bool
    request_id: str
    skill_name: str
    intent: str | None
    decision: RoutingDecision | None = None    # None no fast-path SQL
    rag_files: list[str] = field(default_factory=list)
    vault_context: str = ""
    decomposed: bool = False
    n_docs: int = 0


async def _resolve_routing(text: str) -> OrchestratorDecision:
    """Fase A: substitui a chamada direta a `classify_intent()` — a flag
    decide ANTES de instanciar o orquestrador (decisão Q1, sessão 17):
    `route()` nunca é chamado com `ORCHESTRATOR_ENABLED=false`, então
    `via_fallback` no `OrchestratorDecision` fica reservado só para falha
    real do Kimi em runtime.

    Sessão 20: devolve a decisão inteira, não só `skill_name` — os agentes
    das Fases C/D recebem `routing` no `handle()` (contrato da ABC), e
    construir um objeto vazio só pra satisfazer a assinatura seria pior."""
    if not settings.orchestrator_enabled:
        logger.info("Orquestrador: desligado, usando keyword matching")
        skill_name = classify_intent(text)
        return OrchestratorDecision(
            agent=_SKILL_TO_AGENT.get(skill_name, "default"),
            intent=skill_name, context_hint="", confidence=1.0,
            skill_name=skill_name, via_fallback=False,
        )
    decision = await orchestrator_route(text)
    logger.info(
        "Orquestrador: agent=%s intent=%s via_fallback=%s",
        decision.agent, decision.intent, decision.via_fallback,
    )
    if decision.via_fallback:
        logger.info("Orquestrador: fallback ativado (razão)")
    return decision


async def _resolve_skill_name(text: str) -> str:
    """Compatibilidade: `retrieve()` e os testes ainda querem só o nome."""
    return (await _resolve_routing(text)).skill_name


async def _block1_structural_context() -> str:
    try:
        ctx = await get_structural_context()
        logger.info("Contexto estrutural obtido (%d chars)", len(ctx))
        return ctx
    except _NET_ERRORS as exc:
        logger.warning("Falha ao obter contexto estrutural (Bloco 1): %r", exc)
        return ""


async def _block2_priority_folder(
    text: str, structural_context: str, skill: dict
) -> str | None:
    skill_folder = skill["priority_folder"] or None
    llm_folder = None
    try:
        llm_folder = await identify_relevant_folder(text, structural_context)
    except _NET_ERRORS as exc:
        logger.warning("Falha ao identificar pasta relevante (Bloco 2): %r", exc)
    priority_folder = llm_folder or skill_folder
    logger.info(
        "Pasta prioritária: %s (LLM=%s, skill=%s)",
        priority_folder or "nenhuma", llm_folder or "nenhuma",
        skill_folder or "nenhuma",
    )
    return priority_folder


async def _run_rag(
    text: str, priority_folder: str | None, intent: str | None = None
) -> tuple[str, bool, list[str], int, list[str], int, list[str]]:
    """Decompõe a pergunta e roda o RAG. Retorna
    (vault_context, decomposed, rag_files, n_docs, sub_queries,
    agentic_searches, agentic_queries)."""
    sub_queries = decompose_query(text)
    decomposed = len(sub_queries) > 1
    agentic_searches, agentic_queries = 1, [text]
    if decomposed:
        logger.info("Pergunta decomposta em %d sub-itens", len(sub_queries))
        multi_results = await multi_search(sub_queries, priority_folder, intent=intent)
        n_found = sum(1 for r in multi_results if r["context"] != "NÃO ENCONTRADO")
        logger.info(
            "Multi-search: %d/%d sub-itens com contexto no vault",
            n_found, len(multi_results),
        )
        vault_context = build_decomposed_context(multi_results)
        n_docs = n_found
    else:
        # Ramo não decomposto: a busca one-shot de sempre, opcionalmente
        # seguida do loop de refinamento (Agentic RAG). Com o flag desligado
        # o caminho é idêntico ao antigo `search_vault_hybrid(text, ...)`.
        items = await search_vault_hybrid_ranked(text, priority_folder, intent=intent)
        if settings.agentic_rag_enabled and items:
            agentic = await refine(
                text, priority_folder, initial_items=items, intent=intent
            )
            items = agentic.items
            agentic_searches, agentic_queries = agentic.n_searches, agentic.queries
        vault_context = build_context_blocks(items) if items else ""
        n_docs = vault_context.count("### ")

    if vault_context:
        logger.info(
            "Contexto do vault recuperado (%d docs, %d chars)",
            vault_context.count("### "), len(vault_context),
        )
    else:
        logger.info("Nenhum contexto relevante encontrado no vault")

    rag_files = _RAG_FILE_RE.findall(vault_context)
    return (
        vault_context, decomposed, rag_files, n_docs, sub_queries,
        agentic_searches, agentic_queries,
    )


async def retrieve(
    text: str, *, conv_key: str, skill_name: str | None = None
) -> RetrievalResult:
    """Bloco 1 + skill + Bloco 2 + RAG + Router. Não chama o Bloco 3 (Kimi)."""
    if skill_name is None:
        skill_name = await _resolve_skill_name(text)
    skill = get_skill(skill_name)
    update_context(intent=skill_name)
    logger.info("Skill classificada: %s", skill_name)

    structural_context = await _block1_structural_context()
    priority_folder = await _block2_priority_folder(text, structural_context, skill)
    (
        vault_context, decomposed, rag_files, n_docs, sub_queries,
        agentic_searches, agentic_queries,
    ) = await _run_rag(text, priority_folder, skill_name)

    decision = route(
        text, skill_name,
        decomposed=decomposed, n_sub_queries=len(sub_queries), rag_hits=n_docs,
    )
    update_context(
        intent=decision.intent,
        complexity=decision.complexity,
        tier=decision.tier,
        rag_enabled=bool(vault_context),
        retrieved_documents=n_docs,
    )
    logger.info("Router: %s", decision.as_dict())

    return RetrievalResult(
        skill_name=skill_name, decision=decision,
        structural_context=structural_context, priority_folder=priority_folder,
        vault_context=vault_context, rag_files=rag_files, decomposed=decomposed,
        n_docs=n_docs, sub_queries=sub_queries,
        agentic_searches=agentic_searches, agentic_queries=agentic_queries,
    )


async def _block3(
    text: str, r: RetrievalResult, skill: dict, history: list[Message]
) -> tuple[str, bool]:
    block3_context = build_block3_context(r.structural_context, r.priority_folder)
    try:
        # Resposta decomposta tem N itens numerados + citação de fonte por item —
        # 300 tokens não cobrem uma pergunta de 7 sub-itens.
        num_predict = 600 if r.decomposed else 300
        reply = await generate_response(
            text, block3_context, r.vault_context, history,
            num_predict=num_predict, prompt_injection=skill["prompt_injection"],
            provider_name=r.decision.provider if r.decision.router_enabled else None,
            intent=r.decision.intent if r.decision.router_enabled else None,
        )
        return reply, True
    except _NET_ERRORS as exc:
        logger.warning("Falha ao gerar resposta (Bloco 3): %r", exc)
        return "Yaannk está processando, tenta de novo em instantes.", False


async def answer(
    text: str,
    *,
    conv_key: str,
    autor: str | None = None,
    history: list[Message] | None = None,
    request_id: str | None = None,
    on_slow_path: Callable[[], Awaitable[None]] | None = None,
) -> PipelineResult:
    """Pipeline completo.

    `on_slow_path` é chamado (uma vez) logo antes do Bloco 1, quando o fast-path
    SQL não resolveu — o webhook usa isso para mandar o ack. `history=None` →
    busca do banco; o benchmark passa `[]`.
    """
    request_id = request_id or uuid.uuid4().hex[:12]
    set_context(request_id)

    # Fase B: balanço, financeiro, bills, lista de compras e datas importantes
    # agora vivem no Agent Vida (app/agents/agent_vida.py) — `try_fast_path()`
    # tenta os 5 passos determinísticos sem instanciar o orquestrador (D-05).
    agent_vida = get_agent("vida")
    fast_response = await agent_vida.try_fast_path(text, conv_key, autor)
    if fast_response is not None:
        update_context(
            intent="financial_query", rag_enabled=False, retrieved_documents=0
        )
        logger.info(
            "Agent Vida (fast-path, source=%s) — resposta gerada (%d chars)",
            fast_response.source, len(fast_response.text),
        )
        return PipelineResult(
            reply=fast_response.text, source=fast_response.source, succeeded=True,
            request_id=request_id, skill_name="vida_casal", intent="financial_query",
        )

    if on_slow_path is not None:
        await on_slow_path()

    if history is None:
        history = get_recent_messages(conv_key)

    routing = await _resolve_routing(text)
    skill_name = routing.skill_name

    # Fases C/D: se a skill tem agente registrado, o caminho lento é
    # despachado pra ele. Os agentes RAG são wrappers finos sobre
    # `retrieve()`/`_block3()` com `skill_name` fixo — mesmo comportamento,
    # só com dono explícito. `agent=None` (ex.: "pessoas"/"pendencias", que
    # não têm agente até hoje) segue o caminho genérico abaixo.
    agent = get_agent(routing.agent)
    if agent is not None:
        response = await agent.handle(
            text, routing=routing, history=history, sender=autor or "",
            conv_key=conv_key,
        )
        return PipelineResult(
            reply=response.text, source="llm", succeeded=response.succeeded,
            request_id=request_id,
            skill_name=skill_name, intent=response.intent, decision=response.decision,
            rag_files=response.rag_files, vault_context=response.vault_context,
            decomposed=response.decomposed, n_docs=response.n_docs,
        )

    r = await retrieve(text, conv_key=conv_key, skill_name=skill_name)
    reply, succeeded = await _block3(text, r, get_skill(r.skill_name), history)

    return PipelineResult(
        reply=reply, source="llm", succeeded=succeeded, request_id=request_id,
        skill_name=r.skill_name, intent=r.decision.intent, decision=r.decision,
        rag_files=r.rag_files, vault_context=r.vault_context,
        decomposed=r.decomposed, n_docs=r.n_docs,
    )
