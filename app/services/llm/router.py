"""LLM Router determinístico (Fase 5).

Decide, para o **Bloco 3** (geração final), qual tier/provider usar — a partir
de sinais que o pipeline já calculou (skill, decomposição, nº de docs do RAG) e
de heurísticas de complexidade sobre o texto. **Sem LLM aqui** (spec: não criar
Router baseado em LLM sem dados).

Regras fixas que o Router NÃO mexe:
- Bloco 1 (resumo de regras) e Bloco 2 (identificação de pasta) ficam sempre no
  Ollama local (decisão 28/08). O Router só opina sobre o Bloco 3.
- Com `LLM_ROUTER_ENABLED=false` (default) o Bloco 3 vai para
  `LLM_DEFAULT_PROVIDER` — o Router só produz o objeto de decisão para a
  telemetria, sem alterar o roteamento.

Tiers disponíveis nesta fase:
- **0** — Ollama local (`qwen2.5:3b`): mensagem curta, conversa trivial.
- **1** — Kimi K2.6 (`kimi`): RAG médio, parsing, consulta financeira, default.
- **2** — Kimi K2.7-code (`kimi-heavy`): `technical_rag` complexo, 5+ sub-itens,
  múltiplos documentos, raciocínio crítico.
"""

import logging
import re
from dataclasses import dataclass

from app.config import settings
from app.services.finance_patterns import (
    FINANCIAL_QUERY_RE,
    FINANCIAL_QUERY_STRICT_RE,
)

logger = logging.getLogger(__name__)

# skill (intent_classifier) -> intenção no vocabulário da spec
_SKILL_TO_INTENT = {
    "yaannk_tecnico": "technical_rag",
    "pessoas": "general_rag",
    "pendencias": "general_rag",
    "vida_casal": "conversation",
    "default": "general_rag",
}

_TIER_PROVIDER = {0: "ollama", 1: "kimi", 2: "kimi-heavy"}

# Pergunta de valor/agregação financeira ("quanto gastamos em setembro?") —
# regex centralizada em app/services/finance_patterns.py (larga: só é usada
# aqui, já com a skill = vida_casal garantida).
_QUESTION_RE = re.compile(r"\?|\b(quanto|quantos|quantas|quando|qual|quais|onde|"
                          r"por\s*que|porqu[eê]|o\s+que)\b", re.IGNORECASE)

# Sinais de que a resposta precisa de transcrição literal / citação exata.
_EXACT_CITATION_RE = re.compile(
    r"\b(f[oó]rmula|literal(?:mente)?|texto\s+exato|trecho\s+exato|transcreva|"
    r"valor\s+exato|nome\s+exato|c[oó]digo\s+exato|copie|na\s+[ií]ntegra)\b",
    re.IGNORECASE,
)
_REASONING_RE = re.compile(
    r"\b(compare|compara[çc][ãa]o|diferen[çc]a|versus|rela[çc][ãa]o\s+entre|"
    r"analis[ae]|explique\s+por\s+qu[êe]|racioc[ií]nio|implica[çc][õo]es|"
    r"trade[- ]?off|prós\s+e\s+contras)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoutingDecision:
    intent: str
    complexity: int          # 0–10
    tier: int                # 0 | 1 | 2
    provider: str            # 'ollama' | 'kimi' | 'kimi-heavy'
    requires_rag: bool
    requires_reasoning: bool
    requires_exact_citations: bool
    reason: str
    decomposed: bool = False
    n_sub_queries: int = 1
    degraded: bool = False   # provider do tier indisponível → caiu p/ Ollama
    router_enabled: bool = True

    def as_dict(self) -> dict:
        return {
            "intent": self.intent,
            "complexity": self.complexity,
            "tier": self.tier,
            "provider": self.provider,
            "requires_rag": self.requires_rag,
            "requires_reasoning": self.requires_reasoning,
            "requires_exact_citations": self.requires_exact_citations,
            "decomposed": self.decomposed,
            "n_sub_queries": self.n_sub_queries,
            "degraded": self.degraded,
            "router_enabled": self.router_enabled,
            "reason": self.reason,
        }


def _score_complexity(
    text: str, decomposed: bool, n_sub_queries: int, rag_hits: int,
    exact_citations: bool, reasoning: bool,
) -> int:
    score = 0
    if decomposed:
        score += 3 + min(3, max(0, n_sub_queries - 2))  # +1 por sub-item além de 2
    n = len(text)
    if n > 400:
        score += 2
    elif n > 200:
        score += 1
    if exact_citations:
        score += 2
    if reasoning:
        score += 1
    if rag_hits >= 3:
        score += 1
    return max(0, min(10, score))


def _classify_intent(skill_name: str, text: str, rag_hits: int) -> str:
    base = _SKILL_TO_INTENT.get(skill_name, "general_rag")
    if base == "conversation":
        # vida_casal: separa pergunta financeira de conversa/registro.
        if FINANCIAL_QUERY_STRICT_RE.search(text) or (
            FINANCIAL_QUERY_RE.search(text) and _QUESTION_RE.search(text)
        ):
            return "financial_query"
        return "conversation"
    if base == "general_rag" and skill_name == "default" and rag_hits == 0:
        return "conversation"
    return base


def route(
    text: str,
    skill_name: str,
    *,
    decomposed: bool = False,
    n_sub_queries: int = 1,
    rag_hits: int = 0,
) -> RoutingDecision:
    """Decide o provider do Bloco 3. Puro e determinístico."""
    exact_citations = bool(_EXACT_CITATION_RE.search(text))
    reasoning_kw = bool(_REASONING_RE.search(text))
    intent = _classify_intent(skill_name, text, rag_hits)
    requires_rag = rag_hits > 0 or intent in ("technical_rag", "general_rag", "financial_query")

    complexity = _score_complexity(
        text, decomposed, n_sub_queries, rag_hits, exact_citations, reasoning_kw
    )
    requires_reasoning = reasoning_kw or complexity >= 6 or (decomposed and n_sub_queries >= 5)

    # --- escada de decisão -------------------------------------------------
    if intent == "conversation" and complexity <= 2 and not requires_rag:
        tier, reason = 0, "conversa curta e trivial → local"
    elif intent == "technical_rag" and complexity >= 6:
        tier, reason = 2, f"technical_rag complexo (complexidade {complexity}) → heavy"
    elif intent == "technical_rag" and requires_reasoning:
        tier, reason = 2, "technical_rag + raciocínio/comparação → heavy"
    elif intent == "technical_rag" and complexity <= 2 and n_sub_queries <= 1:
        # Pergunta técnica simples e não decomposta: o Ollama local dá conta e
        # evita os 10–50s do Kimi via OpenRouter no Bloco 3.
        tier, reason = 0, f"technical_rag simples (complexidade {complexity}) → local"
    elif complexity >= 8:
        tier, reason = 2, f"complexidade {complexity} → heavy"
    elif intent == "conversation":
        tier, reason = 1, "conversa não-trivial → Kimi"
    else:
        tier, reason = 1, f"{intent} (complexidade {complexity}) → Kimi default"

    provider = _TIER_PROVIDER[tier]
    degraded = False

    # --- disponibilidade do provider -------------------------------------
    if provider in ("kimi", "kimi-heavy") and not settings.kimi_api_key:
        logger.warning(
            "Router: tier %d pedia %s mas KIMI_API_KEY não está configurada — "
            "degradando para Ollama local (intent=%s, complexidade=%d)",
            tier, provider, intent, complexity,
        )
        tier, provider, degraded = 0, "ollama", True
        reason += " [degradado: Kimi indisponível]"

    # --- router desligado: decisão é só informativa ----------------------
    if not settings.llm_router_enabled:
        default = settings.llm_default_provider
        return RoutingDecision(
            intent=intent, complexity=complexity, tier=(0 if default == "ollama" else 1),
            provider=default, requires_rag=requires_rag,
            requires_reasoning=requires_reasoning,
            requires_exact_citations=exact_citations,
            reason=f"router desligado → {default} (sugeria tier {tier}/{provider})",
            decomposed=decomposed, n_sub_queries=n_sub_queries,
            degraded=False, router_enabled=False,
        )

    logger.info(
        "Router: intent=%s complexidade=%d decomposed=%s(%d) → tier %d (%s) — %s",
        intent, complexity, decomposed, n_sub_queries, tier, provider, reason,
    )
    return RoutingDecision(
        intent=intent, complexity=complexity, tier=tier, provider=provider,
        requires_rag=requires_rag, requires_reasoning=requires_reasoning,
        requires_exact_citations=exact_citations, reason=reason,
        decomposed=decomposed, n_sub_queries=n_sub_queries,
        degraded=degraded, router_enabled=True,
    )
