"""Classificador de intenção — keyword matching simples, sem LLM.

Conta quantas keywords de cada skill aparecem no texto da pergunta e devolve o
nome da skill vencedora. Empate é resolvido por uma ordem de prioridade fixa
(skill com pasta específica vale mais que instrução genérica).
Nenhuma keyword → 'default'.
"""

import logging

from app.services.finance_patterns import FINANCIAL_QUERY_STRICT_RE
from app.services.skills import SKILLS

logger = logging.getLogger(__name__)

# Desempate: skills com pasta prioritária específica primeiro; `vida_casal`
# (uso do dia a dia) na frente.
_TIEBREAK_ORDER = [
    "vida_casal", "yaannk_tecnico", "conhecimento", "carreira",
    "pendencias", "pessoas",
]


def classify_intent(text: str) -> str:
    # Short-circuit (Fase 7): consulta de agregação financeira vai sempre para
    # vida_casal, mesmo sem keyword da skill — habilita o fast-path SQL do
    # webhook (financial_query). Regex estreita, ver finance_patterns.py.
    if FINANCIAL_QUERY_STRICT_RE.search(text):
        logger.info("classify_intent: consulta financeira detectada → vida_casal")
        return "vida_casal"

    low = text.lower()

    scores: dict[str, int] = {}
    for name, skill in SKILLS.items():
        if name == "default":
            continue
        hits = sum(1 for kw in skill["keywords"] if kw in low)
        if hits:
            scores[name] = hits

    if not scores:
        return "default"

    best = max(scores.values())
    for name in _TIEBREAK_ORDER:
        if scores.get(name) == best:
            return name
    # keyword de alguma skill fora da ordem de desempate (não deve acontecer
    # com as skills atuais, mas mantém o classificador robusto)
    return max(scores, key=scores.get)
