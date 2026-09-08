"""Query decomposition — Tarefa 1 do roadmap (YaannkAgent - Evolução).

Perguntas reais chegam de outra IA (Claude Web / Copilot): estruturadas, com
vários sub-itens e vários termos de busca. O `qwen2.5:3b` não raciocina sobre
múltiplos documentos ao mesmo tempo. A saída: em vez de 1 busca com a pergunta
inteira, detectar os sub-itens e fazer N buscas independentes, montando um
contexto já separado por item. O LLM só formata o que o RAG já encontrou.

Sem LLM extra aqui — só regex para a decomposição e o `search_vault_hybrid()`
existente para cada sub-busca.
"""

import asyncio
import logging
import re

from app.services.vault_search_semantic import search_vault_hybrid

logger = logging.getLogger(__name__)

# Teto de sub-buscas por pergunta — cada uma roda MCP + TF-IDF + reranker (CPU
# no notebook), então mais que isso deixa a resposta lenta demais.
_MAX_SUB_QUERIES = 8

# Concorrência das sub-buscas — o reranker roda em CPU pura no i5 7ª gen;
# duas em paralelo é o limite antes de brigar por núcleo.
_SEARCH_CONCURRENCY = 2

_NOT_FOUND = "NÃO ENCONTRADO"

# "1." "1)" "1 -" "1 –" no início da linha, seguido de conteúdo.
_NUMBERED_RE = re.compile(r"^\s*(\d{1,2})\s*[.)\-–]\s+(.+?)\s*$")
# Marcador numerado no meio do texto ("... responda: 1. x 2. y 3. z") — para
# quando o WhatsApp entrega a lista toda numa linha só.
_INLINE_NUMBERED_RE = re.compile(r"(?:(?<=\s)|^)(\d{1,2})[.)]\s+")
# "- " "* " "• " no início da linha.
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+?)\s*$")
# Termos entre aspas separados por barra: "Extremo"/"Sem Nível"/"Fora do intervalo"
_QUOTED_SLASH_RE = re.compile(r'"[^"]+"(?:\s*/\s*"[^"]+")+')


def _clean(item: str) -> str:
    return item.strip().rstrip(";.").strip()


def _inline_numbered(text: str) -> list[str]:
    """Sub-itens numerados numa linha só ("1. x 2. y 3. z"). Só aceita se os
    marcadores formarem uma sequência quase contígua (1,2,3... ou 0,1,2...) —
    evita casar datas/valores soltos ("gastei 50 no dia 2)")."""
    marks = list(_INLINE_NUMBERED_RE.finditer(text))
    if len(marks) < 2:
        return []
    nums = [int(m.group(1)) for m in marks]
    if nums[0] not in (0, 1):
        return []
    if any(b - a not in (0, 1) for a, b in zip(nums, nums[1:])):
        return []
    items = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        items.append(_clean(text[m.end():end]))
    return items


def decompose_query(text: str) -> list[str]:
    """Detecta sub-itens numa pergunta estruturada.

    Retorna a lista de sub-queries preservando a ordem original. Se nada for
    detectado, retorna ``[text]`` — o pipeline segue com o comportamento atual
    (1 busca só).
    """
    lines = text.splitlines()

    numbered = [
        _clean(m.group(2))
        for line in lines
        if (m := _NUMBERED_RE.match(line))
    ]
    if len(numbered) >= 2:
        return _finalize(numbered)

    inline = _inline_numbered(text)
    if len(inline) >= 2:
        return _finalize(inline)

    bullets = [
        _clean(m.group(1))
        for line in lines
        if (m := _BULLET_RE.match(line))
    ]
    if len(bullets) >= 2:
        return _finalize(bullets)

    # Sem lista explícita: procura um bloco de termos entre aspas separados
    # por barra e quebra em uma busca por termo.
    slash_match = _QUOTED_SLASH_RE.search(text)
    if slash_match:
        terms = [_clean(t) for t in re.findall(r'"([^"]+)"', slash_match.group(0))]
        if len(terms) >= 2:
            return _finalize(terms)

    return [text]


def _finalize(items: list[str]) -> list[str]:
    items = [i for i in items if i]
    if len(items) > _MAX_SUB_QUERIES:
        logger.info(
            "Query decomposer: %d sub-itens detectados, truncando para %d",
            len(items), _MAX_SUB_QUERIES,
        )
        items = items[:_MAX_SUB_QUERIES]
    return items


async def multi_search(
    sub_queries: list[str], priority_folder: str | None = None,
    *, intent: str | None = None,
) -> list[dict]:
    """Roda ``search_vault_hybrid()`` para cada sub-query, com concorrência
    limitada. Retorna ``[{"idx", "query", "context"}]`` na ordem original —
    ``context`` é ``"NÃO ENCONTRADO"`` quando a busca não trouxe nada.
    """
    semaphore = asyncio.Semaphore(_SEARCH_CONCURRENCY)

    async def _one(idx: int, query: str) -> dict:
        async with semaphore:
            try:
                context = await search_vault_hybrid(
                    query, priority_folder, intent=intent
                )
            except Exception as exc:  # noqa: BLE001 — uma sub-busca não derruba as outras
                logger.warning("Multi-search: sub-query %d falhou (%r)", idx, exc)
                context = ""
        return {
            "idx": idx,
            "query": query,
            "context": context.strip() if context else _NOT_FOUND,
        }

    results = await asyncio.gather(
        *(_one(i, q) for i, q in enumerate(sub_queries, start=1))
    )
    return sorted(results, key=lambda r: r["idx"])


def build_decomposed_context(results: list[dict]) -> str:
    """Monta o contexto estruturado, um bloco por sub-item, para o Bloco 3."""
    blocks = [
        "Contexto estruturado — uma busca no vault por sub-item da pergunta. "
        "Responda cada item usando SÓ o bloco correspondente. Onde constar "
        f"{_NOT_FOUND}, responda exatamente {_NOT_FOUND}."
    ]
    for r in results:
        blocks.append(f"[Item {r['idx']}] {r['query']}\n{r['context']}")
    return "\n\n".join(blocks)
