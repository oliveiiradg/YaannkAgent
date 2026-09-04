import asyncio
import json
import logging
import math
import re
from pathlib import Path

import httpx

from app.config import settings
from app.services.vault_reranker import rerank
from app.services.vault_search import (
    _MAX_FILES,
    _TOTAL_CHARS_CAP,
    find_note_by_tipo,
    safe_vault_path,
    search_vault,
    search_vault_ranked,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 8.0
_SEARCH_RESULTS_META_KEY = "io.github.istefox.mcp-connector/searchResults"
_RRF_K = 60
# Cada lado busca mais candidatos que o pool final do RRF, para dar à fusão
# material suficiente para recolocar em alta um arquivo que só um dos dois
# métodos rankeou bem.
_RRF_CANDIDATE_POOL = 15
# O reranker (cross-encoder) reavalia esse tanto do topo do RRF...
_RERANK_POOL = 6
# ...e devolve só isso para o Bloco 3 (contexto menor e mais preciso).
_RERANK_TOP_K = 3
# Corte do snippet SÓ para o scoring do reranker — o tokenizer do cross-encoder
# já trunca em 512 tokens, então um valor perto disso não perde nada de sinal e
# o forward pass ONNX (capado em 512 tokens) tem custo idêntico ao de 350.
_RERANK_SNIPPET_CHARS = 512
# Corte do snippet que vai para o Bloco 3 (o LLM) — bem maior que o do reranker.
# Cada bloco é UMA seção da nota (ver _fresh_snippet), então o corte por bloco
# raramente morde; fica como teto de segurança. 3 × 1000 < _TOTAL_CHARS_CAP.
_CONTEXT_SNIPPET_CHARS = 1000

# Perguntas "meta" sobre bugs/erros ("último erro corrigido no gateway", "quais
# bugs já resolvemos") não casam com nenhum chunk específico da nota-catálogo
# `YaannkAgent - Bugs e Erros.md` (cada bug embeda como seu tópico: Redis,
# systemd, stopwords…). O reranker acerta essa nota quando ela chega ao pool —
# o problema é só recall. Quando a pergunta tem intenção de bug, injetamos a
# nota `tipo: bugs` no pool do reranker e deixamos ele decidir a posição.
_BUG_INTENT_KEYWORDS = (
    "bug", "bugs", "erro", "erros", "problema", "problemas", "falha", "falhas",
    "corrigido", "corrigidos", "corrigimos", "corrigir", "consertado",
    "consertamos", "resolvido", "resolvidos", "resolvemos",
)


# Trecho da nota-catálogo quando ela é forçada no contexto final — precisa
# caber todos os bugs (o arquivo tem ~3.5KB), ainda sob o _TOTAL_CHARS_CAP.
_CATALOG_FORCED_CHARS = 3500


def _has_bug_intent(query: str) -> bool:
    low = query.lower()
    return any(re.search(rf"\b{re.escape(k)}\b", low) for k in _BUG_INTENT_KEYWORDS)


def _strip_frontmatter(text: str) -> str:
    """Remove o frontmatter YAML e o cabeçalho de navegação da nota (H1,
    blockquotes de `> Conecta-se com...`, separadores `---` e linhas vazias),
    começando no primeiro trecho de conteúdo real — tipicamente a 1ª seção
    `## `. Sem isso o modelo pequeno ancora numa linha de navegação
    (ex.: "> Projeto pessoal, sem relação com emprego anterior") como se
    fosse a resposta."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end+3:].lstrip()

    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s == "---" or s.startswith("# ") or s.startswith(">"):
            start = i + 1
            continue
        break
    return "\n".join(lines[start:]).strip()


def _catalog_snippet(content: str, limit: int = _RERANK_SNIPPET_CHARS) -> str:
    """Snippet de uma nota-catálogo: pula frontmatter, H1, blockquote de
    navegação e linhas de separador, e começa no primeiro trecho de conteúdo
    real (tipicamente a primeira seção `## `). `limit` controla o corte —
    curto para o reranker, generoso quando a nota é forçada no contexto."""
    from app.services.vault_search import _FRONTMATTER_RE

    body = _FRONTMATTER_RE.sub("", content, count=1)
    return _strip_frontmatter(body)[:limit]


def _fresh_snippet(
    file_path: str, mcp_excerpt: str, limit: int = _CONTEXT_SNIPPET_CHARS
) -> str:
    """Relê o arquivo do disco e devolve a SEÇÃO que casou com a busca.
    O `excerpt` do MCP pode estar defasado — o índice do plugin do Obsidian
    não re-embeda edições feitas fora do editor. Aqui o excerpt só serve para
    reposicionar no trecho certo; o texto vem do disco e é cortado no próximo
    heading (`#`/`##`), pra não despejar a nota inteira no contexto.
    Se o arquivo sumiu (ou o caminho do MCP escaparia da raiz do vault), cai
    no excerpt do MCP."""
    safe = safe_vault_path(file_path)
    if safe is None:
        logger.warning("Snippet: caminho fora do vault ignorado (%r)", file_path)
        return _strip_frontmatter(mcp_excerpt)[:limit]
    try:
        raw = safe.read_text(encoding="utf-8")
    except OSError:
        return _strip_frontmatter(mcp_excerpt)[:limit]

    body = _strip_frontmatter(raw)
    probe = _strip_frontmatter(mcp_excerpt).strip()[:80]
    start = body.find(probe) if probe else -1
    rest = body[start:] if start != -1 else body
    end = re.search(r"\n#{1,2} ", rest[1:])
    section = rest[: end.start() + 1] if end else rest
    return section.strip()[:limit]


class _McpSearchError(Exception):
    """Erro ao chamar o MCP do Obsidian — sinaliza que deve cair no fallback TF-IDF."""


def _parse_mcp_response(response: httpx.Response) -> dict:
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:"):].strip())
        raise _McpSearchError("Resposta SSE do MCP sem linha 'data:'")
    return response.json()


async def _call_search_vault_smart(query: str, limit: int, priority_folder: str | None) -> list[dict]:
    arguments: dict = {"query": query, "limit": limit}
    if priority_folder:
        arguments["filter"] = {"includeFolders": [priority_folder]}

    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "search_vault_smart", "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {settings.obsidian_mcp_token}",
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(settings.obsidian_mcp_url, json=body, headers=headers)
        response.raise_for_status()
        payload = _parse_mcp_response(response)

    if "error" in payload:
        raise _McpSearchError(f"MCP retornou erro JSON-RPC: {payload['error']}")

    result = payload.get("result", {})
    if result.get("isError"):
        raise _McpSearchError(f"MCP retornou isError: {result.get('content')}")

    rows = result.get("_meta", {}).get(_SEARCH_RESULTS_META_KEY, {}).get("rows", [])
    return rows


async def _call_get_backlinks(file_path: str) -> list[dict]:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "get_backlinks", "arguments": {"path": file_path}},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {settings.obsidian_mcp_token}",
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        response = await client.post(settings.obsidian_mcp_url, json=body, headers=headers)
        response.raise_for_status()
        payload = _parse_mcp_response(response)

    if "error" in payload:
        raise _McpSearchError(f"MCP retornou erro JSON-RPC: {payload['error']}")

    result = payload.get("result", {})
    if result.get("isError"):
        raise _McpSearchError(f"MCP retornou isError: {result.get('content')}")

    content_text = result["content"][0]["text"]
    parsed = json.loads(content_text)
    return parsed.get("backlinks", [])


async def _backlink_bonus(file_path: str) -> float:
    """log(1 + n_backlinks) * peso — falha silenciosa (retorna 0.0) se o
    MCP não responder, para não quebrar o pipeline por causa de um sinal
    secundário."""
    try:
        backlinks = await _call_get_backlinks(file_path)
        return math.log(1 + len(backlinks)) * settings.backlink_weight
    except Exception as exc:
        logger.warning("Backlinks: falha ao consultar %s (%r) — bonus 0", file_path, exc)
        return 0.0


async def enrich_with_backlinks(candidates: list[dict]) -> list[dict]:
    """Soma um bônus de backlinks ao rrf_score de cada candidato — arquivos
    muito linkados (tipicamente MOCs/hubs) sobem no ranking.

    NÃO CHAMADA em produção (sessão 5, 26/08/2026): testada e revertida —
    o bônus domina o score global mesmo pra arquivos irrelevantes muito
    linkados (ex: uma nota de créditos com 19 backlinks venceu a pessoa
    certa, que tinha só 4). Mantida no arquivo como referência para uma
    futura versão condicional (só desempatar, não somar sempre). Ver
    YaannkAgent - Bugs e Erros.md.
    """
    bonuses = await asyncio.gather(*(_backlink_bonus(c["filePath"]) for c in candidates))
    for c, bonus in zip(candidates, bonuses):
        c["rrf_score"] += bonus
    return sorted(candidates, key=lambda c: c["rrf_score"], reverse=True)


async def search_vault_semantic_ranked(
    query: str, priority_folder: str | None = None, limit: int = _MAX_FILES
) -> list[dict]:
    """Retorna os arquivos mais relevantes via busca semântica MCP, ranqueados.

    Cada item: {"filePath": str, "snippet": str}. Levanta _McpSearchError em
    qualquer falha (rede, timeout, protocolo, token ausente).
    """
    if not settings.obsidian_mcp_token:
        raise _McpSearchError("OBSIDIAN_MCP_TOKEN não configurado")

    try:
        rows = await _call_search_vault_smart(query, limit, priority_folder)
    except httpx.TimeoutException as exc:
        raise _McpSearchError(f"timeout ({_TIMEOUT}s) chamando o MCP") from exc
    except httpx.HTTPError as exc:
        raise _McpSearchError(f"erro HTTP chamando o MCP: {exc!r}") from exc
    except (KeyError, json.JSONDecodeError) as exc:
        raise _McpSearchError(f"resposta do MCP em formato inesperado: {exc!r}") from exc

    return [{"filePath": row["filePath"], "snippet": row["excerpt"]} for row in rows]


async def search_vault_semantic(
    query: str, priority_folder: str | None = None, limit: int = _MAX_FILES
) -> str:
    """Busca semântica no vault via MCP do Obsidian — mantida para os scripts
    de calibração (scripts/compare_semantic_vs_tfidf.py). O pipeline em
    produção usa search_vault_hybrid(), que funde com o TF-IDF via RRF.
    """
    results = await search_vault_semantic_ranked(query, priority_folder, limit)
    if not results:
        return ""

    blocks = []
    total_len = 0
    for item in results:
        block = f"### {item['filePath']}\n{item['snippet']}"
        if total_len + len(block) > _TOTAL_CHARS_CAP:
            block = block[: _TOTAL_CHARS_CAP - total_len]
        blocks.append(block)
        total_len += len(block)
        if total_len >= _TOTAL_CHARS_CAP:
            break

    return "\n\n".join(blocks)


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = _RRF_K) -> list[dict]:
    """Funde múltiplos rankings via RRF: score = soma de 1/(k + rank).

    rank é 1-indexado dentro de cada lista. Um filePath presente em mais de
    uma lista soma as contribuições — é o efeito desejado: um arquivo bem
    colocado nos dois métodos sobe mais que um bem colocado em só um. Para
    cada filePath, o snippet mantido é o da lista onde ele teve o melhor
    (menor) rank individual.
    """
    scores: dict[str, float] = {}
    best_snippet: dict[str, tuple[int, str]] = {}

    for ranked_list in ranked_lists:
        for idx, item in enumerate(ranked_list):
            rank = idx + 1
            file_path = item["filePath"]
            scores[file_path] = scores.get(file_path, 0.0) + 1.0 / (k + rank)

            current_best = best_snippet.get(file_path)
            if current_best is None or rank < current_best[0]:
                best_snippet[file_path] = (rank, item["snippet"])

    fused = [
        {"filePath": file_path, "snippet": best_snippet[file_path][1], "rrf_score": score}
        for file_path, score in scores.items()
    ]
    fused.sort(key=lambda item: item["rrf_score"], reverse=True)
    return fused


async def search_vault_hybrid_ranked(
    query: str, priority_folder: str | None = None
) -> list[dict]:
    """Busca híbrida real, devolvendo os ITENS escolhidos (sem formatar).

    Semântico (MCP) + TF-IDF sempre, fundidos via RRF, reordenados por um
    cross-encoder (reranker) sobre o topo do RRF.

    Semântico e TF-IDF sempre contribuem para o ranking — RRF soma o score
    dos dois lados, então um arquivo bem ranqueado em ambos sobe mais que um
    bem ranqueado em só um. Se o MCP falhar, a fusão prossegue só com o
    TF-IDF. Em seguida, o reranker reavalia o top-{_RERANK_POOL} do RRF e
    devolve só o top-{_RERANK_TOP_K} — se o reranker falhar, usa a ordem do
    RRF direto (ver vault_reranker.rerank). enrich_with_backlinks() existe
    no arquivo mas NÃO é chamada aqui — testada e revertida (ver docstring
    dela e YaannkAgent - Bugs e Erros.md).

    Cada item: {"filePath": str, "snippet": str} — mais `keep_full: True` na
    nota-catálogo forçada. Separada de `build_context_blocks()` para que o
    Agentic RAG (app/services/agentic_rag.py) possa acumular itens únicos de
    várias buscas antes de montar um único contexto.
    """
    semantic_results: list[dict] = []
    try:
        semantic_results = await search_vault_semantic_ranked(
            query, priority_folder, limit=_RRF_CANDIDATE_POOL
        )
        logger.info("RRF: busca semântica (MCP) — %d candidatos", len(semantic_results))
    except _McpSearchError as exc:
        logger.warning("RRF: busca semântica falhou (%s) — fusão segue só com TF-IDF", exc)

    tfidf_results = search_vault_ranked(query, priority_folder, limit=_RRF_CANDIDATE_POOL)
    logger.info("RRF: busca TF-IDF — %d candidatos", len(tfidf_results))

    fused = _reciprocal_rank_fusion([semantic_results, tfidf_results])

    # Snippet completo — o reranker corta internamente para o scoring; o corte
    # para o Bloco 3 acontece só no loop de blocos, com _CONTEXT_SNIPPET_CHARS.
    rrf_top = [dict(item) for item in fused[:_RERANK_POOL]]

    # Recall de nota-catálogo (ver _BUG_INTENT_KEYWORDS): perguntas "meta" sobre
    # bugs/erros não casam com nenhum chunk específico da nota `tipo: bugs`, e o
    # reranker (cross-encoder) tende a preferir notas de prosa densa a catálogos.
    # Quando a pergunta tem intenção de bug, a nota `tipo: bugs` é injetada no
    # pool E garantida no contexto final — o reranker escolhe as outras.
    bug_note_path: str | None = None
    if _has_bug_intent(query):
        note = find_note_by_tipo("bugs", priority_folder)
        if note:
            bug_note_path = note["filePath"]
            bug_note_snippet = _catalog_snippet(note["content"])
            if all(item["filePath"] != bug_note_path for item in rrf_top):
                rrf_top.append(
                    {"filePath": bug_note_path, "snippet": bug_note_snippet}
                )
            logger.debug("RRF: nota tipo:bugs injetada (intenção de bug): %s", bug_note_path)

    if not rrf_top:
        return []

    logger.info(
        "RRF: %d arquivos únicos fundidos, top %d vão pro reranker", len(fused), len(rrf_top)
    )
    logger.debug("RRF top: %s", [item["filePath"] for item in rrf_top])

    top = await rerank(
        query, rrf_top, top_k=_RERANK_TOP_K, priority_folder=priority_folder,
        score_chars=_RERANK_SNIPPET_CHARS,
    )

    # Garante a nota-catálogo de bugs no contexto final, à frente dos demais.
    # Vai com um trecho maior que o padrão: é um catálogo curto e o modelo
    # precisa ver todos os bugs para responder "qual foi o último".
    if bug_note_path and all(item["filePath"] != bug_note_path for item in top):
        note = find_note_by_tipo("bugs", priority_folder)
        note_item = {
            "filePath": bug_note_path,
            "snippet": _catalog_snippet(note["content"], limit=_CATALOG_FORCED_CHARS),
            "keep_full": True,  # catálogo forçado: não re-truncar no loop de blocos
        }
        # Forçar significa que o reranker não quis a nota — a pergunta é "meta"
        # sobre bugs. Nesse caso o catálogo É a resposta; deixa ele dominar e
        # corta o resto para 1 bloco, senão o modelo pequeno se ancora numa
        # menção lexical em nota de sessão/pendência ("erros do gateway").
        top = [note_item, *top[:1]]
        logger.info("Reranker: nota tipo:bugs forçada e priorizada no contexto final")

    return top


def build_context_blocks(items: list[dict]) -> str:
    """Monta o Bloco 3 a partir dos itens escolhidos: um bloco
    `### <caminho>.md` por item, snippet relido fresco do disco, tudo sob o
    `_TOTAL_CHARS_CAP`. Puro (a não ser pela leitura de arquivo do
    `_fresh_snippet`) e independente de qual busca produziu os itens."""
    blocks = []
    total_len = 0
    for item in items:
        if item.get("keep_full"):
            # Catálogo forçado: `snippet` já foi lido fresco de find_note_by_tipo.
            snippet = _strip_frontmatter(item["snippet"])
        else:
            snippet = _fresh_snippet(item["filePath"], item["snippet"])
            if len(snippet) > _CONTEXT_SNIPPET_CHARS:
                snippet = snippet[:_CONTEXT_SNIPPET_CHARS]
        block = f"### {item['filePath']}\n{snippet}"
        if total_len + len(block) > _TOTAL_CHARS_CAP:
            block = block[: _TOTAL_CHARS_CAP - total_len]
        blocks.append(block)
        total_len += len(block)
        if total_len >= _TOTAL_CHARS_CAP:
            break

    return "\n\n".join(blocks)


async def search_vault_hybrid(query: str, priority_folder: str | None = None) -> str:
    """Busca híbrida (RRF + reranker) já formatada como contexto do Bloco 3.

    Composição de `search_vault_hybrid_ranked()` + `build_context_blocks()`.
    Mantida com a assinatura de sempre — é o que `multi_search()` (ramo
    decomposto) e os scripts de calibração chamam.
    """
    items = await search_vault_hybrid_ranked(query, priority_folder)
    if not items:
        return ""
    return build_context_blocks(items)
