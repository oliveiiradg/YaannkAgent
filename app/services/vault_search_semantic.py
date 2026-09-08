import asyncio
import json
import logging
import math
import re
import unicodedata
from pathlib import Path

import httpx

from app.config import settings
from app.services.skills import SKILLS
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

# Piso e teto da expansão de seção curta (Sessão 22). O piso é o que faz um
# documento valer a vaga que ocupa no top-K; o teto impede que uma nota com
# headings esparsos monopolize o contexto. O corte final continua sendo o
# `limit` de quem chama (512 no reranker, 1000 no Bloco 3), então o teto só
# morde quando o chamador pede mais que isso.
# Seleção de seções da nota-catálogo por relevância (Sessão 22).
_CATALOG_TOP_SECOES = 3
_CATALOG_MIN_KEYWORDS = 2

_MIN_FRESH_CHARS = 300
_MAX_FRESH_CHARS = 2000

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
    # Sessão 22: `tec-18` ("por que um nome próprio comum demais dava falso
    # positivo no RAG?") descreve um bug sem usar a palavra — a resposta está
    # em `Bugs e Erros.md` e a nota não era injetada.
    "falso positivo", "falsos positivos", "falso-positivo", "falsos-positivos",
)


# Trecho da nota-catálogo quando ela é forçada no contexto final — precisa
# caber todos os bugs (o arquivo tem ~3.5KB), ainda sob o _TOTAL_CHARS_CAP.
_CATALOG_FORCED_CHARS = 3500


def _has_bug_intent(query: str) -> bool:
    low = query.lower()
    return any(re.search(rf"\b{re.escape(k)}\b", low) for k in _BUG_INTENT_KEYWORDS)


# Mesmo problema de recall da nota-catálogo de bugs, um nível acima: o
# `YaannkAgent - STATE.md` (`tipo: state`) é a nota de estado do próprio agente
# — longa e heterogênea (repo, fases, infra, decisões). Nenhum chunk dela casa
# bem com uma pergunta técnica inteira ("qual o estado atual do projeto", "como
# funciona o pipeline"), então ela não chega ao pool. Quando a intent é
# `yaannk_tecnico`, injetamos a nota no pool do reranker E garantimos ela no
# contexto final — mesmo padrão de `tipo: bugs`.
_STATE_INTENT = "yaannk_tecnico"
_STATE_FOLDER = SKILLS[_STATE_INTENT]["priority_folder"]
# Trecho do STATE quando ele é forçado no contexto final. Menor que o do
# catálogo de bugs: o STATE é uma nota longa (não um catálogo curto que precisa
# caber inteiro), e 2000 + 2 × _CONTEXT_SNIPPET_CHARS ainda cabe no
# _TOTAL_CHARS_CAP com folga.
_STATE_FORCED_CHARS = 2000


def _yaannk_state_note(intent: str | None) -> dict | None:
    """Nota `tipo: state` do YaannkAgent, quando a intent é `yaannk_tecnico`.

    Ancorada na pasta do projeto de propósito: existe outra nota `tipo: state`
    no vault (um STATE arquivado do DOERJ), e o fallback de
    `find_note_by_tipo()` poderia devolvê-la se a pasta não casasse.
    """
    if intent != _STATE_INTENT:
        return None
    note = find_note_by_tipo("state", _STATE_FOLDER)
    if note and note["filePath"].startswith(_STATE_FOLDER):
        return note
    logger.debug("STATE: nota tipo:state não encontrada em %s", _STATE_FOLDER)
    return None


# --------------------------------------------------------------------------- #
# Query expansion do domínio técnico (Sessão 22 cont. — 08/09/2026)
# --------------------------------------------------------------------------- #
# O vault documenta o roteador como "Tier 0/1/2", "LLM Router",
# "RoutingDecision", "route()"; a pergunta chega como "tiers de roteamento".
# TF-IDF e o embedding do MCP casam por token, não por sinônimo — então
# "quais são os tiers de roteamento?" não trazia NENHUMA das 3 notas que têm
# a resposta. Antes de buscar (só quando `intent == yaannk_tecnico`), a query
# ganha os termos que as notas de fato usam. Tabela estática, sem LLM: o
# jargão do projeto é fechado e estável. NÃO altera a query que vai pro
# reranker nem pro Bloco 3 — só a que alimenta as duas pernas de recuperação.
_EXPANSAO_TECNICA: dict[str, str] = {
    "tier": "Tier 0 Tier 1 Tier 2 LLM Router",
    "tiers": "Tier 0 Tier 1 Tier 2 LLM Router",
    "roteamento": "LLM Router router route RoutingDecision roteador",
    "rotear": "LLM Router router route RoutingDecision",
    "roteador": "LLM Router router route RoutingDecision",
    "router": "LLM Router RoutingDecision route Tier",
    "pipeline": "orquestrador webhook fast-path Bloco 1 Bloco 2 Bloco 3",
    "fluxo": "pipeline orquestrador webhook Bloco",
    "orquestrador": "orchestrator OrchestratorDecision route classify_intent skill",
    "orquestracao": "orchestrator OrchestratorDecision route classify_intent",
    "busca": "RRF reranker TF-IDF semântico híbrido MCP",
    "recuperacao": "RRF reranker TF-IDF semântico RAG",
    "reranker": "cross-encoder bge-reranker-v2-m3 RRF",
    "memoria": "SQLite conversations.db histórico janela chat_id",
    "banco": "SQLite conversations.db",
    "modelo": "Ollama Kimi qwen2.5 OpenRouter",
    "modelos": "Ollama Kimi qwen2.5 OpenRouter",
    "gastos": "expenses dual-write Gastos.md query_expenses record_expense",
    "despesas": "expenses dual-write Gastos.md record_expense",
    "reranqueamento": "reranker cross-encoder RRF",
    "agentico": "Agentic RAG refine juiz Kimi loop refinamento",
    "decomposicao": "decompose_query multi_search sub-query",
}
# Frases (checadas por substring, não por token).
_EXPANSAO_TECNICA_FRASES: dict[str, str] = {
    "banco de dados": "SQLite conversations.db",
    "fast path": "fast-path financial_query query_expenses SQL direto",
    "fast-path": "financial_query query_expenses SQL direto agregação",
    "linguagem natural": "orquestrador Kimi classify_intent",
}
_EXPANSAO_MAX_CHARS = 240


def _norm_ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _expandir_query_tecnica(query: str) -> str:
    """Anexa à query os termos que as notas do YaannkAgent realmente usam,
    a partir de uma tabela de sinônimos do domínio. Idempotente-ish: só
    acrescenta termos ainda ausentes (case-insensitive) e respeita um teto.
    Retorna a query original quando nada casa."""
    base_norm = _norm_ascii(query)
    tokens = re.findall(r"[a-z0-9_]+", base_norm)
    ja_presente = set(tokens)
    adicionais: list[str] = []

    def _absorve(exp: str) -> None:
        for termo in exp.split():
            chave = _norm_ascii(termo).strip("()")
            if chave and chave not in ja_presente:
                adicionais.append(termo)
                ja_presente.add(chave)

    for frase, exp in _EXPANSAO_TECNICA_FRASES.items():
        if frase in base_norm:
            _absorve(exp)

    for token in tokens:
        exp = _EXPANSAO_TECNICA.get(token)
        if exp:
            _absorve(exp)

    if not adicionais:
        return query

    extra = " ".join(adicionais)[:_EXPANSAO_MAX_CHARS].rstrip()
    expandida = f"{query.rstrip('? ').strip()} {extra}"
    logger.info("Query expandida (yaannk_tecnico): +%r", extra)
    return expandida


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


def _stem(palavra: str) -> str:
    """Raiz grosseira para casar variações morfológicas ao pontuar seções:
    tira o plural e corta em 5 chars. "tiers"→"tier" casa com "Tier 0";
    "roteamento"→"rotea" casa com "roteador"/"Router" não, mas com
    "roteamento". Sem isso, "quais são os tiers de roteamento?" não casava
    com a seção "## 4. LLM Router" do STATE e a resposta era NÃO ENCONTRADO
    (`tec-09`). Grosseiro de propósito: a nota já foi escolhida, aqui só se
    decide QUAL seção dela mostrar."""
    return palavra.rstrip("s")[:5]


# Cabeçalho de seção que é diário de sessão, não estado autoritativo.
_SECAO_RECAP_RE = re.compile(r"^#{1,6}\s+Sess(ão|ões|ao)\s", re.IGNORECASE)

# Cabeçalho da seção de catálogo de bugs do STATE. Fora de pergunta com
# intenção de bug ela envenena a seleção: `tec-14` ("qual banco de dados o
# gateway usa?") casava a seção de bugs porque o Bug 9 cita literalmente
# "gateway banco dados PostgreSQL MongoDB" (o exemplo de uma query ruim do
# Agentic RAG) — e o modelo respondia "PostgreSQL e MongoDB". O catálogo de
# bugs tem injeção própria (`_has_bug_intent`); aqui ele só atrapalha.
_SECAO_BUGS_RE = re.compile(r"^#{1,6}\s+.*\bBugs?\b", re.IGNORECASE)

# Acima disso, uma seção `##` do STATE é quase sempre um guarda-chuva
# heterogêneo (o `## 9a` do STATE tem ~11 KB de `###` colados). Para PONTUAR
# qual pedaço mostrar, esses blocos são re-cortados em `###` — senão o
# guarda-chuva casa toda palavra-chave de toda pergunta e vence sempre.
_SECAO_SUBDIVIDE_CHARS = 3000


def _secoes_para_score(body: str) -> list[tuple[int, int]]:
    """Como `_secoes`, mas re-corta em `###` qualquer seção maior que
    `_SECAO_SUBDIVIDE_CHARS` — usado só na seleção lexical de trecho."""
    finas: list[tuple[int, int]] = []
    for ini, fim in _secoes(body):
        if fim - ini <= _SECAO_SUBDIVIDE_CHARS:
            finas.append((ini, fim))
            continue
        sub = [ini + m.start() for m in re.finditer(r"(?m)^### ", body[ini:fim])]
        limites = [ini] + [s for s in sub if s > ini] + [fim]
        finas.extend(
            (limites[j], limites[j + 1]) for j in range(len(limites) - 1)
        )
    return finas


def _secoes_relevantes(body: str, query: str, limit: int) -> str | None:
    """As seções `##` do corpo mais relevantes para `query`, em ordem de
    documento, ou `None` se nenhuma passar do piso.

    Pontuação por sobreposição de palavras-chave (`vault_search._keywords`),
    não pelo cross-encoder: o STATE tem ~30 seções e rerankear todas somaria
    ~5× o trabalho do reranker em toda pergunta técnica. O sinal lexical
    basta aqui porque a nota já foi escolhida — a dúvida é só QUAL pedaço
    dela mostrar.
    """
    from app.services.vault_search import _keywords, _word_counts

    kws = {_stem(k) for k in _keywords(query)}
    if not kws:
        return None

    piso = 1 if len(kws) < 3 else _CATALOG_MIN_KEYWORDS
    pontuadas = []
    for i, (ini, fim) in enumerate(_secoes_para_score(body)):
        trecho = body[ini:fim]
        # Seções de recap de sessão ("## Sessão 22 — …", "### Sessões 19 e 20")
        # são log cronológico, não o estado autoritativo: usam o mesmo
        # vocabulário da pergunta ("Router", "tiers", "SQL") ao DESCREVER o
        # trabalho de uma sessão, e por serem densas em palavra-chave abafavam
        # a seção que de fato responde (`## 4. LLM Router` para `tec-09`).
        # A resposta "o que é X" mora nas seções numeradas `## N.`, não no diário.
        primeira_linha = trecho.lstrip().split("\n", 1)[0]
        if _SECAO_RECAP_RE.match(primeira_linha):
            continue
        if not _has_bug_intent(query) and _SECAO_BUGS_RE.match(primeira_linha):
            continue
        counts: dict[str, int] = {}
        for palavra, n in _word_counts(trecho).items():
            raiz = _stem(palavra)
            if raiz in kws:
                counts[raiz] = counts.get(raiz, 0) + n
        distintas = len(counts)
        if distintas < piso:
            continue
        pontuadas.append((distintas, sum(counts.values()), i, trecho))

    if not pontuadas:
        return None

    # Aceita por SCORE enquanto couber no orçamento, depois devolve em ordem
    # de documento. Cortar a concatenação com um `[:limit]` cru descartava a
    # seção mais relevante sempre que ela aparecia depois de outra no arquivo
    # — foi o que aconteceu em `tec-15`: a seção de infraestrutura (com
    # `yaannk-gateway.service`) era selecionada e depois truncada fora.
    escolhidas: list[tuple] = []
    usado = 0
    for secao in sorted(pontuadas, key=lambda p: (-p[0], -p[1], p[2])):
        trecho = secao[3].strip()
        if len(escolhidas) >= _CATALOG_TOP_SECOES:
            break
        if escolhidas and usado + len(trecho) > limit:
            continue
        escolhidas.append(secao)
        usado += len(trecho) + 2

    if not escolhidas:
        return None
    return "\n\n".join(
        s[3].strip() for s in sorted(escolhidas, key=lambda p: p[2])
    ).strip()[:limit]


def _catalog_snippet(
    content: str, limit: int = _RERANK_SNIPPET_CHARS, query: str | None = None
) -> str:
    """Snippet de uma nota-catálogo: pula frontmatter, H1, blockquote de
    navegação e linhas de separador, e começa no primeiro trecho de conteúdo
    real (tipicamente a primeira seção `## `). `limit` controla o corte —
    curto para o reranker, generoso quando a nota é forçada no contexto.

    Com `query`, escolhe as seções mais relevantes em vez do topo do arquivo
    (Sessão 22). O STATE entra como documento extra em TODA pergunta técnica,
    e o trecho vinha sempre de "## Repositório": perguntar sobre tiers de
    roteamento trazia a seção de repositório e o modelo respondia NÃO
    ENCONTRADO (`tec-09`). Sem query, ou quando nenhuma seção passa do piso,
    mantém o comportamento antigo — o topo costuma ser o resumo da nota.
    """
    from app.services.vault_search import _FRONTMATTER_RE

    body = _strip_frontmatter(_FRONTMATTER_RE.sub("", content, count=1))
    if query:
        secoes = _secoes_relevantes(body, query, limit)
        if secoes:
            return secoes
    return body[:limit]


def _secoes(body: str) -> list[tuple[int, int]]:
    """Spans `(início, fim)` das seções de nível `#`/`##` do corpo da nota.
    O texto antes do primeiro heading (preâmbulo) é a primeira seção."""
    cortes = [m.start() for m in re.finditer(r"(?m)^#{1,2} ", body)]
    limites = [0] + [c for c in cortes if c > 0] + [len(body)]
    return [(limites[i], limites[i + 1]) for i in range(len(limites) - 1)]


def _fresh_snippet(
    file_path: str, mcp_excerpt: str, limit: int = _CONTEXT_SNIPPET_CHARS
) -> str:
    """Relê o arquivo do disco e devolve a SEÇÃO que casou com a busca.
    O `excerpt` do MCP pode estar defasado — o índice do plugin do Obsidian
    não re-embeda edições feitas fora do editor. Aqui o excerpt só serve para
    reposicionar no trecho certo; o texto vem do disco e é cortado no próximo
    heading (`#`/`##`), pra não despejar a nota inteira no contexto.
    Se o arquivo sumiu (ou o caminho do MCP escaparia da raiz do vault), cai
    no excerpt do MCP.

    Sessão 22: seção curta demais é EXPANDIDA com as vizinhas até
    `_MIN_FRESH_CHARS`. Sem isso, um excerpt que caísse numa seção minúscula
    gastava uma vaga do top-3 com ~100 chars de nada — observado com
    `Pendências.md` ("Atualizado ao final de cada sessão... ---") em
    "o que é o RRF no pipeline de busca do Yaannk?".
    """
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

    if len(section.strip()) < _MIN_FRESH_CHARS:
        section = _expande_secao(body, start if start != -1 else 0, section)

    return section.strip()[:limit]


def _expande_secao(body: str, pos: int, section: str) -> str:
    """Cresce a seção com as vizinhas (seguinte primeiro, depois anterior)
    até `_MIN_FRESH_CHARS` ou o fim do arquivo, com teto de
    `_MAX_FRESH_CHARS`. Devolve a `section` original se não houver o que
    anexar."""
    spans = _secoes(body)
    if not spans:
        return section

    atual = next(
        (i for i, (ini, fim) in enumerate(spans) if ini <= pos < fim), 0
    )
    ini, fim = spans[atual]
    antes, depois = atual - 1, atual + 1

    while len(body[ini:fim].strip()) < _MIN_FRESH_CHARS:
        cresceu = False
        if depois < len(spans) and fim - ini < _MAX_FRESH_CHARS:
            fim = spans[depois][1]
            depois += 1
            cresceu = True
        if (
            len(body[ini:fim].strip()) < _MIN_FRESH_CHARS
            and antes >= 0
            and fim - ini < _MAX_FRESH_CHARS
        ):
            ini = spans[antes][0]
            antes -= 1
            cresceu = True
        if not cresceu:
            break

    expandida = body[ini:fim][:_MAX_FRESH_CHARS]
    logger.debug(
        "Snippet: seção de %d chars expandida para %d",
        len(section.strip()), len(expandida.strip()),
    )
    return expandida


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


def _dedup_por_arquivo(ranked: list[dict]) -> list[dict]:
    """Mantém só a primeira (melhor) ocorrência de cada `filePath` na lista,
    preservando a ordem. A perna semântica ranqueia chunks; para o RRF o que
    vale é a posição do ARQUIVO."""
    visto: set[str] = set()
    unicos = []
    for item in ranked:
        if item["filePath"] in visto:
            continue
        visto.add(item["filePath"])
        unicos.append(item)
    return unicos


def _reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = _RRF_K) -> list[dict]:
    """Funde múltiplos rankings via RRF: score = soma de 1/(k + rank).

    rank é 1-indexado dentro de cada lista. Um filePath presente em mais de
    uma lista soma as contribuições — é o efeito desejado: um arquivo bem
    colocado nos dois métodos sobe mais que um bem colocado em só um. Para
    cada filePath, o snippet mantido é o da lista onde ele teve o melhor
    (menor) rank individual.

    A soma vale ENTRE listas, nunca dentro da mesma (Sessão 21): a perna
    semântica devolve *chunks*, não arquivos, e uma nota fatiada em muitos
    pedaços somava uma contribuição por pedaço. Medido em
    "como funciona o pipeline do yaannkagent": `Evolução e Próximos
    Passos.md` apareceu 6 vezes nos 15 candidatos e ficou com 0,107 contra
    0,014 da `Arquitetura.md` — 7,8× de vantagem só por estar mais fatiada,
    além de ocupar 6 dos 15 slots do pool. `_dedup_por_arquivo()` mantém só
    a melhor posição de cada arquivo em cada lista antes da soma.
    """
    scores: dict[str, float] = {}
    best_snippet: dict[str, tuple[int, str]] = {}

    for ranked_list in ranked_lists:
        ranked_list = _dedup_por_arquivo(ranked_list)
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
    query: str, priority_folder: str | None = None, *, intent: str | None = None
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
    # Query expansion do domínio (só para as duas pernas de recuperação — o
    # reranker, `_has_bug_intent`, `_catalog_snippet` etc. seguem com a query
    # crua). Fora de `yaannk_tecnico` a query passa intacta.
    busca_query = (
        _expandir_query_tecnica(query) if intent == _STATE_INTENT else query
    )

    semantic_results: list[dict] = []
    try:
        semantic_results = await search_vault_semantic_ranked(
            busca_query, priority_folder, limit=_RRF_CANDIDATE_POOL
        )
        logger.info("RRF: busca semântica (MCP) — %d candidatos", len(semantic_results))
    except _McpSearchError as exc:
        logger.warning("RRF: busca semântica falhou (%s) — fusão segue só com TF-IDF", exc)

    tfidf_results = search_vault_ranked(
        busca_query, priority_folder, limit=_RRF_CANDIDATE_POOL
    )
    logger.info("RRF: busca TF-IDF — %d candidatos", len(tfidf_results))

    fused = _reciprocal_rank_fusion([semantic_results, tfidf_results])

    # Bug 10 (Sessão 17): descarta candidatos de subpastas de planejamento/spec
    # dentro do escopo certo (priority_folder) mas de fase errada — competiam
    # no RRF/reranker contra a documentação do estado real do código. Aplicado
    # já na fusão, antes do reranker ver qualquer coisa.
    excluded = SKILLS.get(intent, {}).get("excluded_subfolders", []) if intent else []
    if excluded:
        before = len(fused)
        fused = [
            item for item in fused
            if not any(item["filePath"].startswith(prefix) for prefix in excluded)
        ]
        if len(fused) != before:
            logger.info(
                "RRF: %d candidato(s) descartado(s) por excluded_subfolders (intent=%s)",
                before - len(fused), intent,
            )

    # O reranker julga o trecho FRESCO do disco, não o chunk do índice MCP
    # (Sessão 21). O excerpt do MCP é um pedaço arbitrário da nota: em
    # "como funciona o pipeline do yaannkagent" ele devolveu a seção
    # "## Extensão futura" da `Arquitetura.md` — a nota certa, o trecho
    # errado — e o cross-encoder, julgando só isso, a colocou em último.
    # `_fresh_snippet()` reposiciona pelo excerpt e relê do disco até o
    # próximo heading; é a mesma chamada que `build_context_blocks()` já
    # fazia depois, agora antecipada para o reranker ver o mesmo texto que
    # vai pro Bloco 3. Custo: _RERANK_POOL leituras em vez de _RERANK_TOP_K.
    rrf_top = [
        {**item, "snippet": _fresh_snippet(item["filePath"], item["snippet"]), "fresh": True}
        for item in fused[:_RERANK_POOL]
    ]

    # Recall de nota-catálogo (ver _BUG_INTENT_KEYWORDS): perguntas "meta" sobre
    # bugs/erros não casam com nenhum chunk específico da nota `tipo: bugs`, e o
    # reranker (cross-encoder) tende a preferir notas de prosa densa a catálogos.
    # Quando a pergunta tem intenção de bug, a nota `tipo: bugs` é injetada no
    # pool E garantida no contexto final — o reranker escolhe as outras.
    bug_note_path: str | None = None
    bug_note_forced = False
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

    # Recall do STATE (ver _yaannk_state_note): disparado pela intent, não por
    # keyword como a nota-catálogo de bugs.
    #
    # Bug 11 (Sessão 22): o STATE NÃO entra mais no pool do reranker nem
    # disputa vaga no top-K — ele é anexado depois, como documento EXTRA.
    # Antes ele era injetado no pool e, se o reranker não o escolhesse, era
    # posto na frente com `[state_item, *top][:_RERANK_TOP_K]`, o que
    # expulsava o 3º colocado legítimo: em "como funciona o pipeline"
    # derrubava a `Arquitetura.md`, e em `tec-06` ("o que é o RRF") entrava
    # com a seção "Repositório", alheia à pergunta, no lugar de um candidato
    # que o cross-encoder tinha aprovado. Se o STATE for relevante por conta
    # própria, ele chega ao top-K pela busca — e aí não é duplicado.
    state_note = _yaannk_state_note(intent)
    state_note_path: str | None = state_note["filePath"] if state_note else None

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
        bug_note_forced = True
        logger.info("Reranker: nota tipo:bugs forçada e priorizada no contexto final")

    # STATE como documento EXTRA, depois do top-K — nunca no lugar de um dos
    # K melhores (Bug 11, Sessão 22). Fica de fora quando o catálogo de bugs
    # foi forçado: ali o catálogo É a resposta e o corte deliberado para 2
    # blocos (acima) não pode ser diluído.
    # Cabe no orçamento: K × _CONTEXT_SNIPPET_CHARS + _STATE_FORCED_CHARS
    # = 3 × 1000 + 2000 = 5000 < _TOTAL_CHARS_CAP (6000).
    if (
        state_note_path
        and not bug_note_forced
        and all(item["filePath"] != state_note_path for item in top)
    ):
        state_item = {
            "filePath": state_note_path,
            "snippet": _catalog_snippet(
                state_note["content"], limit=_STATE_FORCED_CHARS, query=query
            ),
            "keep_full": True,  # STATE forçado: não re-truncar no loop de blocos
        }
        top = [*top[:_RERANK_TOP_K], state_item]
        logger.info(
            "Reranker: nota tipo:state anexada como documento extra (%s)", state_note_path
        )

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
            # `fresh`: já releu do disco antes do reranker (Sessão 21) — reler
            # aqui daria o mesmo texto ao custo de outro I/O.
            snippet = (
                item["snippet"] if item.get("fresh")
                else _fresh_snippet(item["filePath"], item["snippet"])
            )
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


async def search_vault_hybrid(
    query: str, priority_folder: str | None = None, *, intent: str | None = None
) -> str:
    """Busca híbrida (RRF + reranker) já formatada como contexto do Bloco 3.

    Composição de `search_vault_hybrid_ranked()` + `build_context_blocks()`.
    Mantida com a assinatura de sempre — é o que `multi_search()` (ramo
    decomposto) e os scripts de calibração chamam.
    """
    items = await search_vault_hybrid_ranked(query, priority_folder, intent=intent)
    if not items:
        return ""
    return build_context_blocks(items)
