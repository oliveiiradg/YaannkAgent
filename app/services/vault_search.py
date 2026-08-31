import math
import re
import time
from collections import Counter
from pathlib import Path

from app.config import settings
from app.services.ollama_client import ollama_chat

_EXCLUDED_DIRS = {".obsidian", ".trash", ".claude", ".claudian", ".git"}
_STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "é", "em",
    "no", "na", "nos", "nas", "um", "uma", "uns", "umas", "para", "por",
    "com", "sem", "que", "qual", "quais", "quando", "onde", "como", "se",
    "ou", "mas", "meu", "minha", "meus", "minhas", "seu", "sua", "seus",
    "suas", "isso", "isto", "esse", "essa", "este", "esta", "eu", "voce",
    "você", "vc", "the", "is", "are", "of", "to", "in", "on", "sobre",
    "já", "ja", "ainda", "aqui", "ali", "lá", "la", "só", "so", "muito",
    "muitos", "muita", "muitas", "todo", "toda", "todos", "todas", "tudo",
    "nada", "algo", "alguns", "algumas", "outro", "outra", "outros",
    "outras", "vou", "vai", "foi", "ser", "ter", "tem", "tinha", "estou",
    "está", "esta", "estava", "preciso", "precisa", "precisava", "saber",
}
# Nomes próprios / termos que aparecem em quase todo arquivo do vault e não
# discriminam — configurável por deploy (EXTRA_STOPWORDS no .env, csv).
_STOPWORDS |= set(settings.extra_stopwords)
_WORD_RE = re.compile(r"[a-zà-ú0-9]+", re.IGNORECASE)

_MAX_FILES = 5
_SNIPPET_CHARS = 1200
_TOTAL_CHARS_CAP = 6000
_MIN_DISTINCT_KEYWORDS = 3
_MIN_SCORE = 1.0

_RECENCY_WINDOW_DAYS = 14
_RECENCY_MAX_BOOST = 0.6
_STATE_FILE_BOOST = 1.3
_PRIORITY_FOLDER_BOOST = 1.8


async def identify_relevant_folder(question: str, structural_context: str) -> str | None:
    if not structural_context:
        return None

    messages = [
        {
            "role": "system",
            "content": (
                "Você identifica em qual pasta de projeto do vault a "
                "resposta a uma pergunta provavelmente está, com base no "
                "mapa de projetos fornecido. Responda APENAS com o caminho "
                "exato da pasta (copiado do mapa, sem alterar), ou com a "
                "palavra NENHUMA se não for possível identificar com "
                "confiança. Não explique, não adicione texto extra."
            ),
        },
        {
            "role": "user",
            "content": f"Mapa de projetos:\n{structural_context}\n\nPergunta: {question}",
        },
    ]
    answer = await ollama_chat(messages, num_predict=60, block="bloco2")
    answer = answer.strip().strip("`").strip()

    if not answer or answer.upper().startswith("NENHUMA"):
        return None

    candidate = (Path(settings.vault_path) / answer).resolve()
    root = Path(settings.vault_path).resolve()
    if not str(candidate).startswith(str(root)) or not candidate.is_dir():
        return None

    return answer


def _keywords(query: str) -> list[str]:
    words = _WORD_RE.findall(query.lower())
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


def _iter_vault_files() -> list[Path]:
    root = Path(settings.vault_path)
    files = []
    for path in root.rglob("*.md"):
        if any(part in _EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def _word_counts(text: str) -> Counter:
    return Counter(_WORD_RE.findall(text.lower()))


def _snippet_around_match(content: str, counts: Counter, keywords: list[str]) -> str:
    content_lower = content.lower()
    present = [kw for kw in keywords if counts.get(kw)]

    positions = sorted(
        m.start()
        for kw in present
        for m in re.finditer(rf"\b{re.escape(kw)}\b", content_lower)
    )
    if not positions:
        return content[:_SNIPPET_CHARS].strip()

    # Pick the window with the most keyword hits, not just the first match —
    # a lone match in the frontmatter tags would otherwise anchor the whole
    # snippet on metadata instead of the relevant body text.
    best_start = positions[0]
    best_count = 0
    for p in positions:
        window_end = p + _SNIPPET_CHARS
        count = sum(1 for q in positions if p <= q < window_end)
        if count > best_count:
            best_count = count
            best_start = p

    start = max(0, best_start - 200)
    end = min(len(content), start + _SNIPPET_CHARS)
    return content[start:end].strip()


def _boost_factor(path: Path, root: Path, priority_folder: str | None) -> float:
    now_ts = time.time()
    try:
        age_days = max(0.0, (now_ts - path.stat().st_mtime) / 86400)
    except OSError:
        age_days = _RECENCY_WINDOW_DAYS
    recency_factor = 1.0 + max(
        0.0, (_RECENCY_WINDOW_DAYS - age_days) / _RECENCY_WINDOW_DAYS
    ) * _RECENCY_MAX_BOOST

    state_factor = _STATE_FILE_BOOST if path.name.lower().startswith("state") else 1.0

    folder_factor = 1.0
    if priority_folder:
        rel = str(path.relative_to(root))
        if rel.startswith(priority_folder.rstrip("/")):
            folder_factor = _PRIORITY_FOLDER_BOOST

    return recency_factor * state_factor * folder_factor


def search_vault_ranked(
    query: str, priority_folder: str | None = None, limit: int = _MAX_FILES
) -> list[dict]:
    """Retorna os arquivos mais relevantes via TF-IDF, ranqueados.

    Cada item: {"filePath": str, "snippet": str}. Lista vazia se nada
    passar nos gates de relevância (_MIN_DISTINCT_KEYWORDS / _MIN_SCORE).
    Primitivo usado tanto por search_vault() (string formatada, mantida
    para compatibilidade) quanto pela fusão RRF em vault_search_semantic.py.
    """
    keywords = _keywords(query)
    if not keywords:
        return []

    docs: list[tuple[Path, str, Counter]] = []
    for path in _iter_vault_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        docs.append((path, content, _word_counts(content)))

    if not docs:
        return []

    doc_freq = {
        kw: sum(1 for _, _, counts in docs if counts.get(kw)) for kw in keywords
    }
    n_docs = len(docs)
    idf = {
        kw: math.log((n_docs + 1) / (doc_freq[kw] + 1)) + 1 for kw in keywords
    }

    min_distinct = min(_MIN_DISTINCT_KEYWORDS, len(keywords))
    root = Path(settings.vault_path)

    # The minimum-relevance gate below uses the raw topical score only —
    # boosts (recency, STATE.md, priority folder) affect ranking among
    # already-qualified files, never let an off-topic file pass the gate.
    scored: list[tuple[float, Path, str, Counter]] = []
    for path, content, counts in docs:
        matched = [kw for kw in keywords if counts.get(kw)]
        if len(matched) < min_distinct:
            continue
        tf_idf = sum(counts[kw] * idf[kw] for kw in matched)
        content_len = sum(counts.values()) or 1
        raw_score = tf_idf / math.sqrt(content_len)
        if raw_score >= _MIN_SCORE:
            boosted_score = raw_score * _boost_factor(path, root, priority_folder)
            scored.append((boosted_score, path, content, counts))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for _, path, content, counts in scored[:limit]:
        snippet = _snippet_around_match(content, counts, keywords)
        results.append({"filePath": str(path.relative_to(root)), "snippet": snippet})
    return results


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _frontmatter_tipo(content: str) -> str | None:
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return None
    for line in m.group(1).splitlines():
        line = line.strip()
        if line.lower().startswith("tipo:"):
            return line.split(":", 1)[1].strip().strip("\"'").lower()
    return None


def find_note_by_tipo(tipo: str, folder: str | None = None) -> dict | None:
    """Localiza uma nota pelo campo `tipo` do frontmatter (ex: `tipo: bugs`).

    Retorna {"filePath": str, "content": str} da primeira correspondência —
    dando preferência a uma que esteja dentro de `folder`, se informado.
    Serve para dar recall a notas-catálogo (bugs, índices) que a busca
    semântica não alcança em perguntas genéricas ("último erro corrigido").
    """
    tipo = tipo.lower()
    root = Path(settings.vault_path)
    fallback: dict | None = None
    for path in _iter_vault_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _frontmatter_tipo(content) != tipo:
            continue
        rel = str(path.relative_to(root))
        hit = {"filePath": rel, "content": content}
        if folder and rel.startswith(folder.rstrip("/")):
            return hit
        fallback = fallback or hit
    return fallback


def search_vault(query: str, priority_folder: str | None = None) -> str:
    results = search_vault_ranked(query, priority_folder)
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
