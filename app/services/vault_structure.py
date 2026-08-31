import re
import sqlite3
from pathlib import Path

from app.config import settings
from app.services.db import get_connection
from app.services.ollama_client import ollama_chat

_CLAUDE_MD = "CLAUDE.md"
_CURRENT_MD = "CURRENT.md"
_MTIME_TRIGGER_FILE = _CLAUDE_MD
_PROJECT_ROOT = "01 - Projetos"


def _init_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vault_structure_cache (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            source_mtime REAL NOT NULL,
            summary TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )


def _read_cache(conn: sqlite3.Connection) -> tuple[float, str] | None:
    row = conn.execute(
        "SELECT source_mtime, summary FROM vault_structure_cache WHERE id = 1"
    ).fetchone()
    return (row[0], row[1]) if row else None


def _write_cache(conn: sqlite3.Connection, mtime: float, summary: str) -> None:
    conn.execute(
        """
        INSERT INTO vault_structure_cache (id, source_mtime, summary, updated_at)
        VALUES (1, ?, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            source_mtime = excluded.source_mtime,
            summary = excluded.summary,
            updated_at = excluded.updated_at
        """,
        (mtime, summary),
    )
    conn.commit()


def _read_file(root: Path, name: str) -> str:
    try:
        return (root / name).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _extract_section(md: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)"
    match = re.search(pattern, md, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_table_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(cells)
    return rows[1:] if rows else rows  # skip header row


def _parse_folder_structure(claude_md: str) -> str:
    rows = _parse_table_rows(_extract_section(claude_md, "Estrutura de pastas"))
    return "\n".join(f"{cells[0]}: {cells[1]}" for cells in rows if len(cells) >= 2)


def _parse_active_projects(current_md: str) -> list[tuple[str, str, str]]:
    rows = _parse_table_rows(_extract_section(current_md, "Projetos Ativos"))
    return [(c[0], c[1], c[2]) for c in rows if len(c) >= 3]


def _all_project_dirs(root: Path) -> list[Path]:
    project_root = root / _PROJECT_ROOT
    return [p for p in project_root.rglob("*") if p.is_dir()]


def _match_project_folder(name: str, folders: list[Path], root: Path) -> str | None:
    name_words = {w.lower() for w in re.findall(r"[a-zà-ú0-9]+", name) if len(w) > 2}
    if not name_words:
        return None
    best, best_score = None, 0
    for folder in folders:
        folder_words = {
            w.lower() for w in re.findall(r"[a-zà-ú0-9]+", folder.name) if len(w) > 2
        }
        score = len(name_words & folder_words)
        if score > best_score:
            best, best_score = folder, score
    return str(best.relative_to(root)) if best else None


def _build_folder_map(current_md: str, root: Path) -> str:
    projects = _parse_active_projects(current_md)
    folders = _all_project_dirs(root)
    lines = []
    for name, fase, bloqueio in projects:
        folder = _match_project_folder(name, folders, root)
        folder_str = folder or "(pasta não identificada automaticamente)"
        lines.append(f"- {name} → pasta: {folder_str} | fase/contexto: {fase} | bloqueio: {bloqueio}")
    return "\n".join(lines)


async def _summarize_rules(claude_md: str) -> str:
    section = _extract_section(claude_md, "Regras Anti-Alucinação")
    if not section:
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "Resuma a tabela de regras abaixo em no máximo 3 frases "
                "corridas, em português, mantendo o sentido exato de cada "
                "regra. Não adicione nada que não esteja na tabela, não "
                "invente contexto adicional."
            ),
        },
        {"role": "user", "content": section},
    ]
    # Narrow, single-purpose call — small enough to run reliably within the
    # default timeout even on weak CPU hardware. Fica sempre no Ollama local.
    return await ollama_chat(messages, num_predict=200, timeout=150.0, block="bloco1")


async def _build_summary() -> str:
    root = Path(settings.vault_path)
    claude_md = _read_file(root, _CLAUDE_MD)
    current_md = _read_file(root, _CURRENT_MD)

    # Só a síntese das regras (tarefa estreita) vai para o Ollama — o
    # mapeamento projeto→pasta é mecânico e feito aqui em Python: passar o
    # CURRENT.md inteiro para o modelo de 3B fazia ele tentar "reescrever"
    # o arquivo em vez de extrair dados, mesmo com prompts diretivos.
    rules_summary = await _summarize_rules(claude_md)
    folder_structure = _parse_folder_structure(claude_md)
    folder_map = _build_folder_map(current_md, root)

    parts = []
    if rules_summary:
        parts.append(f"REGRAS PRINCIPAIS:\n{rules_summary}")
    if folder_structure:
        parts.append(f"ESTRUTURA DE PASTAS:\n{folder_structure}")
    if folder_map:
        parts.append(f"PROJETOS ATIVOS → PASTA:\n{folder_map}")

    return "\n\n".join(parts)


def build_block3_context(full_context: str, priority_folder: str | None) -> str:
    """Trims the structural context down to what Block 3 should see.

    The full multi-project map (all active projects listed together) is what
    Block 2 needs to pick a folder — but feeding that same list into Block 3
    lets the model cross-reference the wrong project's fase/bloqueio when
    answering (seen empirically: a project status question answered with
    another project's current sprint data, because both lines sat next to
    each other in context).
    Block 3 gets the rules + folder structure always, plus only the single
    map line for the project Block 2 already identified (if any) — dropping
    the whole map loses real, useful summary info; keeping all of it risks
    conflating projects.
    """
    marker = "\n\nPROJETOS ATIVOS → PASTA:"
    idx = full_context.find(marker)
    if idx == -1:
        return full_context

    base = full_context[:idx]
    if not priority_folder:
        return base

    map_section = full_context[idx + len(marker):]
    matching_lines = [
        line
        for line in map_section.splitlines()
        if line.strip().startswith("-") and priority_folder in line
    ]
    if not matching_lines:
        return base

    return base + "\n\nPROJETO RELEVANTE:\n" + "\n".join(matching_lines)


async def get_structural_context() -> str:
    root = Path(settings.vault_path)
    trigger_path = root / _MTIME_TRIGGER_FILE
    current_mtime = trigger_path.stat().st_mtime

    conn = get_connection()
    try:
        _init_table(conn)
        cached = _read_cache(conn)
        if cached and cached[0] == current_mtime:
            return cached[1]
    finally:
        conn.close()

    summary = await _build_summary()

    conn = get_connection()
    try:
        _init_table(conn)
        _write_cache(conn, current_mtime, summary)
    finally:
        conn.close()

    return summary
