"""Backfill da tabela SQL `expenses` a partir do markdown `03 - Vida/Finanças/
Gastos.md` do vault.

Os gastos registrados via WhatsApp antes da Fase 7 (dual-write) só existem no
markdown do Obsidian. Este script lê a tabela `## Registros` dessa nota e insere
no SQLite (`data/conversations.db`) as linhas que ainda não estão lá.

- Lê o arquivo do disco (OneDrive sincroniza), não via MCP.
- Confia nas colunas do markdown (Categoria / Descrição / Registrou verbatim);
  só usa `vault_writer._valor_float` para limpar o "R$ 50".
- Idempotente: antes de inserir, checa (chat_id, data, valor, descricao, autor).
- `--dry-run` mostra o que faria sem gravar.

Uso:
  venv/bin/python scripts/backfill_expenses.py --chat-id <JID> --dry-run
  venv/bin/python scripts/backfill_expenses.py --chat-id 5511999999999@s.whatsapp.net
  venv/bin/python scripts/backfill_expenses.py --chat-id 000000000000000000@g.us --file /caminho/Gastos.md

O `--chat-id` deve casar com o `chat_id` que o gateway grava (o JID do grupo,
tipo `<id>@g.us`, ou o número `<num>@s.whatsapp.net` em conversa individual).
"""

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.db import get_connection
from app.services.vault_writer import _valor_float

_NOTE_REL = "03 - Vida/Finanças/Gastos.md"
_SECTION = "Registros"


class RowError(Exception):
    """Linha da tabela que não dá para converter em lançamento."""


def _extract_section(md: str, heading: str) -> list[str]:
    """Linhas da seção `## {heading}` (exclusive da própria heading), até a
    próxima heading `#`/`##`/... ou o fim do arquivo."""
    lines = md.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lstrip("#").strip().lower() == heading.lower() and ln.lstrip().startswith("#"):
            start = i + 1
            break
    if start is None:
        raise SystemExit(f"seção '## {heading}' não encontrada em {_NOTE_REL}")
    out = []
    for ln in lines[start:]:
        if ln.lstrip().startswith("#"):
            break
        out.append(ln)
    return out


def _table_rows(section_lines: list[str]) -> list[list[str]]:
    """Células das linhas de dados de uma tabela pipe — pula header e separador."""
    rows = []
    for ln in section_lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not cells or set("".join(cells)) <= {"-", ":"}:  # linha separadora
            continue
        rows.append(cells)
    return rows[1:] if rows else []  # rows[0] = header


def _parse_row(cells: list[str]) -> dict:
    if len(cells) < 5:
        raise RowError(f"colunas insuficientes ({len(cells)}): {cells}")
    data_raw, descricao, valor_raw, categoria, autor = (c.strip() for c in cells[:5])

    try:
        data = datetime.datetime.strptime(data_raw, "%d/%m/%Y").date().isoformat()
    except ValueError:
        raise RowError(f"data inválida: {data_raw!r} (esperado dd/mm/aaaa)")

    valor = _valor_float(valor_raw)
    if valor is None:
        raise RowError(f"valor não parseável: {valor_raw!r}")

    return {
        "data": data,
        "valor": round(valor, 2),
        "categoria": categoria or "Outros",
        "descricao": descricao,
        "autor": autor,
    }


def _exists(conn, chat_id: str, row: dict) -> bool:
    hit = conn.execute(
        "SELECT 1 FROM expenses WHERE chat_id = ? AND data = ? "
        "AND ROUND(valor, 2) = ? AND descricao = ? AND autor = ? LIMIT 1",
        (chat_id, row["data"], row["valor"], row["descricao"], row["autor"]),
    ).fetchone()
    return hit is not None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="não grava, só mostra")
    ap.add_argument("--chat-id", required=True, help="JID da conversa (ver docstring)")
    ap.add_argument(
        "--file", type=Path,
        default=Path(settings.vault_path) / _NOTE_REL,
        help=f"default: $VAULT_PATH/{_NOTE_REL}",
    )
    args = ap.parse_args()

    if not args.file.is_file():
        raise SystemExit(f"arquivo não encontrado: {args.file}")

    md = args.file.read_text(encoding="utf-8")
    rows = _table_rows(_extract_section(md, _SECTION))

    print(f"Backfill de expenses — {args.file}")
    print(f"chat_id: {args.chat_id}")
    print(f"linhas na tabela '## {_SECTION}': {len(rows)}")
    print("-" * 60)

    now = datetime.datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    inseridos = ignorados = 0
    erros: list[str] = []

    try:
        for n, cells in enumerate(rows, start=1):
            try:
                row = _parse_row(cells)
            except RowError as exc:
                erros.append(f"linha {n}: {exc}")
                continue

            if _exists(conn, args.chat_id, row):
                ignorados += 1
                print(f"  ~ ignorado  {row['data']}  R$ {row['valor']:.2f}  "
                      f"{row['categoria']:<12} {row['descricao']} ({row['autor']})")
                continue

            print(f"  + inserir   {row['data']}  R$ {row['valor']:.2f}  "
                  f"{row['categoria']:<12} {row['descricao']} ({row['autor']})")
            if not args.dry_run:
                conn.execute(
                    "INSERT INTO expenses "
                    "(chat_id, autor, valor, categoria, descricao, data, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (args.chat_id, row["autor"], row["valor"], row["categoria"],
                     row["descricao"], row["data"], now),
                )
            inseridos += 1

        if not args.dry_run:
            conn.commit()
    finally:
        conn.close()

    print("-" * 60)
    print(f"inseridos: {inseridos}")
    print(f"ignorados: {ignorados}   (já existem na tabela)")
    print(f"com erro:  {len(erros)}")
    for e in erros:
        print(f"  {e}")
    if args.dry_run:
        print("\n[DRY-RUN — nada foi gravado]")


if __name__ == "__main__":
    main()
