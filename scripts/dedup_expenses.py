"""Remove linhas duplicadas da tabela `expenses` — script one-shot.

Duas linhas são consideradas a mesma quando batem em
`(chat_id, data, ROUND(valor,2), descricao, autor)` — a mesma chave usada pelo
`scripts/backfill_expenses.py` para decidir se já inseriu. De cada grupo
duplicado sobrevive o **menor id** (o registro mais antigo).

Idempotente por construção: rodando de novo, nada resta para remover.

Uso:
  venv/bin/python scripts/dedup_expenses.py --dry-run
  venv/bin/python scripts/dedup_expenses.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.db import get_connection

_GROUP_BY = "chat_id, data, ROUND(valor, 2), descricao, autor"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="não deleta, só mostra")
    args = ap.parse_args()

    conn = get_connection()
    try:
        total_antes = conn.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]

        grupos = conn.execute(
            f"""
            SELECT chat_id, data, ROUND(valor, 2), descricao, autor,
                   COUNT(*)                    AS n,
                   MIN(id)                     AS keep_id,
                   GROUP_CONCAT(id)            AS ids
            FROM expenses
            GROUP BY {_GROUP_BY}
            HAVING COUNT(*) > 1
            ORDER BY MIN(id)
            """
        ).fetchall()

        print("Dedup da tabela expenses")
        print(f"linhas antes: {total_antes}")
        print("-" * 66)

        removidos = 0
        for chat_id, data, valor, descricao, autor, n, keep_id, ids in grupos:
            drop = [i for i in (int(x) for x in ids.split(",")) if i != keep_id]
            removidos += len(drop)
            print(f"  {data}  R$ {valor:.2f}  {descricao} ({autor})")
            print(f"    {n} ocorrências · mantém id={keep_id} · remove ids={drop}")

        if not grupos:
            print("  (nenhuma duplicata encontrada)")

        if not args.dry_run and removidos:
            conn.execute(
                f"""
                DELETE FROM expenses WHERE id NOT IN (
                    SELECT MIN(id) FROM expenses GROUP BY {_GROUP_BY}
                )
                """
            )
            conn.commit()

        total_depois = conn.execute("SELECT COUNT(*) FROM expenses").fetchone()[0]
    finally:
        conn.close()

    print("-" * 66)
    print(f"grupos duplicados: {len(grupos)}")
    print(f"linhas removidas:  {removidos if not args.dry_run else 0}")
    print(f"linhas restantes:  {total_depois}")
    if args.dry_run:
        print(f"\n[DRY-RUN — nada foi removido; removeria {removidos}]")


if __name__ == "__main__":
    main()
