#!/usr/bin/env python3
"""Migração (Sessão 19): `Gastos.md` (arquivo único cumulativo) ->
`Gastos - YYYY-MM.md` (um por mês).

Decisão da Sessão 19: o VAULT é a fonte de verdade — `Gastos.md` já tem 8
linhas contra 7 no SQLite, e o `record_expense()` (dual-write) é best-effort
("qualquer falha aqui é logada e engolida", ver `vault_writer.py`). Por isso
este script lê do vault, não do SQL. O SQLite só entra ao final, como
conferência: linhas do SQL sem correspondência aproximada no vault são só
IMPRESSAS — nada é escrito a partir do SQL.

Script standalone — não é importado pelo gateway, roda uma vez.

Uso:
    python3 gateway/tools/migrate_expenses_to_vault.py           # escreve de verdade
    python3 gateway/tools/migrate_expenses_to_vault.py --dry-run # só mostra o que geraria
"""

import argparse
import datetime
import re
import sqlite3
import sys
from pathlib import Path

VAULT_PATH = Path.home() / "OneDrive/Documents/Yaannk"
GASTOS_MD = VAULT_PATH / "03 - Vida/Finanças/Gastos.md"
GASTOS_DIR = VAULT_PATH / "03 - Vida/Finanças"
SQLITE_DB = Path(__file__).resolve().parent.parent / "data/conversations.db"

_MES_NOME = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro",
    12: "Dezembro",
}

_DATA_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
_VALOR_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)")

# Colunas obrigatórias no cabeçalho de `Gastos.md` — nomes exatamente como
# aparecem hoje no arquivo (confirmado por leitura manual, Sessão 19).
_COLUNAS_OBRIGATORIAS = {"data", "descrição", "valor", "categoria", "registrou"}


def _parse_valor(raw: str) -> float | None:
    """'R$ 204,00' -> 204.0, 'R$ 50' -> 50.0. Sem dígito reconhecível -> None."""
    m = _VALOR_RE.search(raw)
    if not m:
        return None
    num = m.group(1)
    if "," in num:
        num = num.replace(".", "").replace(",", ".")
    try:
        return round(float(num), 2)
    except ValueError:
        return None


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator_row(cols: list[str]) -> bool:
    return all(set(c) <= set("-: ") for c in cols)


def ler_gastos_md() -> list[dict]:
    """Lê e parseia a tabela de `Gastos.md`. Não assume a ordem das colunas —
    localiza cada uma pelo nome do cabeçalho, como pedido."""
    if not GASTOS_MD.is_file():
        print(f"ERRO: {GASTOS_MD} não encontrado.")
        sys.exit(1)

    linhas = GASTOS_MD.read_text(encoding="utf-8").splitlines()

    header_idx = None
    colunas: list[str] = []
    for i, ln in enumerate(linhas):
        s = ln.strip()
        if s.startswith("|") and "data" in s.lower():
            colunas = _split_row(s)
            header_idx = i
            break
    if header_idx is None:
        print("ERRO: cabeçalho de tabela não encontrado em Gastos.md.")
        sys.exit(1)

    print(f"Cabeçalho encontrado em Gastos.md: {colunas}")
    idx = {c.strip().lower(): n for n, c in enumerate(colunas)}
    faltando = _COLUNAS_OBRIGATORIAS - set(idx)
    if faltando:
        print(f"ERRO: colunas esperadas não encontradas no cabeçalho: {faltando}")
        sys.exit(1)

    registros: list[dict] = []
    i = header_idx + 1
    while i < len(linhas):
        stripped = linhas[i].strip()
        if not stripped.startswith("|"):
            break
        cols = _split_row(stripped)
        if _is_separator_row(cols):
            i += 1
            continue
        if len(cols) < len(colunas):
            print(f"AVISO: linha mal formada ignorada: {stripped!r}")
            i += 1
            continue

        data_raw = cols[idx["data"]]
        m = _DATA_RE.match(data_raw)
        if not m:
            print(f"AVISO: data não reconhecida ({data_raw!r}), linha ignorada: {stripped!r}")
            i += 1
            continue
        dia, mes, ano = m.groups()

        valor = _parse_valor(cols[idx["valor"]])
        if valor is None:
            print(f"AVISO: valor não reconhecido, linha ignorada: {stripped!r}")
            i += 1
            continue

        registros.append({
            "data_iso": f"{ano}-{mes}-{dia}",   # ordenação interna
            "data_br": f"{dia}-{mes}-{ano[2:]}",  # dd-mm-aa, formato da tabela (Sessão 20)
            "mes": f"{ano}-{mes}",
            "descricao": cols[idx["descrição"]],
            "valor": valor,
            "categoria": cols[idx["categoria"]],
            "pessoa": cols[idx["registrou"]],
        })
        i += 1

    return registros


def _render_mes(mes: str, registros_mes: list[dict]) -> str:
    ano, mes_num = mes.split("-")
    registros_mes = sorted(registros_mes, key=lambda r: r["data_iso"])
    linhas_tabela = [
        f"| {r['data_br']} | {r['pessoa']} | {r['categoria']} | "
        f"{r['descricao']} | {r['valor']:.2f} |"
        for r in registros_mes
    ]
    mes_label = f"{_MES_NOME[int(mes_num)]} {ano}"
    return (
        "---\n"
        "tipo: gastos\n"
        f"mes: {mes}\n"
        f"atualizado: {datetime.date.today().isoformat()}\n"
        "---\n\n"
        f"# Gastos — {mes_label}\n\n"
        "| Data       | Pessoa  | Categoria   | Descrição           | Valor  |\n"
        "|------------|---------|-------------|---------------------|--------|\n"
        + "\n".join(linhas_tabela) + "\n"
    )


def gerar_arquivos_mensais(registros: list[dict], dry_run: bool) -> list[Path]:
    por_mes: dict[str, list[dict]] = {}
    for r in registros:
        por_mes.setdefault(r["mes"], []).append(r)

    # Duas passadas: primeiro monta tudo e confere que nenhum destino já
    # existe — só then escreve. Evita deixar meses parcialmente migrados se
    # abortar no meio (um `sys.exit` a meio do loop de escrita deixaria os
    # arquivos anteriores já gravados no disco).
    pendentes: list[tuple[Path, str]] = []
    for mes, regs in sorted(por_mes.items()):
        destino = GASTOS_DIR / f"Gastos - {mes}.md"
        if destino.exists():
            print(
                f"ABORTADO: {destino} já existe — não sobrescrevo. "
                "Nenhum arquivo foi escrito nesta rodada."
            )
            sys.exit(1)
        pendentes.append((destino, _render_mes(mes, regs)))

    for destino, conteudo in pendentes:
        print(f"\n--- {destino} ---")
        print(conteudo)
        if not dry_run:
            destino.write_text(conteudo, encoding="utf-8")

    return [destino for destino, _ in pendentes]


def _aproxima(reg_vault: dict, data: str, valor: float, descricao: str) -> bool:
    if reg_vault["data_iso"] != data:
        return False
    if abs(reg_vault["valor"] - valor) > 0.01:
        return False
    d1, d2 = reg_vault["descricao"].lower(), (descricao or "").lower()
    return d1 in d2 or d2 in d1 or d1 == d2


def conferencia_sql(registros_vault: list[dict]) -> None:
    """Só imprime — nunca escreve nada a partir do SQL."""
    print("\n=== Conferência SQL -> vault ===")
    if not SQLITE_DB.is_file():
        print(f"AVISO: {SQLITE_DB} não encontrado — pulando conferência.")
        return

    conn = sqlite3.connect(SQLITE_DB)
    try:
        rows = conn.execute(
            "SELECT data, valor, descricao, autor FROM expenses ORDER BY data"
        ).fetchall()
    finally:
        conn.close()

    sem_correspondencia = [
        (data, valor, descricao, autor)
        for data, valor, descricao, autor in rows
        if not any(_aproxima(r, data, valor, descricao) for r in registros_vault)
    ]

    if not sem_correspondencia:
        print(f"Todas as {len(rows)} linha(s) do SQL têm correspondência aproximada no vault.")
        return

    print(
        f"{len(sem_correspondencia)}/{len(rows)} linha(s) do SQL SEM correspondência "
        "no vault — decisão manual do Douglas, nada foi escrito automaticamente:"
    )
    for data, valor, descricao, autor in sem_correspondencia:
        print(f"  - {data} | R$ {valor:.2f} | {descricao!r} | autor={autor}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="não escreve nada — só mostra o que seria gerado",
    )
    args = parser.parse_args()

    print(f"Lendo {GASTOS_MD} ...")
    registros = ler_gastos_md()
    print(f"{len(registros)} linha(s) reconhecida(s) em Gastos.md")

    arquivos = gerar_arquivos_mensais(registros, args.dry_run)
    conferencia_sql(registros)

    print("\n=== Resumo ===")
    print(f"Meses gerados: {len(arquivos)}")
    print(f"Linhas migradas: {len(registros)}")
    print("Arquivos que seriam criados (--dry-run):" if args.dry_run else "Arquivos criados:")
    for a in arquivos:
        print(f"  - {a}")
    print(f"\n{GASTOS_MD} não foi tocado.")


if __name__ == "__main__":
    main()
