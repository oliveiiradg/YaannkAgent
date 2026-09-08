"""Banco de fixture do benchmark — Fase 8.

Aponta `db.DB_PATH` para um arquivo temporário e semeia a tabela `expenses` com
um conjunto conhecido de lançamentos, para que os casos de consulta financeira
("quanto gastamos esse mês?") tenham um total determinístico.

As datas são relativas a `date.today()` — assim casos podem dizer "esse mês" /
"mês passado" sem depender do dia em que a suíte roda.

Uso (no runner, ANTES de rodar qualquer caso):

    from benchmark.fixture import use_fixture_db, FIXTURE_CHAT
    use_fixture_db()
"""

import datetime
import tempfile
from pathlib import Path

FIXTURE_CHAT = "bench@g.us"
FIXTURE_DB = Path(tempfile.gettempdir()) / "yaannk_benchmark.db"

# Autor default de quem "envia" no benchmark (casos "quanto eu gastei" passam
# `expect.autor` explícito).
DEFAULT_AUTOR = "Alice"


def _first_of_month(d: datetime.date) -> datetime.date:
    return d.replace(day=1)


def _last_month(d: datetime.date) -> datetime.date:
    first = _first_of_month(d)
    return _first_of_month(first - datetime.timedelta(days=1))


def fixture_rows(today: datetime.date | None = None) -> list[tuple]:
    """(autor, valor, categoria, descricao, data_iso). ~15 lançamentos."""
    today = today or datetime.date.today()
    tm = _first_of_month(today)          # início deste mês
    lm = _last_month(today)              # início do mês passado

    def d(base: datetime.date, day: int) -> str:
        return base.replace(day=day).isoformat()

    return [
        # ---- este mês ----
        (DEFAULT_AUTOR, 150.00, "Mercado",     "compra no mercado",   d(tm, 3)),
        ("Bob",          80.00, "Mercado",     "feira",               d(tm, 6)),
        ("Bob",          33.50, "Transporte",  "uber pro trabalho",   d(tm, 8)),
        (DEFAULT_AUTOR,  22.00, "Transporte",  "uber de volta",       d(tm, 8)),
        (DEFAULT_AUTOR, 800.00, "Moradia",     "aluguel",             d(tm, 5)),
        ("Bob",          60.00, "Alimentação", "ifood",               d(tm, 10)),
        (DEFAULT_AUTOR,  45.90, "Lazer",       "cinema",              d(tm, 12)),
        ("Bob",          32.00, "Outros",      "api da kimi",         d(tm, 2)),
        # ---- mês passado ----
        (DEFAULT_AUTOR, 120.00, "Mercado",     "mercado mês passado", d(lm, 4)),
        ("Bob",         200.00, "Saúde",       "consulta",            d(lm, 15)),
        (DEFAULT_AUTOR, 800.00, "Moradia",     "aluguel mês passado", d(lm, 5)),
        ("Bob",          55.00, "Alimentação", "restaurante",         d(lm, 20)),
        (DEFAULT_AUTOR,  90.00, "Contas",      "internet",            d(lm, 10)),
    ]


def seed_expenses(conn, today: datetime.date | None = None) -> int:
    now_iso = datetime.datetime.now().isoformat(timespec="seconds")
    rows = fixture_rows(today)
    conn.executemany(
        "INSERT INTO expenses "
        "(chat_id, autor, valor, categoria, descricao, data, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(FIXTURE_CHAT, a, v, c, desc, dt, now_iso) for a, v, c, desc, dt in rows],
    )
    conn.commit()
    return len(rows)


def use_fixture_db(path: Path | None = None, today: datetime.date | None = None) -> Path:
    """Redireciona `db.DB_PATH` para um arquivo limpo e semeia `expenses`.
    Idempotente (recria do zero). Retorna o path."""
    from app.services import db

    p = Path(path) if path else FIXTURE_DB
    p.unlink(missing_ok=True)
    db.DB_PATH = p
    conn = db.get_connection()          # cria o schema (_SCHEMA)
    try:
        seed_expenses(conn, today)
    finally:
        conn.close()
    return p


# --- gastos no vault (Sessão 20) --------------------------------------------
# A partir da Fase B os gastos variáveis saíram do SQL e passaram a viver em
# `03 - Vida/Finanças/Gastos - YYYY-MM.md`. Semear só o SQL deixou os 17 casos
# `cas-*` financeiros lendo o vault REAL do casal — total não determinístico,
# `reply_equals` impossível. Estas funções escrevem as MESMAS linhas da fixture
# como notas de mês, para o benchmark voltar a ter dados próprios.

_MES_NOME_FIXTURE = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro",
    12: "Dezembro",
}


def gastos_md_por_mes(today: datetime.date | None = None) -> dict[str, str]:
    """`{"Gastos - YYYY-MM.md": conteúdo}` a partir de `fixture_rows()`.
    Mesmo formato que `vault_writer` escreve: colunas
    `Data | Pessoa | Categoria | Descrição | Valor`, data em `dd-mm-aa`."""
    por_mes: dict[str, list[tuple]] = {}
    for autor, valor, categoria, descricao, data_iso in fixture_rows(today):
        por_mes.setdefault(data_iso[:7], []).append(
            (data_iso, autor, valor, categoria, descricao)
        )

    notas: dict[str, str] = {}
    for mes, linhas in por_mes.items():
        ano_s, mes_s = mes.split("-")
        corpo = "\n".join(
            f"| {d[8:10]}-{d[5:7]}-{d[2:4]} | {autor} | {categoria} | "
            f"{descricao} | {valor:.2f} |"
            for d, autor, valor, categoria, descricao in sorted(linhas)
        )
        notas[f"Gastos - {mes}.md"] = (
            "---\n"
            "tipo: gastos\n"
            f"mes: {mes}\n"
            f"atualizado: {datetime.date.today().isoformat()}\n"
            "---\n\n"
            f"# Gastos — {_MES_NOME_FIXTURE[int(mes_s)]} {ano_s}\n\n"
            "| Data       | Pessoa  | Categoria   | Descrição           | Valor  |\n"
            "|------------|---------|-------------|---------------------|--------|\n"
            + corpo + "\n"
        )
    return notas


def use_fixture_vault(today: datetime.date | None = None) -> Path:
    """Escreve as notas de gasto da fixture num vault temporário e aponta
    `agent_vida.GASTOS_ROOT_OVERRIDE` para ele.

    Aponta SÓ a leitura de gastos. A primeira versão repontava
    `vault_search.settings.vault_path`, que é a raiz do TF-IDF: a perna
    TF-IDF do RRF passou a varrer um diretório com duas notas e o RAG
    técnico regrediu (tec-16, tec-17 e adv-08 perderam `Bugs e Erros.md`
    do top-3). O resto da busca continua vendo o vault real.
    """
    from app.agents import agent_vida

    raiz = Path(tempfile.gettempdir()) / "yaannk_benchmark_vault"
    financas = raiz / "03 - Vida" / "Finanças"
    financas.mkdir(parents=True, exist_ok=True)
    for nome, conteudo in gastos_md_por_mes(today).items():
        (financas / nome).write_text(conteudo, encoding="utf-8")

    agent_vida.GASTOS_ROOT_OVERRIDE = str(raiz)
    return raiz
