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
