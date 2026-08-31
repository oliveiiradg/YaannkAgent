"""Consultas financeiras determinísticas sobre a tabela SQL `expenses` (Fase 7b).

O dual-write que popula essa tabela está em `vault_writer.record_expense`. Aqui
fica só o lado de leitura: `query_expenses` responde perguntas de **agregação
simples** ("quanto gastamos em setembro?", "total de mercado esse mês") direto
por SQL, sem passar pelo LLM+RAG.

Contrato com o webhook (Bloco 3): se `query_expenses` devolve uma string, essa é
a resposta final. Se devolve `None` (pergunta não é agregação simples, ou zero
lançamentos no SQL), o webhook segue para o fluxo LLM+RAG normal — que ainda
enxerga o histórico do markdown que não foi migrado para o SQL.
"""

import calendar
import datetime
import logging
import re

from app.config import settings
from app.services.db import get_connection
from app.services.vault_writer import _CATEGORIA_KEYWORDS

logger = logging.getLogger(__name__)

# Verbo/expressão de agregação — espelha o _FINANCIAL_QUERY_RE do llm/router.py.
# Sem isso no texto, não tratamos como agregação simples (→ None).
_AGGREGATION_RE = re.compile(
    r"\b(quanto|qual\s+o\s+total|total\s+(?:de|gasto|dos?|das?)|m[eé]dia|"
    r"somat[oó]rio|gastamos|gastei|gastaram|gastos?\s+(?:com|em|no|na|de|d[oa]s?)|"
    r"quanto\s+custou|quem\s+(?:gastou|gasta|pagou|paga)\s+mais)\b",
    re.IGNORECASE,
)

# Gasto individual ("quanto EU gastei") vs. do casal ("quanto gastamos").
_PESSOAL_RE = re.compile(r"\b(gastei|paguei|comprei|eu|meu|minha|sozinh[oa])\b", re.IGNORECASE)
_CASAL_RE = re.compile(r"\b(gastamos|pagamos|compramos|a\s+gente|n[óo]s)\b", re.IGNORECASE)
# Quebra por pessoa: "quanto cada um gastou?", "gastos por pessoa", "quem gastou mais?".
_POR_PESSOA_RE = re.compile(
    r"\b(cada\s+um|cada\s+pessoa|por\s+pessoa|por\s+cada\s+um|"
    r"quem\s+(?:gastou|gasta|paga|pagou)\s+mais|separad[oa]\s+por\s+pessoa|"
    r"divid(?:id[oa]|indo)\s+por\s+pessoa)\b",
    re.IGNORECASE,
)

_MESES = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}
_MES_NOME = {v: k for k, v in _MESES.items() if k != "marco"}


def _month_range(year: int, month: int) -> tuple[str, str]:
    start = datetime.date(year, month, 1)
    last = calendar.monthrange(year, month)[1]
    end = datetime.date(year, month, last) + datetime.timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _parse_periodo(text: str, today: datetime.date) -> tuple[str, str, str]:
    """Devolve (data_ini_iso, data_fim_exclusiva_iso, rótulo). Sem período
    citado no texto, assume o mês corrente."""
    low = text.lower()

    if re.search(r"\b(hoje)\b", low):
        return today.isoformat(), (today + datetime.timedelta(days=1)).isoformat(), "hoje"
    if re.search(r"\b(ontem)\b", low):
        y = today - datetime.timedelta(days=1)
        return y.isoformat(), today.isoformat(), "ontem"
    if re.search(r"\b(essa|esta|nesta|desta)\s+semana\b", low):
        seg = today - datetime.timedelta(days=today.weekday())
        return seg.isoformat(), (seg + datetime.timedelta(days=7)).isoformat(), "nesta semana"
    if re.search(r"\bsemana\s+passada\b", low):
        seg = today - datetime.timedelta(days=today.weekday() + 7)
        return seg.isoformat(), (seg + datetime.timedelta(days=7)).isoformat(), "na semana passada"
    if re.search(r"\bm[êe]s\s+(passad[oa]|anterior)\b", low):
        m, y = (12, today.year - 1) if today.month == 1 else (today.month - 1, today.year)
        ini, fim = _month_range(y, m)
        return ini, fim, f"em {_MES_NOME[m]}/{y}"
    if re.search(r"\bano\s+passado\b", low):
        y = today.year - 1
        return f"{y}-01-01", f"{y + 1}-01-01", f"em {y}"
    if re.search(r"\b(esse|este|neste|deste)\s+ano\b", low):
        return f"{today.year}-01-01", f"{today.year + 1}-01-01", f"em {today.year}"

    # mês por nome, com ano opcional ("setembro", "em março de 2024")
    for nome, m in _MESES.items():
        if re.search(rf"\b{nome}\b", low):
            ym = re.search(rf"{nome}\b(?:\s+(?:de\s+)?(\d{{4}}))?", low)
            y = int(ym.group(1)) if ym and ym.group(1) else today.year
            # "setembro" sem ano, mas mês ainda não chegou este ano → ano passado
            if not (ym and ym.group(1)) and m > today.month:
                y -= 1
            ini, fim = _month_range(y, m)
            return ini, fim, f"em {_MES_NOME[m]}/{y}"

    # default: mês corrente
    ini, fim = _month_range(today.year, today.month)
    return ini, fim, "neste mês"


def _match_categoria(text: str) -> str | None:
    low = text.lower()
    for cat, kws in _CATEGORIA_KEYWORDS.items():
        if cat.lower() in low or any(kw in low for kw in kws):
            return cat
    return None


def _brl(valor: float) -> str:
    inteiro, _, cents = f"{valor:.2f}".partition(".")
    neg = inteiro.startswith("-")
    inteiro = inteiro.lstrip("-")
    milhar = re.sub(r"(?<=\d)(?=(\d{3})+$)", ".", inteiro)
    return f"R$ {'-' if neg else ''}{milhar},{cents}"


def _people_in_scope(chat_id: str) -> list[str]:
    """Nomes de pessoas conhecidas nessa conversa: os de `KNOWN_NAMES` /
    `OWNER_NAME` / `PARTNER_NAME` mais os `autor` distintos já gravados na
    tabela para esse `chat_id` (auto-configura no benchmark e em produção)."""
    names = set(settings.known_names.values())
    for n in (settings.owner_name, settings.partner_name):
        if n:
            names.add(n)
    try:
        conn = get_connection()
        try:
            names |= {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT autor FROM expenses "
                    "WHERE chat_id = ? AND autor IS NOT NULL AND autor != ''",
                    (chat_id,),
                )
            }
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — sem nomes, cai no comportamento antigo
        pass
    return [n for n in names if n]


def _named_person(text: str, chat_id: str) -> str | None:
    low = text.lower()
    for name in _people_in_scope(chat_id):
        if re.search(rf"\b{re.escape(name.lower())}\b", low):
            return name
    return None


def query_expenses(chat_id: str, text: str, autor: str | None = None) -> str | None:
    """Responde uma pergunta de agregação simples sobre gastos via SQL.

    Devolve a resposta formatada (PT-BR, formato WhatsApp) ou `None` quando a
    pergunta não é agregação simples ou não há lançamentos no período — nos
    dois casos o webhook cai no fluxo LLM+RAG.

    Cobre: total do casal, filtro por categoria/período, "quanto EU gastei"
    (usa `autor`), "quanto o Fulano gastou" (nome citado no texto) e
    "quanto cada um gastou" (quebra por pessoa).
    """
    if not _AGGREGATION_RE.search(text):
        return None

    today = datetime.date.today()
    data_ini, data_fim, rotulo = _parse_periodo(text, today)
    categoria = _match_categoria(text)
    cab = rotulo[0].upper() + rotulo[1:]

    where = ["chat_id = ?", "data >= ?", "data < ?"]
    params: list = [chat_id, data_ini, data_fim]
    if categoria:
        where.append("categoria = ?")
        params.append(categoria)

    # --- "quanto cada um gastou?" → quebra por pessoa ---------------------
    if _POR_PESSOA_RE.search(text):
        try:
            conn = get_connection()
            try:
                rows = conn.execute(
                    f"SELECT autor, SUM(valor), COUNT(*) FROM expenses "
                    f"WHERE {' AND '.join(where)} "
                    "GROUP BY autor ORDER BY SUM(valor) DESC",
                    params,
                ).fetchall()
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_expenses (por pessoa): %r — fallback p/ LLM", exc)
            return None
        if not rows:
            return None
        alvo = f" em {categoria}" if categoria else ""
        total = sum(v or 0 for _, v, _ in rows)
        linhas = [f"{cab}{alvo}, por pessoa:"]
        linhas += [
            f"- {a or 'sem nome'}: {_brl(v)} ({c} lanç.)" for a, v, c in rows
        ]
        linhas.append(f"Total: {_brl(total)}")
        return "\n".join(linhas)

    # --- filtro por pessoa: nome citado > "eu gastei" > casal ------------
    named = _named_person(text, chat_id)
    pessoal = bool(autor and _PESSOAL_RE.search(text) and not _CASAL_RE.search(text))
    if named:
        where.append("autor = ?")
        params.append(named)
        sujeito = f"{named} gastou"
    elif pessoal:
        where.append("autor = ?")
        params.append(autor)
        sujeito = "você gastou"
    else:
        sujeito = "vocês gastaram"
    where_sql = " AND ".join(where)

    try:
        conn = get_connection()
        try:
            total, n = conn.execute(
                f"SELECT COALESCE(SUM(valor), 0), COUNT(*) FROM expenses WHERE {where_sql}",
                params,
            ).fetchone()
            breakdown = conn.execute(
                f"SELECT categoria, SUM(valor) FROM expenses WHERE {where_sql} "
                "GROUP BY categoria ORDER BY SUM(valor) DESC",
                params,
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — qualquer falha de SQL → fallback
        logger.warning("query_expenses: consulta falhou (%r) — fallback p/ LLM", exc)
        return None

    if not n:
        logger.info(
            "query_expenses: 0 lançamentos (chat=%s, %s, cat=%s, pessoa=%s) — fallback p/ LLM",
            chat_id, rotulo, categoria, named or ("eu" if pessoal else None),
        )
        return None

    lanc = "lançamento" if n == 1 else "lançamentos"
    if categoria:
        return f"{cab}, {sujeito} {_brl(total)} em {categoria} ({n} {lanc})."

    linha = f"{cab}, {sujeito} {_brl(total)} ({n} {lanc})."
    if len(breakdown) > 1:
        itens = "\n".join(f"- {cat}: {_brl(v)}" for cat, v in breakdown)
        return f"{linha}\n{itens}"
    return linha


def balance_report(chat_id: str, today: datetime.date | None = None) -> str:
    """Balanço do mês corrente: total geral + quebra por categoria.

    Responde ao comando explícito `@yaannk balanço`. Diferente de
    `query_expenses`, **sempre** devolve uma string — inclusive quando não há
    lançamento no mês. Comando explícito merece resposta explícita; cair no
    LLM+RAG aqui só produziria ruído.
    """
    today = today or datetime.date.today()
    data_ini, data_fim = _month_range(today.year, today.month)
    titulo = f"{_MES_NOME[today.month]}/{today.year}"

    try:
        conn = get_connection()
        try:
            total, n = conn.execute(
                "SELECT COALESCE(SUM(valor), 0), COUNT(*) FROM expenses "
                "WHERE chat_id = ? AND data >= ? AND data < ?",
                (chat_id, data_ini, data_fim),
            ).fetchone()
            breakdown = conn.execute(
                "SELECT categoria, SUM(valor) FROM expenses "
                "WHERE chat_id = ? AND data >= ? AND data < ? "
                "GROUP BY categoria ORDER BY SUM(valor) DESC",
                (chat_id, data_ini, data_fim),
            ).fetchall()
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — comando não pode explodir no grupo
        logger.warning("balance_report: consulta falhou (%r)", exc)
        return "Não consegui montar o balanço agora. Tenta de novo daqui a pouco."

    if not n:
        return f"📊 Balanço de {titulo}\nNenhum gasto registrado ainda."

    lanc = "lançamento" if n == 1 else "lançamentos"
    linhas = [
        f"📊 Balanço de {titulo}",
        f"Total: {_brl(total)} ({n} {lanc})",
    ]
    linhas += [f"- {cat}: {_brl(v)}" for cat, v in breakdown]
    return "\n".join(linhas)
