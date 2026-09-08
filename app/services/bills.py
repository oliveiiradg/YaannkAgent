"""Consulta determinística sobre as contas fixas do mês — Grupo do casal.

Fonte: a nota mensal `Contas - AAAA-MM.md` em
`02 - Áreas/Finanças/Controle de Gastos/`. Diferente de `expenses.py` (gastos
variáveis, com dual-write em SQL via mensagens do WhatsApp), a nota de contas
fixas não tem write-path pelo Yaannk — é editada direto no Obsidian pelo casal,
uma vez por mês. Por isso não há segunda fonte de verdade em SQL aqui: a nota
é lida e parseada sob demanda, do mesmo jeito que `vault_search.py` lê o vault
direto do disco (sem MCP).

As três notas existentes (jun/jul/ago-2026) usam três formatos ligeiramente
diferentes de tabela (com/sem heading `## Contas Fixas`, vencimento em
"DD/MM" ou "dia N", status em "Pago"/"✅ Pago"/"✅" bare). O parser localiza a
tabela pelo cabeçalho de colunas (Descrição/Valor/Vencimento/Status), não por
uma seção fixa, e tolera essas três variações — mas não é garantido que
cubra formatos futuros ainda não vistos.
"""

import datetime
import logging
import re
from dataclasses import dataclass

from app.services.expenses import _brl
from app.services.finance_patterns import BILLS_QUERY_RE
from app.services.vault_search import safe_vault_path
from app.services.vault_writer import _valor_float

logger = logging.getLogger(__name__)

_CONTAS_DIR = "03 - Vida/Finanças"

_MES_NOME = {
    1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio",
    6: "junho", 7: "julho", 8: "agosto", 9: "setembro", 10: "outubro",
    11: "novembro", 12: "dezembro",
}

# Cabeçalho da tabela "Contas Fixas": exige Vencimento + Status (nessa ordem,
# depois de Descrição/Valor) pra não casar com a tabela de "Gastos Variáveis"
# (Data/Descrição/Valor/Categoria), que também tem Descrição e Valor.
_HEADER_RE = re.compile(
    r"^\|.*(?:descri[cç][ãa]o|conta).*\|.*valor.*\|.*vencimento.*\|.*status.*\|",
    re.IGNORECASE,
)
_SEP_RE = re.compile(r"^\|[\s:-]+\|")

_STATUS_CONFIRMAR_RE = re.compile(r"⚠️|confirmar", re.IGNORECASE)
_STATUS_PAGO_RE = re.compile(r"✅|\bpago\b", re.IGNORECASE)
_STATUS_PENDENTE_RE = re.compile(r"⏳")

_DIA_MES_RE = re.compile(r"\bdia\s*0?(\d{1,2})\b", re.IGNORECASE)
_DATA_BR_RE = re.compile(r"\b0?(\d{1,2})/0?(\d{1,2})\b")

_STATUS_ROTULO = {
    "pago": "já foi paga ✅",
    "pendente": "ainda está pendente ⏳",
    "confirmar": "está marcada para confirmar ⚠️",
    "desconhecido": "sem status claro no registro",
}


def _split_row(line: str) -> list[str] | None:
    line = line.strip()
    if not line.startswith("|"):
        return None
    cols = [c.strip() for c in line.strip("|").split("|")]
    return cols or None


def _parse_status(raw: str) -> str:
    if not raw.strip():
        return "desconhecido"
    if _STATUS_CONFIRMAR_RE.search(raw):
        return "confirmar"
    if _STATUS_PAGO_RE.search(raw):
        return "pago"
    if _STATUS_PENDENTE_RE.search(raw):
        return "pendente"
    return "desconhecido"


def _parse_dia_vencimento(raw: str) -> int | None:
    m = _DIA_MES_RE.search(raw)
    if m:
        return int(m.group(1))
    m = _DATA_BR_RE.search(raw)
    if m:
        return int(m.group(1))
    return None


def parse_contas_fixas(content: str) -> list[dict]:
    """Extrai as linhas da tabela "Contas Fixas" de uma nota `Contas - AAAA-MM.md`.

    Cada item: `{"descricao": str, "valor": float, "dia_vencimento": int|None,
    "status": "pago"|"pendente"|"confirmar"|"desconhecido"}`.

    Localiza a tabela pelo cabeçalho, não por uma seção fixa (ver docstring do
    módulo). Devolve `[]` se não achar a tabela — nunca levanta exceção por
    formato inesperado. Linha sem valor monetário reconhecível é ignorada
    (log de aviso), o resto da tabela continua sendo parseado.
    """
    lines = content.splitlines()
    rows: list[dict] = []
    i = 0
    n = len(lines)
    while i < n:
        if not _HEADER_RE.match(lines[i].strip()):
            i += 1
            continue
        i += 1  # pula o cabeçalho, entra na tabela
        while i < n:
            stripped = lines[i].strip()
            if not stripped.startswith("|"):
                break
            if _SEP_RE.match(stripped):
                i += 1
                continue
            cols = _split_row(stripped)
            if not cols or len(cols) < 4:
                logger.warning("bills: linha de tabela mal formada ignorada: %r", stripped)
                i += 1
                continue
            descricao = cols[0]
            valor_raw = cols[1]
            vencimento_raw = cols[2]
            status_raw = cols[4] if len(cols) > 4 else cols[3]
            if not descricao:
                i += 1
                continue
            valor = _valor_float(valor_raw)
            if valor is None:
                logger.warning("bills: linha sem valor numérico ignorada: %r", stripped)
                i += 1
                continue
            rows.append({
                "descricao": descricao,
                "valor": valor,
                "dia_vencimento": _parse_dia_vencimento(vencimento_raw),
                "status": _parse_status(status_raw),
            })
            i += 1
        # `i` já passou do fim da tabela — o loop externo segue procurando o
        # próximo cabeçalho a partir daqui, sem parar no primeiro bloco.
    return rows


def _contas_relpath(ano: int, mes: int) -> str:
    return f"{_CONTAS_DIR}/Contas - {ano:04d}-{mes:02d}.md"


def get_contas_do_mes(ano: int, mes: int) -> list[dict] | None:
    """Lê e parseia `Contas - AAAA-MM.md` direto do disco (mesmo padrão de
    leitura de `vault_search.py` — sem MCP, sem SQL).

    Devolve `None` se a nota do mês ainda não existe (o casal cria uma nota
    nova a cada mês; nada garante que a do mês corrente já foi criada) — quem
    chama cai no fluxo LLM+RAG normal, igual ao contrato de `query_expenses`.
    """
    path = safe_vault_path(_contas_relpath(ano, mes))
    if path is None or not path.is_file():
        logger.info("bills: nota de %04d-%02d não encontrada — fallback p/ RAG", ano, mes)
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("bills: falha ao ler nota de %04d-%02d (%r)", ano, mes, exc)
        return None
    return parse_contas_fixas(content)


def _match_conta(contas: list[dict], low_text: str) -> dict | None:
    """Acha a conta cujo nome é citado na pergunta (substring, case-insensitive).

    Também tenta o nome sem o complemento entre parênteses (`"Rodrigo (Moto)"`
    → `"rodrigo"`). NÃO faz mapeamento de sinônimo/categoria — "conta de
    internet" só casa se alguma linha da tabela se chamar literalmente algo
    com "internet" (ver observação no relatório: hoje a tabela usa "Wifi").
    """
    for conta in contas:
        nome = conta["descricao"].lower()
        base = re.sub(r"\(.*?\)", "", nome).strip()
        for candidate in (nome, base):
            if candidate and candidate in low_text:
                return conta
    return None


@dataclass(frozen=True)
class BillsResult:
    """Resultado estruturado de uma consulta de contas fixas (Fase B, D-09).

    `answer_bills_query()` não formata mais a resposta final — devolve isso, e
    quem chama (Agent Vida) decide entre o template Python de sempre (rápido,
    sem Kimi, reconstrói exatamente as strings de antes) ou passar este dict
    pro Kimi formatar quando a pergunta pede mais do que o template cobre
    (ex.: "só as pagas", que não tem branch determinístico).
    """

    kind: str                       # "specific" | "due_today" | "pending" | "total"
    mes_label: str                  # ex.: "setembro/2026"
    contas: list[dict]              # todas as contas do mês
    pagas: list[dict]               # subconjunto de `contas` com status == "pago"
    pendentes: list[dict]           # subconjunto com status != "pago"
    total_geral: float
    total_pendente: float
    hoje: int | None = None         # dia do mês da consulta — só em kind="due_today"
    conta_alvo: dict | None = None  # conta citada por nome — só em kind="specific"


def answer_bills_query(
    text: str, *, today: datetime.date | None = None
) -> BillsResult | None:
    """Monta o resultado estruturado de uma pergunta sobre contas fixas, sem LLM.

    Devolve `None` quando o texto não bate o vocabulário de contas fixas
    (`BILLS_QUERY_RE`), a nota do mês não existe, a nota existe mas não tem
    nenhuma linha reconhecível, ou a pergunta cita uma conta específica que
    não foi encontrada por nome — em todos os casos quem chama cai no fluxo
    LLM+RAG normal. Quando devolve um `BillsResult`, quem chama decide como
    formatar (ver docstring de `BillsResult`).
    """
    if not BILLS_QUERY_RE.search(text):
        return None

    today = today or datetime.date.today()
    contas = get_contas_do_mes(today.year, today.month)
    if not contas:
        return None

    low = text.lower()
    pagas = [c for c in contas if c["status"] == "pago"]
    pendentes = [c for c in contas if c["status"] != "pago"]
    total_geral = sum(c["valor"] for c in contas)
    total_pendente = sum(c["valor"] for c in pendentes)
    mes_label = f"{_MES_NOME[today.month]}/{today.year}"

    def _result(**kw) -> BillsResult:
        return BillsResult(
            mes_label=mes_label, contas=contas, pagas=pagas, pendentes=pendentes,
            total_geral=total_geral, total_pendente=total_pendente, **kw,
        )

    # Pergunta sobre 1 conta específica: "a conta de internet já foi paga?"
    if re.search(r"\bj[áa]\s+foi\s+pag", low):
        found = _match_conta(contas, low)
        if found is None:
            return None
        return _result(kind="specific", conta_alvo=found)

    # "quais contas vencem hoje" — filtro por dia de vencimento == hoje.
    if re.search(r"venc\w*\s+hoje\b|\bhoje\b.*\bvenc\w*", low):
        return _result(kind="due_today", hoje=today.day)

    # "o que falta pagar", "contas pendentes", "contas a vencer" — lista geral
    # de pendências (inclui "confirmar" e "desconhecido": nenhum dos dois é
    # "pago" confirmado).
    if re.search(r"falta\s+pagar|pendente\w*|\bvenc\w*", low):
        return _result(kind="pending")

    # "quanto tenho de conta fixa" — total geral do mês (pagas + pendentes).
    return _result(kind="total")
