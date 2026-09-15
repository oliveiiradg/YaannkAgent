"""Consulta determinística sobre as contas fixas do mês — Grupo do casal.

Fonte: a nota mensal `Contas - AAAA-MM.md` em `02 - Áreas/Finanças/`.
Diferente de `expenses.py` (gastos variáveis, com dual-write em SQL via
mensagens do WhatsApp), a nota de contas fixas não tem write-path pelo
Yaannk fora do que este módulo escreve (`mark_bill_paid`/`remove_bill`/
`add_bill`/`create_next_month_note`) — o resto é editado direto no Obsidian
pelo casal. Por isso não há segunda fonte de verdade em SQL aqui: a nota é
lida e parseada sob demanda, do mesmo jeito que `vault_search.py` lê o vault
direto do disco (sem MCP).

Formato da nota (Sessão 23, D-10 — substituiu o formato anterior de duas
seções `## PAGAS`/`## PENDENTES` com subtabelas por categoria `### `):
uma lista só de checkboxes, sem seções nem subtabelas —

    - [ ] Aluguel — R$ 1.650,00 — dia 5
    - [x] Condomínio — R$ 350,00 — dia 5

`[x]` = paga, `[ ]` = pendente. O formato anterior (seções + subtabelas)
quebrava toda vez que `mark_bill_paid` precisava mover uma linha entre
seções — over-engineering desnecessário pra um dado que é só "nome, valor,
dia, pago ou não". Notas anteriores a set/2026 (formato de tabela única com
coluna de Status) e o formato de seções (set/2026, Sessão 20-22) NÃO são mais
suportados por `parse_contas_fixas` — a virada pra checkbox reconstrói a
nota do mês corrente; meses antigos não têm write-path de qualquer forma.
"""

import datetime
import logging
import re
from dataclasses import dataclass

from app.services.expenses import _brl
from app.services.finance_patterns import BILLS_QUERY_RE
from app.services.vault_search import safe_vault_path
from app.services.vault_writer import VaultWriteError, _mcp_call, _valor_float

logger = logging.getLogger(__name__)

_CONTAS_DIR = "02 - Áreas/Finanças"

_MES_NOME = {
    1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio",
    6: "junho", 7: "julho", 8: "agosto", 9: "setembro", 10: "outubro",
    11: "novembro", 12: "dezembro",
}

_STATUS_ROTULO = {
    "pago": "já foi paga ✅",
    "pendente": "ainda está pendente ⏳",
}

# "- [ ] Nome — R$ 530,14 — dia 15" / "- [x] Nome — R$ 57,00" (dia opcional —
# nem toda conta tem vencimento fixo, ex. parcela já quitada). Separador
# aceita travessão (`—`, o que `add_bill`/`create_next_month_note` sempre
# escrevem) ou hífen simples (tolerância a edição manual no Obsidian).
#
# Sem `$` no fim de propósito: o plugin Tasks do Obsidian anexa
# `✅ AAAA-MM-DD` na linha quando alguém marca o checkbox pela UI (achado
# real na nota de 2026-09, Sessão 23) — com âncora de fim de linha essa
# sujeira derrubava a linha inteira do parser (10 de 22 contas sumiam).
# `.match()` já ignora qualquer coisa depois do que os grupos capturam.
_CHECKBOX_RE = re.compile(
    r"^-\s*\[(?P<check>[ xX])\]\s*(?P<nome>.+?)\s*(?:—|-)\s*R\$\s*(?P<valor>[\d.,]+)"
    r"(?:\s*(?:—|-)\s*dia\s*0?(?P<dia>\d{1,2}))?",
    re.IGNORECASE,
)


def _parse_checkbox_linha(linha: str) -> dict | None:
    """`(descricao, valor, dia_vencimento, status)` de uma linha de checkbox,
    ou `None` se a linha não bater o formato (linha de texto solto, título,
    etc. — ignorada silenciosamente, sem levantar exceção)."""
    m = _CHECKBOX_RE.match(linha.strip())
    if not m:
        return None
    valor = _valor_float(m.group("valor"))
    if valor is None:
        logger.warning("bills: linha de conta sem valor numérico ignorada: %r", linha)
        return None
    dia = m.group("dia")
    return {
        "descricao": m.group("nome").strip(),
        "valor": valor,
        "dia_vencimento": int(dia) if dia else None,
        "status": "pago" if m.group("check").lower() == "x" else "pendente",
    }


def parse_contas_fixas(content: str) -> list[dict]:
    """Extrai as contas fixas de uma nota `Contas - AAAA-MM.md` no formato
    de checkbox (ver docstring do módulo).

    Cada item: `{"descricao": str, "valor": float, "dia_vencimento": int|None,
    "status": "pago"|"pendente"}`. Devolve `[]` se não achar nenhuma linha de
    checkbox reconhecível — nunca levanta exceção por formato inesperado.
    """
    rows: list[dict] = []
    for linha in content.splitlines():
        conta = _parse_checkbox_linha(linha)
        if conta is not None:
            rows.append(conta)
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
    internet" só casa se alguma linha se chamar literalmente algo com
    "internet" (ver observação no relatório: hoje a nota usa "Wifi").
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
    # de pendências.
    if re.search(r"falta\s+pagar|pendente\w*|\bvenc\w*", low):
        return _result(kind="pending")

    # "quanto tenho de conta fixa" — total geral do mês (pagas + pendentes).
    return _result(kind="total")


# --- Módulo proativo (Fase E) -------------------------------------------------


def _dias_label(dias: int) -> str:
    if dias == 0:
        return "vence hoje"
    if dias == 1:
        return "vence amanhã"
    return f"vence em {dias} dias"


def get_bills_due(
    days_ahead: int = 3, *, today: datetime.date | None = None
) -> list[dict]:
    """Contas fixas não pagas com vencimento entre `today` e `today +
    days_ahead` dias (inclusive dos dois extremos).

    Devolve `[{"name": str, "amount": float, "due_date": "AAAA-MM-DD",
    "days_until_due": int}]`, ordenado por data de vencimento. Carrega a nota
    de cada mês que o intervalo tocar (cobre a virada de mês perto do dia 1).
    Só considera contas com `dia_vencimento` conhecido e `status != "pago"`.
    Nunca levanta exceção — mês sem nota (`get_contas_do_mes` devolve `None`)
    é tratado como sem contas nesse mês.
    """
    today = today or datetime.date.today()
    end = today + datetime.timedelta(days=days_ahead)

    meses: set[tuple[int, int]] = set()
    d = today
    while d <= end:
        meses.add((d.year, d.month))
        d += datetime.timedelta(days=1)

    encontradas: list[dict] = []
    for ano, mes in sorted(meses):
        contas = get_contas_do_mes(ano, mes) or []
        for c in contas:
            if c["status"] == "pago" or c["dia_vencimento"] is None:
                continue
            try:
                vencimento = datetime.date(ano, mes, c["dia_vencimento"])
            except ValueError:
                # dia de vencimento inválido pro mês (ex.: 31 em mês de 30
                # dias) — ignora essa conta em vez de levantar exceção.
                continue
            if today <= vencimento <= end:
                encontradas.append({
                    "name": c["descricao"],
                    "amount": c["valor"],
                    "due_date": vencimento.isoformat(),
                    "days_until_due": (vencimento - today).days,
                })

    encontradas.sort(key=lambda b: b["due_date"])
    return encontradas


def _parse_month(month: str) -> tuple[int, int] | None:
    """`"AAAA-MM"` -> `(ano, mes)`, ou `None` se inválido. Usado por todas as
    operações de escrita (`mark_bill_paid`, `remove_bill`, `add_bill`)."""
    try:
        ano_s, mes_s = month.split("-", 1)
        ano, mes = int(ano_s), int(mes_s)
        if not (1 <= mes <= 12):
            raise ValueError
        return ano, mes
    except (ValueError, AttributeError):
        return None


def _acha_linha_conta(
    lines: list[str], bill_name: str
) -> list[tuple[int, dict]]:
    """Todas as linhas de checkbox cuja descrição bate `bill_name` (substring
    case-insensitive, tolerando complemento entre parênteses — mesma
    heurística de `_match_conta`), na ordem em que aparecem na nota.
    Devolve `[(índice, conta_parseada), ...]`."""
    low_name = bill_name.strip().lower()
    achadas: list[tuple[int, dict]] = []
    for idx, linha in enumerate(lines):
        conta = _parse_checkbox_linha(linha)
        if conta is None:
            continue
        nome = conta["descricao"].lower()
        base = re.sub(r"\(.*?\)", "", nome).strip()
        if low_name in nome or (base and low_name in base):
            achadas.append((idx, conta))
    return achadas


async def mark_bill_paid(bill_name: str, month: str) -> dict:
    """Marca a conta `bill_name` como paga (`- [ ]` → `- [x]`) na nota
    `Contas - AAAA-MM.md` de `month` (formato `"AAAA-MM"`) e regrava via MCP
    do Obsidian (mesmo mecanismo de escrita de `vault_writer`).

    Simples troca de caractere na linha — sem mover nada entre seções (a
    nota é uma lista só de checkboxes, D-10). Devolve `{"success": bool,
    "message": str}` — nunca levanta exceção. `bill_name` é casado por
    substring, case-insensitive, contra a descrição da conta (mesma
    heurística de `_match_conta`, incluindo tolerância a complemento entre
    parênteses). Se houver mais de uma pendente com nome batendo, marca a
    primeira encontrada na nota.
    """
    parsed = _parse_month(month)
    if parsed is None:
        return {"success": False, "message": f"mês inválido: {month!r} (use AAAA-MM)"}
    ano, mes = parsed

    relpath = _contas_relpath(ano, mes)
    path = safe_vault_path(relpath)
    if path is None or not path.is_file():
        return {"success": False, "message": f"nota de {month} não encontrada no vault"}
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"success": False, "message": f"falha ao ler a nota: {exc}"}

    lines = content.splitlines()
    achadas = _acha_linha_conta(lines, bill_name)
    if not achadas:
        return {
            "success": False,
            "message": f"conta {bill_name!r} não encontrada entre as pendentes",
        }

    pendente = next((item for item in achadas if item[1]["status"] == "pendente"), None)
    if pendente is None:
        return {"success": False, "message": f"{bill_name!r} já está marcada como paga"}

    idx, conta = pendente
    lines[idx] = re.sub(r"^(\s*-\s*\[)[ ]", r"\1x", lines[idx], count=1)
    novo_content = "\n".join(lines) + ("\n" if content.endswith("\n") else "")

    try:
        await _mcp_call("create_vault_file", {"path": relpath, "content": novo_content})
    except VaultWriteError as exc:
        return {"success": False, "message": f"falha ao gravar no vault: {exc}"}

    logger.info("bills: %r marcada como paga em %s", conta["descricao"], relpath)
    return {"success": True, "message": f"{conta['descricao']} marcada como paga"}


async def remove_bill(bill_name: str, month: str) -> dict:
    """Remove a linha da conta `bill_name` (paga ou pendente) da nota
    `Contas - AAAA-MM.md` de `month`, regrava via MCP. Devolve
    `{"success","message"}`, nunca levanta exceção."""
    parsed = _parse_month(month)
    if parsed is None:
        return {"success": False, "message": f"mês inválido: {month!r} (use AAAA-MM)"}
    ano, mes = parsed

    relpath = _contas_relpath(ano, mes)
    path = safe_vault_path(relpath)
    if path is None or not path.is_file():
        return {"success": False, "message": f"nota de {month} não encontrada no vault"}
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"success": False, "message": f"falha ao ler a nota: {exc}"}

    lines = content.splitlines()
    achadas = _acha_linha_conta(lines, bill_name)
    if not achadas:
        return {"success": False, "message": f"conta {bill_name!r} não encontrada na nota"}
    idx, conta = achadas[0]

    novas_linhas = lines[:idx] + lines[idx + 1:]
    novo_content = "\n".join(novas_linhas) + ("\n" if content.endswith("\n") else "")

    try:
        await _mcp_call("create_vault_file", {"path": relpath, "content": novo_content})
    except VaultWriteError as exc:
        return {"success": False, "message": f"falha ao gravar no vault: {exc}"}

    logger.info("bills: %r removida de %s", conta["descricao"], relpath)
    return {"success": True, "message": f"{conta['descricao']} removida"}


def _formata_linha_conta(nome: str, valor: float, dia_vencimento: int | None) -> str:
    if dia_vencimento is None:
        return f"- [ ] {nome} — {_brl(valor)}"
    return f"- [ ] {nome} — {_brl(valor)} — dia {dia_vencimento}"


def _append_conta(content: str, linha_nova: str) -> str:
    """Insere `linha_nova` logo após a última linha de checkbox reconhecida
    (mantém a lista contígua mesmo se a nota tiver texto depois, ex. uma
    seção de observações); se a nota ainda não tiver nenhum checkbox, anexa
    no fim do arquivo."""
    lines = content.splitlines()
    last_idx: int | None = None
    for i, linha in enumerate(lines):
        if _parse_checkbox_linha(linha) is not None:
            last_idx = i
    if last_idx is None:
        novas = lines + [linha_nova]
    else:
        novas = lines[: last_idx + 1] + [linha_nova] + lines[last_idx + 1:]
    texto = "\n".join(novas)
    return texto + "\n" if content.endswith("\n") else texto


async def add_bill(
    bill_name: str, valor: float, dia_vencimento: int, month: str,
    categoria: str | None = None,
) -> dict:
    """Insere uma conta nova (pendente) na nota `Contas - AAAA-MM.md` de
    `month`, logo após a última conta existente. `categoria` não é mais
    usado (D-10 tirou as subseções por categoria) — mantido no parâmetro só
    pra não quebrar chamadores existentes. Regrava via MCP. Devolve
    `{"success","message"}`, nunca levanta exceção.
    """
    parsed = _parse_month(month)
    if parsed is None:
        return {"success": False, "message": f"mês inválido: {month!r} (use AAAA-MM)"}
    ano, mes = parsed

    relpath = _contas_relpath(ano, mes)
    path = safe_vault_path(relpath)
    if path is None or not path.is_file():
        return {"success": False, "message": f"nota de {month} não encontrada no vault"}
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"success": False, "message": f"falha ao ler a nota: {exc}"}

    linha_nova = _formata_linha_conta(bill_name, valor, dia_vencimento)
    novo_content = _append_conta(content, linha_nova)

    try:
        await _mcp_call("create_vault_file", {"path": relpath, "content": novo_content})
    except VaultWriteError as exc:
        return {"success": False, "message": f"falha ao gravar no vault: {exc}"}

    logger.info("bills: %r adicionada em %s", bill_name, relpath)
    return {"success": True, "message": f"{bill_name} adicionada"}


def _proximo_mes(ano: int, mes: int) -> tuple[int, int]:
    return (ano + 1, 1) if mes == 12 else (ano, mes + 1)


async def create_next_month_note(ano: int, mes: int, *, force: bool = False) -> dict:
    """Cria `Contas - AAAA-MM.md` do mês seguinte a `(ano, mes)`: todas as
    contas da nota atual (pagas + pendentes) viram pendentes (`- [ ]`) no mês
    novo, mesmo nome/valor/dia. Parcelas terminadas (ex.: "CEA 6/6") são
    copiadas sem filtro — ajuste manual depois via `remove_bill`.

    `force=False` (padrão) recusa sobrescrever uma nota de destino já
    existente. Devolve `{"success","month","note_path","message"}` (mais
    `"n_contas"`/`"total_pendente"` em caso de sucesso), nunca levanta
    exceção.
    """
    relpath_atual = _contas_relpath(ano, mes)
    path_atual = safe_vault_path(relpath_atual)
    if path_atual is None or not path_atual.is_file():
        return {"success": False, "message": f"nota de {ano:04d}-{mes:02d} não encontrada"}
    try:
        content_atual = path_atual.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"success": False, "message": f"falha ao ler a nota atual: {exc}"}

    ano_prox, mes_prox = _proximo_mes(ano, mes)
    relpath_prox = _contas_relpath(ano_prox, mes_prox)
    path_prox = safe_vault_path(relpath_prox)
    if not force and path_prox is not None and path_prox.is_file():
        return {
            "success": False,
            "message": (
                f"nota de {ano_prox:04d}-{mes_prox:02d} já existe "
                "(use force=true pra sobrescrever)"
            ),
        }

    contas_atuais = parse_contas_fixas(content_atual)
    mes_label = f"{_MES_NOME[mes_prox].title()}/{ano_prox}"

    linhas: list[str] = [
        "---",
        "tipo: financas-mensal",
        f"mes: {ano_prox:04d}-{mes_prox:02d}",
        f"tags: [finanças, mensal, {ano_prox}]",
        "---",
        "",
        f"# 📋 Contas — {mes_label}",
        "",
    ]
    linhas += [
        _formata_linha_conta(c["descricao"], c["valor"], c["dia_vencimento"])
        for c in contas_atuais
    ]
    linhas.append("")
    novo_content = "\n".join(linhas)

    try:
        await _mcp_call("create_vault_file", {"path": relpath_prox, "content": novo_content})
    except VaultWriteError as exc:
        return {"success": False, "message": f"falha ao gravar a nota nova: {exc}"}

    n_contas = len(contas_atuais)
    total = sum(c["valor"] for c in contas_atuais)

    logger.info(
        "bills: nota de %04d-%02d criada a partir de %04d-%02d (%d contas, %s)",
        ano_prox, mes_prox, ano, mes, n_contas, _brl(total),
    )
    return {
        "success": True,
        "month": f"{ano_prox:04d}-{mes_prox:02d}",
        "note_path": relpath_prox,
        "n_contas": n_contas,
        "total_pendente": total,
        "message": (
            f"📅 Nota de {mes_label} criada com {n_contas} conta(s) "
            f"pendente(s), total {_brl(total)}. Dá uma olhada e ajusta o que "
            'mudou (ex.: "remove CEA", "adiciona academia R$ 80 dia 10").'
        ),
    }


def format_bills_due_message(bills: list[dict]) -> str:
    """Formata a lista de `get_bills_due()` como mensagem pronta pro WhatsApp."""
    if not bills:
        return "✅ Nenhuma conta a vencer no período."
    linhas = [
        f"{b['name']} — {_brl(b['amount'])} ({_dias_label(b['days_until_due'])})"
        for b in bills
    ]
    total = sum(b["amount"] for b in bills)
    return "⚠️ Lembrete de contas\n\n" + "\n".join(linhas) + f"\n\nTotal: {_brl(total)}"
