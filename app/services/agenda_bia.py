"""Agenda de clientes da Bia (V5) — chat privado dela (`BIA_JID`).

Fonte: nota mensal `Agenda - AAAA-MM.md` em `02 - Áreas/Bia/`. Mesmo padrão de
`bills.py`: leitura direto do disco via `safe_vault_path()` (sem MCP),
escrita via MCP (`vault_writer._mcp_call`). Formato de tabela único (não há
notas legadas com formatos diferentes, ao contrário de `bills.py`):

    | Cliente | Data | Dia | Hora | Status |
    |---|---|---|---|---|
    | Fulana | 03/09 | quarta | 15:00 | confirmado |

`Status` é `confirmado` ou `cancelado` — cancelamento marca a linha em vez de
removê-la (histórico do mês fica visível na nota).

Conflito de horário (decisão fechada): mesmo horário exato no mesmo dia. Sem
janela de duração — a Bia controla o espaçamento entre atendimentos ela
mesma.
"""

import datetime
import logging
import re

from app.services.vault_search import safe_vault_path
from app.services.vault_writer import VaultWriteError, _mcp_call

logger = logging.getLogger(__name__)

_BIA_DIR = "02 - Áreas/Bia"

_MES_NOME = {
    1: "janeiro", 2: "fevereiro", 3: "março", 4: "abril", 5: "maio",
    6: "junho", 7: "julho", 8: "agosto", 9: "setembro", 10: "outubro",
    11: "novembro", 12: "dezembro",
}

_DIA_SEMANA_PT = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]

_SEP_RE = re.compile(r"^\|[\s:-]+\|")


def _agenda_relpath(ano: int, mes: int) -> str:
    return f"{_BIA_DIR}/Agenda - {ano:04d}-{mes:02d}.md"


def _split_row(line: str) -> list[str] | None:
    line = line.strip()
    if not line.startswith("|"):
        return None
    cols = [c.strip() for c in line.strip("|").split("|")]
    return cols or None


def _col_index(headers: list[str], *keywords: str) -> int | None:
    for idx, h in enumerate(headers):
        low = h.lower()
        if any(kw in low for kw in keywords):
            return idx
    return None


def _header_cols(headers: list[str]) -> dict[str, int] | None:
    """Mapa `{"cliente","data","dia","hora","status"} -> índice` se a linha
    for reconhecível como cabeçalho da tabela de agenda (exige Cliente +
    Hora, no mínimo); senão `None`."""
    idx_cliente = _col_index(headers, "cliente")
    idx_hora = _col_index(headers, "hora")
    if idx_cliente is None or idx_hora is None:
        return None
    return {
        "cliente": idx_cliente,
        "data": _col_index(headers, "data"),
        "dia": _col_index(headers, "dia"),
        "hora": idx_hora,
        "status": _col_index(headers, "status"),
    }


def _iter_tables(lines: list[str]):
    """Gera `(row_idx, cols, col_idx)` para cada linha de dados de qualquer
    tabela de agenda reconhecida no arquivo."""
    n = len(lines)
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("|") and not _SEP_RE.match(stripped):
            headers = _split_row(stripped) or []
            col_idx = _header_cols(headers)
            if col_idx is not None:
                i += 1
                while i < n:
                    row_stripped = lines[i].strip()
                    if not row_stripped.startswith("|"):
                        break
                    if _SEP_RE.match(row_stripped):
                        i += 1
                        continue
                    cols = _split_row(row_stripped)
                    if cols and len(cols) > max(v for v in col_idx.values() if v is not None):
                        yield i, cols, col_idx
                    i += 1
                continue
        i += 1


def parse_agenda(content: str) -> list[dict]:
    """Extrai as linhas da tabela de agenda: `{"cliente","data" (DD/MM),
    "dia_semana","hora" (HH:MM),"status"}`. Nunca levanta exceção — devolve
    `[]` se não achar nenhuma tabela reconhecível."""
    rows: list[dict] = []
    for _, cols, col_idx in _iter_tables(content.splitlines()):
        cliente = cols[col_idx["cliente"]]
        if not cliente:
            continue
        rows.append({
            "cliente": cliente,
            "data": cols[col_idx["data"]] if col_idx["data"] is not None else "",
            "dia_semana": cols[col_idx["dia"]] if col_idx["dia"] is not None else "",
            "hora": cols[col_idx["hora"]],
            "status": (
                cols[col_idx["status"]].strip().lower()
                if col_idx["status"] is not None else "confirmado"
            ),
        })
    return rows


def get_agenda_do_mes(ano: int, mes: int) -> list[dict] | None:
    """Lê e parseia `Agenda - AAAA-MM.md` direto do disco. Devolve `None` se
    a nota do mês ainda não existe."""
    path = safe_vault_path(_agenda_relpath(ano, mes))
    if path is None or not path.is_file():
        logger.info("agenda_bia: nota de %04d-%02d não encontrada", ano, mes)
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("agenda_bia: falha ao ler nota de %04d-%02d (%r)", ano, mes, exc)
        return None
    return parse_agenda(content)


def get_agenda_dia(data: datetime.date) -> list[dict]:
    """Agendamentos não cancelados de um dia específico."""
    entradas = get_agenda_do_mes(data.year, data.month) or []
    data_str = data.strftime("%d/%m")
    return [e for e in entradas if e["data"] == data_str and e["status"] != "cancelado"]


def get_bia_agenda_today() -> list[dict]:
    """Usado pelo endpoint `GET /proactive/bia-agenda-today` — lista enxuta
    (cliente/hora/status) do dia corrente, ordenada por horário."""
    entradas = sorted(get_agenda_dia(datetime.date.today()), key=lambda e: e["hora"])
    return [{"cliente": e["cliente"], "hora": e["hora"], "status": e["status"]} for e in entradas]


def format_agenda_dia_message(entradas: list[dict], data: datetime.date) -> str:
    """Formata a lista de agendamentos de um dia pronta pro WhatsApp — usado
    pela consulta em linguagem natural e por `/proactive/bia-agenda-today-message`."""
    rotulo = "hoje" if data == datetime.date.today() else data.strftime("%d/%m")
    if not entradas:
        return f"📅 Nenhum cliente agendado para {rotulo}."
    ordenadas = sorted(entradas, key=lambda e: e["hora"])
    linhas = [f"📅 Clientes de {rotulo}:"]
    linhas += [f"- {e['hora']} — {e['cliente']}" for e in ordenadas]
    return "\n".join(linhas)


def _nota_nova(ano: int, mes: int) -> str:
    mes_label = f"{_MES_NOME[mes].title()}/{ano}"
    return (
        "---\n"
        "tipo: agenda-bia\n"
        f"mes: {ano:04d}-{mes:02d}\n"
        "---\n\n"
        f"# Agenda — {mes_label}\n\n"
        "| Cliente | Data | Dia | Hora | Status |\n"
        "|---|---|---|---|---|\n"
    )


def _insert_row(content: str, linha_nova: str) -> str:
    """Insere `linha_nova` no fim da última tabela de agenda reconhecida
    (mesmo que ela ainda esteja vazia, caso da nota recém-criada); se não
    achar nenhuma, anexa uma tabela nova no fim do arquivo."""
    lines = content.splitlines()
    n = len(lines)
    insert_at: int | None = None
    i = 0
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("|") and not _SEP_RE.match(stripped):
            headers = _split_row(stripped) or []
            if _header_cols(headers) is not None:
                j = i + 1
                if j < n and _SEP_RE.match(lines[j].strip()):
                    j += 1
                while j < n and lines[j].strip().startswith("|"):
                    j += 1
                insert_at = j
                i = j
                continue
        i += 1
    if insert_at is None:
        bloco = ["", "| Cliente | Data | Dia | Hora | Status |", "|---|---|---|---|---|", linha_nova]
        novas = lines + bloco
    else:
        novas = lines[:insert_at] + [linha_nova] + lines[insert_at:]
    texto = "\n".join(novas)
    return texto + "\n" if content.endswith("\n") else texto


async def add_agendamento(cliente: str, data: datetime.date, hora: str) -> dict:
    """Registra `cliente` em `data`/`hora` na nota do mês (cria a nota se
    ainda não existir). Recusa gravar se já houver alguém não-cancelado no
    mesmo dia+horário exato — devolve `{"conflict": True}` nesse caso, sem
    escrever. Nunca levanta exceção."""
    relpath = _agenda_relpath(data.year, data.month)
    path = safe_vault_path(relpath)
    if path is not None and path.is_file():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return {"success": False, "conflict": False, "message": f"falha ao ler a agenda: {exc}"}
    else:
        content = _nota_nova(data.year, data.month)

    data_str = data.strftime("%d/%m")
    conflito = next(
        (
            e for e in parse_agenda(content)
            if e["data"] == data_str and e["hora"] == hora and e["status"] != "cancelado"
        ),
        None,
    )
    if conflito is not None:
        return {
            "success": False, "conflict": True,
            "message": (
                f"conflito: {conflito['cliente']} já está marcada às {hora} "
                f"no dia {data_str} — escolhe outro horário"
            ),
        }

    dia_semana = _DIA_SEMANA_PT[data.weekday()]
    linha = f"| {cliente} | {data_str} | {dia_semana} | {hora} | confirmado |"
    novo_content = _insert_row(content, linha)

    try:
        await _mcp_call("create_vault_file", {"path": relpath, "content": novo_content})
    except VaultWriteError as exc:
        return {"success": False, "conflict": False, "message": f"falha ao gravar no vault: {exc}"}

    logger.info("agenda_bia: %r agendada para %s %s em %s", cliente, data_str, hora, relpath)
    return {
        "success": True, "conflict": False,
        "message": f"{cliente} agendada para {data_str} ({dia_semana}) às {hora}",
    }


def _find_row(
    lines: list[str], low_nome: str, data_str: str | None
) -> tuple[int, list[str], dict[str, int]] | None:
    """1ª linha não-cancelada cujo cliente bate `low_nome` (substring,
    case-insensitive) e, se `data_str` for dado, cuja data bate também."""
    for idx, cols, col_idx in _iter_tables(lines):
        nome = cols[col_idx["cliente"]].lower()
        if low_nome not in nome:
            continue
        if data_str is not None and col_idx["data"] is not None and cols[col_idx["data"]] != data_str:
            continue
        status = cols[col_idx["status"]].strip().lower() if col_idx["status"] is not None else "confirmado"
        if status == "cancelado":
            continue
        return idx, cols, col_idx
    return None


async def cancel_agendamento(cliente: str, data: datetime.date | None) -> dict:
    """Marca o agendamento de `cliente` como `cancelado` (mantém a linha —
    histórico do mês). Se `data` não for informada, procura no mês corrente.
    Nunca levanta exceção."""
    hoje = datetime.date.today()
    ano, mes = (data.year, data.month) if data else (hoje.year, hoje.month)
    relpath = _agenda_relpath(ano, mes)
    path = safe_vault_path(relpath)
    if path is None or not path.is_file():
        return {"success": False, "message": f"agenda de {ano:04d}-{mes:02d} não encontrada"}
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"success": False, "message": f"falha ao ler a agenda: {exc}"}

    lines = content.splitlines()
    data_str = data.strftime("%d/%m") if data else None
    achado = _find_row(lines, cliente.strip().lower(), data_str)
    if achado is None:
        return {"success": False, "message": f"agendamento de {cliente!r} não encontrado"}
    idx, cols, col_idx = achado
    if col_idx["status"] is None:
        return {
            "success": False,
            "message": "a tabela dessa nota não tem coluna de Status (formato incompatível)",
        }

    nome_original = cols[col_idx["cliente"]]
    cols[col_idx["status"]] = "cancelado"
    lines[idx] = "| " + " | ".join(cols) + " |"
    novo_content = "\n".join(lines)
    if content.endswith("\n"):
        novo_content += "\n"

    try:
        await _mcp_call("create_vault_file", {"path": relpath, "content": novo_content})
    except VaultWriteError as exc:
        return {"success": False, "message": f"falha ao gravar no vault: {exc}"}

    logger.info("agenda_bia: agendamento de %r cancelado em %s", nome_original, relpath)
    return {"success": True, "message": f"Agendamento de {nome_original} cancelado"}
