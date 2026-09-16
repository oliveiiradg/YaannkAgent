"""Agenda de clientes da Bia (V5) — chat privado dela (`BIA_JID`).

Fonte: nota mensal `Agenda - AAAA-MM.md` em `02 - Áreas/Bia/`. Mesmo padrão de
`bills.py`: leitura direto do disco via `safe_vault_path()` (sem MCP),
escrita via MCP (`vault_writer._mcp_call`).

Formato (D-13): um título por dia, em ordem de data; um atendimento por linha,
em ordem de hora. Cancelamento risca a linha em vez de removê-la (histórico do
mês fica visível na nota):

    ## Quarta, 16/09
    - 08:00 — Thayna
    - ~~09:30 — Dalmacia~~ (cancelado)

Transição: o parser também lê a tabela antiga
(`| Cliente | Data | Dia | Hora | Status |`). Qualquer escrita converte a nota
inteira para o formato novo antes de mexer (`migrar_nota`).

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
_DIA_RE = re.compile(r"^##\s+[^,\d]*,?\s*(\d{1,2})/(\d{1,2})\s*$")
_ITEM_RE = re.compile(
    r"^[-*]\s+(~~)?\s*(\d{1,2}):(\d{2})\s+[—–-]\s+(.+?)\s*(~~)?\s*(\(cancelado\))?\s*$",
    re.IGNORECASE,
)


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


def _iter_itens(lines: list[str]):
    """Gera `(idx, entrada)` para cada atendimento do formato por dia."""
    dia: tuple[int, int] | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            m = _DIA_RE.match(stripped)
            dia = (int(m.group(1)), int(m.group(2))) if m else None
            continue
        if dia is None:
            continue
        m = _ITEM_RE.match(stripped)
        if not m:
            continue
        riscado, hh, mm, cliente, _, marcado = m.groups()
        yield idx, {
            "cliente": cliente.strip(),
            "data": f"{dia[0]:02d}/{dia[1]:02d}",
            "hora": f"{int(hh):02d}:{mm}",
            "status": "cancelado" if (riscado or marcado) else "confirmado",
        }


def _entradas_tabela(lines: list[str]):
    for _, cols, col_idx in _iter_tables(lines):
        cliente = cols[col_idx["cliente"]]
        if not cliente:
            continue
        yield {
            "cliente": cliente,
            "data": cols[col_idx["data"]] if col_idx["data"] is not None else "",
            "dia_semana": cols[col_idx["dia"]] if col_idx["dia"] is not None else "",
            "hora": cols[col_idx["hora"]],
            "status": (
                cols[col_idx["status"]].strip().lower()
                if col_idx["status"] is not None else "confirmado"
            ),
        }


def _dia_semana(data_str: str, ano: int | None) -> str:
    try:
        dia, mes = (int(x) for x in data_str.split("/"))
        return _DIA_SEMANA_PT[datetime.date(ano or datetime.date.today().year, mes, dia).weekday()]
    except ValueError:
        return ""


def parse_agenda(content: str, ano: int | None = None) -> list[dict]:
    """Atendimentos da nota, nos dois formatos (por dia e tabela antiga):
    `{"cliente","data" (DD/MM),"dia_semana","hora" (HH:MM),"status"}`.
    Nunca levanta exceção — devolve `[]` se não achar nada reconhecível."""
    lines = content.splitlines()
    rows = list(_entradas_tabela(lines))
    for _, entrada in _iter_itens(lines):
        rows.append({**entrada, "dia_semana": _dia_semana(entrada["data"], ano)})
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
    return parse_agenda(content, ano)


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
        f"# Agenda — {mes_label}\n"
    )


def _linha_item(hora: str, cliente: str, cancelado: bool = False) -> str:
    return f"- ~~{hora} — {cliente}~~ (cancelado)" if cancelado else f"- {hora} — {cliente}"


def _titulo_dia(data: datetime.date) -> str:
    return f"## {_DIA_SEMANA_PT[data.weekday()].title()}, {data:%d/%m}"


def _inserir(lines: list[str], data: datetime.date, hora: str, linha: str) -> list[str]:
    """Insere `linha` sob o título do dia (criado na ordem de data se faltar),
    na ordem de hora."""
    alvo = (data.month, data.day)
    titulos = [
        (idx, (int(m.group(2)), int(m.group(1))))
        for idx, line in enumerate(lines)
        if (m := _DIA_RE.match(line.strip()))
    ]

    for idx, dia in titulos:
        if dia == alvo:
            fim = next(
                (j for j in range(idx + 1, len(lines)) if lines[j].strip().startswith("#")),
                len(lines),
            )
            insert_at = idx + 1
            for j in range(idx + 1, fim):
                m = _ITEM_RE.match(lines[j].strip())
                if not m:
                    continue
                if f"{int(m.group(2)):02d}:{m.group(3)}" > hora:
                    break
                insert_at = j + 1
            return lines[:insert_at] + [linha] + lines[insert_at:]
        if dia > alvo:
            return lines[:idx] + [_titulo_dia(data), linha, ""] + lines[idx:]

    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines + ["", _titulo_dia(data), linha]


def migrar_nota(content: str, ano: int) -> str:
    """Converte as tabelas antigas da nota para o formato por dia. O resto da
    nota (frontmatter, título, texto livre) fica como está. Sem tabela, devolve
    `content` sem mudança."""
    lines = content.splitlines()
    entradas = list(_entradas_tabela(lines))
    if not entradas:
        return content
    remover = {idx for idx, _, _ in _iter_tables(lines)}
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and (
            _SEP_RE.match(stripped) or _header_cols(_split_row(stripped) or []) is not None
        ):
            remover.add(idx)
    novas: list[str] = []
    for idx, line in enumerate(lines):
        if idx in remover or (not line.strip() and novas and not novas[-1].strip()):
            continue
        novas.append(line)
    for entrada in entradas:
        try:
            dia, mes = (int(x) for x in entrada["data"].split("/"))
            data = datetime.date(ano, mes, dia)
        except ValueError:
            logger.warning("agenda_bia: linha sem data válida na migração: %r", entrada)
            continue
        linha = _linha_item(entrada["hora"], entrada["cliente"], entrada["status"] == "cancelado")
        novas = _inserir(novas, data, entrada["hora"], linha)
    return "\n".join(novas).rstrip("\n") + "\n"


def _migra_se_tabela(content: str, ano: int, relpath: str) -> str:
    migrada = migrar_nota(content, ano)
    if migrada != content:
        logger.info("agenda_bia: %s convertida de tabela para o formato por dia", relpath)
    return migrada


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
            logger.warning("agenda_bia: falha ao ler %s (%r)", relpath, exc)
            return {"success": False, "conflict": False, "message": f"falha ao ler a agenda: {exc}"}
    else:
        content = _nota_nova(data.year, data.month)

    data_str = data.strftime("%d/%m")
    conflito = next(
        (
            e for e in parse_agenda(content, data.year)
            if e["data"] == data_str and e["hora"] == hora and e["status"] != "cancelado"
        ),
        None,
    )
    if conflito is not None:
        logger.info(
            "agenda_bia: conflito ao agendar %r em %s %s (já tem %r)",
            cliente, data_str, hora, conflito["cliente"],
        )
        return {
            "success": False, "conflict": True,
            "message": (
                f"conflito: {conflito['cliente']} já está marcada às {hora} "
                f"no dia {data_str} — escolhe outro horário"
            ),
        }

    lines = _migra_se_tabela(content, data.year, relpath).splitlines()
    novo_content = "\n".join(_inserir(lines, data, hora, _linha_item(hora, cliente))) + "\n"

    try:
        await _mcp_call("create_vault_file", {"path": relpath, "content": novo_content})
    except VaultWriteError as exc:
        logger.warning("agenda_bia: falha ao gravar agendamento de %r em %s (%r)", cliente, relpath, exc)
        return {"success": False, "conflict": False, "message": f"falha ao gravar no vault: {exc}"}

    dia_semana = _DIA_SEMANA_PT[data.weekday()]
    logger.info("agenda_bia: %r agendada para %s %s em %s", cliente, data_str, hora, relpath)
    return {
        "success": True, "conflict": False,
        "message": f"{cliente} agendada para {data_str} ({dia_semana}) às {hora}",
    }


async def cancel_agendamento(cliente: str, data: datetime.date | None) -> dict:
    """Risca o agendamento de `cliente` (mantém a linha — histórico do mês).
    Se `data` não for informada, procura no mês corrente. Nunca levanta
    exceção."""
    hoje = datetime.date.today()
    ano, mes = (data.year, data.month) if data else (hoje.year, hoje.month)
    relpath = _agenda_relpath(ano, mes)
    path = safe_vault_path(relpath)
    if path is None or not path.is_file():
        return {"success": False, "message": f"agenda de {ano:04d}-{mes:02d} não encontrada"}
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("agenda_bia: falha ao ler %s (%r)", relpath, exc)
        return {"success": False, "message": f"falha ao ler a agenda: {exc}"}

    lines = _migra_se_tabela(content, ano, relpath).splitlines()
    data_str = data.strftime("%d/%m") if data else None
    low_nome = cliente.strip().lower()
    achado = next(
        (
            (idx, e) for idx, e in _iter_itens(lines)
            if low_nome in e["cliente"].lower()
            and (data_str is None or e["data"] == data_str)
            and e["status"] != "cancelado"
        ),
        None,
    )
    if achado is None:
        logger.info("agenda_bia: agendamento de %r não encontrado em %s", cliente, relpath)
        return {"success": False, "message": f"agendamento de {cliente!r} não encontrado"}
    idx, entrada = achado
    lines[idx] = _linha_item(entrada["hora"], entrada["cliente"], cancelado=True)
    novo_content = "\n".join(lines) + "\n"

    try:
        await _mcp_call("create_vault_file", {"path": relpath, "content": novo_content})
    except VaultWriteError as exc:
        logger.warning("agenda_bia: falha ao gravar cancelamento em %s (%r)", relpath, exc)
        return {"success": False, "message": f"falha ao gravar no vault: {exc}"}

    logger.info("agenda_bia: agendamento de %r cancelado em %s", entrada["cliente"], relpath)
    return {"success": True, "message": f"Agendamento de {entrada['cliente']} cancelado"}
