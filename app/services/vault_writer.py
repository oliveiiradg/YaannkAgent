"""Escrita no vault via MCP do Obsidian — Yaannk no grupo do casal (Tarefa 5).

`save_to_vault(tipo, conteudo, quem)` anexa a informação na nota certa da pasta
`03 - Vida/` e devolve uma frase de confirmação (o que foi salvo e onde) pronta
para mandar no WhatsApp.

`detect_save_intent(text)` decide, sem LLM, se a mensagem é um pedido de registro
e de que tipo. A extração de campos (valor, data, item) é heurística leve — o
fluxo "anota aí" precisa ser rápido, então não passa pelo modelo. O texto
original vai sempre na linha salva, então nada se perde se a heurística errar
uma categoria.
"""

import calendar
import datetime
import json
import logging
import re
import sqlite3

import httpx

from app.config import settings
from app.services.db import get_connection

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0

# tipo -> (caminho da nota, heading sob o qual anexar)
_TARGETS: dict[str, tuple[str, str]] = {
    "gasto": ("03 - Vida/Finanças/Gastos.md", "Registros"),
    "data": ("03 - Vida/Datas/Datas Importantes.md", "Registros"),
    "lembrete": ("03 - Vida/Datas/Datas Importantes.md", "Registros"),
    "lista": ("03 - Vida/Listas/Lista de Compras.md", "Itens"),
}

_NOME_NOTA = {
    "gasto": "03 - Vida/Finanças/Gastos.md",
    "data": "03 - Vida/Datas/Datas Importantes.md",
    "lembrete": "03 - Vida/Datas/Datas Importantes.md",
    "lista": "03 - Vida/Listas/Lista de Compras.md",
}

# Verbos/expressões que sinalizam "registra isso pra mim".
_SAVE_INTENT_RE = re.compile(
    r"\b("
    r"anota|anote|anotar|anota[ -]?a[íi]|"
    r"registra|registre|registrar|"
    r"salva|salve|salvar|guarda\s+(?:isso|que|essa)|"
    r"gastei|gastamos|paguei|pagamos|comprei|compramos|"
    r"lembra\s+que|lembre\s+que|lembrete|me\s+lembra|"
    r"adiciona|adicione|coloca\s+na\s+lista|p[õo]e\s+na\s+lista|bota\s+na\s+lista|"
    r"marca\s+(?:a|no|pra)|agenda(?:r)?"
    r")\b",
    re.IGNORECASE,
)

_MONEY_RE = re.compile(
    r"(?:r\$\s*)?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\d+(?:[.,]\d{1,2})?)\s*"
    r"(?:reais|real|conto|pila)?",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?\b")

_CATEGORIA_KEYWORDS = {
    "Mercado": ("mercado", "supermercado", "feira", "hortifruti", "açougue", "padaria"),
    "Alimentação": ("ifood", "restaurante", "lanche", "almoço", "jantar", "delivery", "pizza"),
    "Transporte": ("uber", "99", "gasolina", "combustível", "ônibus", "metrô", "passagem", "estacionamento"),
    "Moradia": ("aluguel", "condomínio", "luz", "água", "energia", "gás", "internet", "iptu"),
    "Saúde": ("farmácia", "remédio", "médico", "consulta", "exame", "dentista", "plano de saúde"),
    "Lazer": ("cinema", "show", "viagem", "bar", "netflix", "spotify", "streaming"),
    "Contas": ("boleto", "fatura", "cartão", "conta de", "assinatura"),
}

# Frases-gatilho que só entram na sub-heurística; ajudam a separar "gasto" de
# "lista" quando ambas as palavras aparecem ("comprei pão" vs "comprar pão").
_LISTA_RE = re.compile(
    r"\b(lista\s+de\s+compras|na\s+lista|comprar|preciso\s+de|falta(?:ndo)?|acabou\s+o|t[áa]\s+acabando)\b",
    re.IGNORECASE,
)
_DATA_RE = re.compile(
    r"\b(anivers[áa]rio|casamento|consulta|reuni[ãa]o|compromisso|renova(?:r|ção)|"
    r"vence|vencimento|feriado|viagem|dia\s+\d{1,2})\b",
    re.IGNORECASE,
)
_GASTO_RE = re.compile(
    r"\b(gastei|gastamos|paguei|pagamos|comprei|compramos|gasto|custou|"
    r"r\$|reais|boleto|fatura|conta\s+de)\b",
    re.IGNORECASE,
)


class VaultWriteError(Exception):
    """Falha ao escrever no vault via MCP — o webhook responde com um aviso."""


# Verbos imperativos de registro que vencem qualquer cara de pergunta.
_EXPLICIT_SAVE_RE = re.compile(
    r"\b(anota|anote|anotar|registra|registre|registrar|salva|salve|salvar|"
    r"adiciona|adicione|coloca\s+na\s+lista|p[õo]e\s+na\s+lista|bota\s+na\s+lista)\b",
    re.IGNORECASE,
)
_PERGUNTA_RE = re.compile(
    r"\?|\b(quanto|quantos|quantas|quando|qual|quais|cad[êe]|onde|"
    r"o\s+que|pq|por\s*que|porqu[êe])\b",
    re.IGNORECASE,
)


def detect_save_intent(text: str) -> str | None:
    """Retorna o tipo (`gasto` | `data` | `lembrete` | `lista`) se o texto é um
    pedido de registro; senão `None`."""
    if not _SAVE_INTENT_RE.search(text):
        return None
    # "quanto gastei esse mês?" tem verbo + número mas é consulta, não registro.
    if _PERGUNTA_RE.search(text) and not _EXPLICIT_SAVE_RE.search(text):
        return None

    has_money = bool(_MONEY_RE.search(text)) and bool(
        re.search(r"r\$|reais|real|\d", text, re.IGNORECASE)
    )
    if _GASTO_RE.search(text) and has_money:
        return "gasto"
    if _LISTA_RE.search(text):
        return "lista"
    if _DATA_RE.search(text) or _DATE_RE.search(text):
        return "data"
    if re.search(r"\blembr", text, re.IGNORECASE):
        return "lembrete"
    if _GASTO_RE.search(text):
        return "gasto"
    # Pedido de registro explícito mas tipo ambíguo: cai em lembrete (catch-all
    # da nota de Datas), e a confirmação diz onde foi — fácil de corrigir.
    return "lembrete"


def _hoje() -> str:
    return datetime.date.today().strftime("%d/%m/%Y")


def _extrai_valor(text: str) -> str:
    m = _MONEY_RE.search(text)
    if not m or not re.search(r"\d", m.group(1)):
        return "?"
    return f"R$ {m.group(1)}"


def _extrai_data_evento(text: str) -> str:
    m = _DATE_RE.search(text)
    if not m:
        return _hoje()
    dia, mes, ano = m.group(1), m.group(2), m.group(3)
    if not ano:
        ano = str(datetime.date.today().year)
    elif len(ano) == 2:
        ano = "20" + ano
    return f"{int(dia):02d}/{int(mes):02d}/{ano}"


# Token monetário para o dual-write SQL — próprio (não o _MONEY_RE, que corta
# "33.50" em "33"): pega o número mais longo, com milhar e/ou decimal.
_VALOR_SQL_RE = re.compile(r"(?<![\w,.])(\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)(?![\w])")


def _valor_float(text: str) -> float | None:
    """Parseia o primeiro valor monetário do texto (formato BR) para float.
    `"R$ 1.200,50"` → 1200.5, `"gastei 50"` → 50.0, `"12,90"` → 12.9,
    `"uber 33.50"` → 33.5. Sem dígito → None."""
    m = _VALOR_SQL_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    if "." in raw and "," in raw:            # 1.200,50 → ponto=milhar, vírgula=decimal
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:                          # 12,90 → vírgula decimal
        raw = raw.replace(",", ".")
    elif raw.count(".") == 1 and len(raw.split(".")[1]) != 3:
        pass                                 # 12.90 → ponto decimal, mantém
    else:
        raw = raw.replace(".", "")           # 1.200 → milhar
    try:
        return round(float(raw), 2)
    except ValueError:
        return None


# Datas relativas faladas ("gastei 30 no lanche ontem"). "semana passada" e
# "mês passado" não nomeiam um dia único — resolvem para uma aproximação que
# cai no bucket certo das agregações de `expenses.py`.
_REL_DATE_RE = re.compile(
    r"\b(hoje|ontem|anteontem|semana\s+passada|m[êe]s\s+passado)\b", re.IGNORECASE
)


def _mes_anterior(d: datetime.date) -> datetime.date:
    """Mesmo dia do mês anterior, clampado ao último dia (31/03 → 28/02)."""
    ano, mes = (d.year - 1, 12) if d.month == 1 else (d.year, d.month - 1)
    ultimo = calendar.monthrange(ano, mes)[1]
    return datetime.date(ano, mes, min(d.day, ultimo))


def _data_relativa(token: str, hoje: datetime.date) -> datetime.date:
    t = re.sub(r"\s+", " ", token.strip().lower())
    if t == "ontem":
        return hoje - datetime.timedelta(days=1)
    if t == "anteontem":
        return hoje - datetime.timedelta(days=2)
    if t == "semana passada":
        return hoje - datetime.timedelta(days=7)
    if t in ("mês passado", "mes passado"):
        return _mes_anterior(hoje)
    return hoje  # "hoje"


def _data_iso_gasto(text: str) -> str:
    """Data do gasto em ISO8601 (YYYY-MM-DD).

    Prioridade: data explícita no texto (`dd/mm[/aaaa]`) > data relativa
    falada (`ontem`, `anteontem`, `semana passada`, `mês passado`) > hoje.
    A explícita vence porque é mais específica ("gastei 30 dia 12/08 ontem"
    deve gravar 12/08).
    """
    hoje = datetime.date.today()

    m = _DATE_RE.search(text)
    if m:
        dia, mes, ano = m.group(1), m.group(2), m.group(3)
        if not ano:
            ano = str(hoje.year)
        elif len(ano) == 2:
            ano = "20" + ano
        try:
            return datetime.date(int(ano), int(mes), int(dia)).isoformat()
        except ValueError:
            return hoje.isoformat()

    rel = _REL_DATE_RE.search(text)
    if rel:
        return _data_relativa(rel.group(1), hoje).isoformat()

    return hoje.isoformat()


def record_expense(chat_id: str | None, autor: str, conteudo: str) -> None:
    """Dual-write do gasto na tabela SQL `expenses` (Fase 7). Best-effort:
    o markdown do Obsidian é a fonte de verdade, então qualquer falha aqui é
    logada e engolida — não quebra a confirmação no WhatsApp."""
    valor = _valor_float(conteudo)
    if valor is None:
        logger.warning("record_expense: sem valor numérico na mensagem — SQL pulado")
        return
    try:
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO expenses "
                "(chat_id, autor, valor, categoria, descricao, data, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chat_id,
                    autor,
                    valor,
                    _categoria(conteudo),
                    _limpa_descricao(conteudo),
                    _data_iso_gasto(conteudo),
                    datetime.datetime.now().isoformat(timespec="seconds"),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("record_expense: gasto registrado no SQL (cat=%s)", _categoria(conteudo))
    except sqlite3.Error as exc:
        logger.warning("record_expense: INSERT falhou (%r) — só markdown gravado", exc)


def _categoria(text: str) -> str:
    low = text.lower()
    for cat, kws in _CATEGORIA_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return cat
    return "Outros"


def _limpa_descricao(text: str) -> str:
    """Tira o verbo de comando, o valor monetário e as palavras de ligação
    das pontas, deixando o miolo do que foi dito."""
    desc = _SAVE_INTENT_RE.sub(" ", text)
    desc = re.sub(r"r\$\s*\d[\d.,]*\s*(reais|real)?", " ", desc, flags=re.IGNORECASE)
    desc = re.sub(r"\b\d[\d.,]*\s*(reais|real|conto|pila)\b", " ", desc, flags=re.IGNORECASE)
    desc = _DATE_RE.sub(" ", desc)
    # número solto de dinheiro ("12,50", "1.200") — data já foi removida acima
    desc = re.sub(r"\b\d{1,3}(?:\.\d{3})+(?:,\d{2})?\b|\b\d+,\d{2}\b", " ", desc)
    # palavras de ligação/dêiticos soltos nas pontas
    filler = (
        r"que|dia|de|do|da|no|na|o|a|os|as|um|uma|pra|para|com|isso|a[íi]|"
        r"hoje|amanh[ãa]|ontem|essa|esse|meu|minha|nossa|nosso|lista|compras"
    )
    desc = re.sub(rf"^(?:\s*\b(?:{filler})\b)+", "", desc.strip(), flags=re.IGNORECASE)
    desc = re.sub(rf"(?:\b(?:{filler})\b\s*)+$", "", desc.strip(), flags=re.IGNORECASE)
    desc = re.sub(r"\s{2,}", " ", desc).strip(" ,.-:;")
    return desc or text.strip()


def _monta_linha(tipo: str, conteudo: str, quem: str) -> tuple[str, str]:
    """Devolve (linha_markdown, frase_de_confirmacao_do_conteudo)."""
    if tipo == "gasto":
        valor = _extrai_valor(conteudo)
        cat = _categoria(conteudo)
        desc = _limpa_descricao(conteudo)
        # Mesma data que vai para o SQL (`record_expense`) — inclui as relativas
        # ("ontem"), só reformatada para dd/mm/aaaa da tabela do Obsidian.
        data = datetime.date.fromisoformat(_data_iso_gasto(conteudo)).strftime("%d/%m/%Y")
        linha = f"| {data} | {desc} | {valor} | {cat} | {quem} |"
        return linha, f"{desc} — {valor} ({cat})"

    if tipo in ("data", "lembrete"):
        data_ev = _extrai_data_evento(conteudo)
        desc = _limpa_descricao(conteudo)
        rotulo = "Lembrete" if tipo == "lembrete" else "Data"
        linha = f"| {data_ev} | {desc} | {rotulo} | {quem} |"
        return linha, f"{desc} — {data_ev}"

    # lista
    item = _limpa_descricao(conteudo)
    linha = f"- [ ] {item} _(add: {quem}, {_hoje()})_"
    return linha, item


async def _mcp_call(name: str, arguments: dict) -> dict:
    if not settings.obsidian_mcp_token:
        raise VaultWriteError("OBSIDIAN_MCP_TOKEN não configurado")

    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Authorization": f"Bearer {settings.obsidian_mcp_token}",
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.post(
                settings.obsidian_mcp_url, json=body, headers=headers
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise VaultWriteError(f"erro HTTP chamando o MCP: {exc!r}") from exc

    content_type = response.headers.get("content-type", "")
    payload: dict | None = None
    if "text/event-stream" in content_type:
        for line in response.text.splitlines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data:"):].strip())
                break
    else:
        payload = response.json()

    if payload is None:
        raise VaultWriteError("resposta do MCP em formato inesperado")
    if "error" in payload:
        raise VaultWriteError(f"MCP retornou erro JSON-RPC: {payload['error']}")
    result = payload.get("result", {})
    if result.get("isError"):
        raise VaultWriteError(f"MCP retornou isError: {result.get('content')}")
    return result


def _splice_linha(conteudo: str, heading: str, linha: str) -> str:
    """Insere `linha` logo após a última linha não-vazia da seção `## {heading}`
    (o fim da tabela ou da lista), com uma única quebra — sem linha em branco no
    meio, que quebraria a renderização da tabela no Obsidian."""
    linhas = conteudo.splitlines()
    try:
        h_idx = next(
            i for i, ln in enumerate(linhas)
            if ln.strip().lower().lstrip("#").strip() == heading.lower()
            and ln.lstrip().startswith("#")
        )
    except StopIteration:
        # sem o heading esperado: anexa no fim do arquivo
        return conteudo.rstrip() + "\n" + linha + "\n"

    # fim da seção = próximo heading de nível <= ou fim do arquivo
    end = len(linhas)
    for i in range(h_idx + 1, len(linhas)):
        if linhas[i].lstrip().startswith("#"):
            end = i
            break

    last_content = h_idx
    for i in range(h_idx + 1, end):
        if linhas[i].strip():
            last_content = i

    linhas.insert(last_content + 1, linha)
    return "\n".join(linhas) + ("\n" if conteudo.endswith("\n") else "")


async def _append_sob_heading(path: str, heading: str, linha: str) -> None:
    """Lê a nota via MCP, insere `linha` no fim da seção e regrava.

    Read-modify-write (em vez de patch append do connector, que insere uma
    linha em branco antes e quebra a tabela). Volume do casal é baixo, então
    a corrida de escrita concorrente é aceitável.
    """
    result = await _mcp_call("get_vault_file", {"path": path, "format": "text"})
    try:
        conteudo = result["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VaultWriteError(f"get_vault_file devolveu formato inesperado: {exc!r}") from exc

    novo = _splice_linha(conteudo, heading, linha)
    await _mcp_call("create_vault_file", {"path": path, "content": novo})


async def save_to_vault(
    tipo: str, conteudo: str, quem: str = "?", voc: str = "", chat_id: str | None = None
) -> str:
    """Salva `conteudo` na nota de `03 - Vida/` correspondente a `tipo` e
    devolve a frase de confirmação para o WhatsApp.

    tipo ∈ {gasto, data, lembrete, lista}. `voc` é um vocativo opcional já
    formatado (ex: ", Alice") que entra na frase de confirmação. `chat_id`
    identifica a conversa e é gravado no dual-write SQL de gastos (Fase 7).
    Levanta VaultWriteError se o MCP do Obsidian não estiver acessível.
    """
    if tipo not in _TARGETS:
        raise VaultWriteError(f"tipo desconhecido: {tipo!r}")

    path, heading = _TARGETS[tipo]
    linha, resumo = _monta_linha(tipo, conteudo, quem)

    await _append_sob_heading(path, heading, linha)
    logger.info("save_to_vault: registro do tipo %s anexado em %s", tipo, path)

    # Fase 7 — dual-write: gasto também vai para a tabela SQL `expenses`, para
    # consultas financeiras determinísticas (ver app/services/expenses.py).
    if tipo == "gasto":
        record_expense(chat_id, quem, conteudo)

    onde = _NOME_NOTA[tipo]
    rotulo = {
        "gasto": "Anotado",
        "data": "Anotado",
        "lembrete": "Anotado",
        "lista": "Adicionado à lista",
    }[tipo]
    return f"✅ {rotulo}{voc}! Registrei em _{onde}_:\n• {resumo}"
