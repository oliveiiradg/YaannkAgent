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

import asyncio
import calendar
import datetime
import json
import logging
import re
import sqlite3
from collections.abc import Callable

import httpx

from app.config import settings
from app.services.db import get_connection
from app.services.llm.base import LLMError
from app.services.llm.registry import get_provider

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0

_MES_NOME = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro",
    12: "Dezembro",
}


def _gastos_path(today: datetime.date | None = None) -> str:
    """Sessão 19: gastos passaram de nota única (`Gastos.md`) pra uma por mês —
    mesmo formato gerado por `tools/migrate_expenses_to_vault.py`."""
    today = today or datetime.date.today()
    return f"03 - Vida/Finanças/Gastos - {today.year:04d}-{today.month:02d}.md"


def _data_br_gasto(text: str) -> str:
    """Data do gasto em `dd-mm-yy` (Sessão 20) — formato da coluna Data das
    notas `Gastos - YYYY-MM.md`. Deriva da mesma resolução ISO usada antes
    (data explícita > relativa falada > hoje), só reformatada."""
    return datetime.date.fromisoformat(_data_iso_gasto(text)).strftime("%d-%m-%y")


def _gastos_arquivo_novo(path: str) -> str:
    """Frontmatter + título + cabeçalho da tabela pra um mês de gastos que
    ainda não tem nota — mesmo formato dos arquivos gerados pela migração
    (sem heading `## Registros`: só H1 + tabela)."""
    m = re.search(r"Gastos - (\d{4})-(\d{2})\.md$", path)
    ano, mes_num = m.group(1), m.group(2)
    mes_label = f"{_MES_NOME[int(mes_num)]} {ano}"
    return (
        "---\n"
        "tipo: gastos\n"
        f"mes: {ano}-{mes_num}\n"
        f"atualizado: {datetime.date.today().isoformat()}\n"
        "---\n\n"
        f"# Gastos — {mes_label}\n\n"
        "| Data       | Pessoa  | Categoria   | Descrição           | Valor  |\n"
        "|------------|---------|-------------|---------------------|--------|\n"
    )


# tipo -> (caminho da nota ou função que calcula o caminho, heading sob o
# qual anexar — `None` = sempre anexa no fim do arquivo, sem procurar
# heading; usado por "gasto" porque a nota mensal não tem `## Registros`,
# só H1 + tabela).
_TARGETS: dict[str, tuple[str | Callable[[], str], str | None]] = {
    "gasto": (_gastos_path, None),
    "data": ("03 - Vida/Datas/Datas Importantes.md", "Registros"),
    "lembrete": ("03 - Vida/Datas/Datas Importantes.md", "Registros"),
    "lista": ("03 - Vida/Listas/Lista de Compras.md", "Itens"),
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


# DEPRECATED (Sessão 19) — remover após validação em produção. Sem outros
# callers além do antigo call site em `save_to_vault()` (removido nesta
# sessão; grep confirmou: só a definição e a chamada interna que caiu). O
# vault virou a única fonte de verdade dos gastos — não precisa mais do
# dual-write SQL (que, aliás, já divergia do vault antes desta sessão).
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


async def _inferir_categoria_e_resumo_kimi(descricao_bruta: str) -> tuple[str, str]:
    """Kimi infere categoria e um resumo curto da descrição, numa chamada só
    (evita dobrar a latência/custo com duas chamadas separadas). Timeout
    curto (5s) — qualquer falha devolve `("", descricao_bruta)`: categoria
    vazia e a descrição como `_limpa_descricao()` já entregava (sem
    regressão pro comportamento de antes desta correção).

    Corrige achado da Sessão 20: `_limpa_descricao()` só REMOVE padrões
    conhecidos (verbo, valor, data) — pra mensagens sem nenhum desses
    padrões (ex.: sem "gastei"/"paguei"), ela devolve o texto praticamente
    inalterado. Resumir de verdade precisa de LLM, regex não alcança."""
    prompt = (
        "Duas coisas sobre este gasto, cada uma numa linha, sem explicação:\n"
        "1. Categoria curta (máx 2 palavras em português)\n"
        "2. Resumo curto da descrição (máx 6 palavras, sem valor nem data)\n\n"
        f"Gasto: {descricao_bruta}\n\n"
        "Responda EXATAMENTE neste formato, nada mais:\n"
        "categoria: <categoria>\n"
        "descricao: <resumo>"
    )
    try:
        provider = get_provider("kimi")
        result = await asyncio.wait_for(
            provider.generate(
                [{"role": "user", "content": prompt}], max_tokens=40, timeout=5.0,
            ),
            timeout=5.0,
        )
        categoria, resumo = "", descricao_bruta
        for ln in result.text.strip().splitlines():
            baixo = ln.strip().lower()
            if baixo.startswith("categoria:"):
                categoria = ln.split(":", 1)[1].strip()
            elif baixo.startswith(("descricao:", "descrição:")):
                valor = ln.split(":", 1)[1].strip()
                if valor:
                    resumo = valor
        return categoria, resumo
    except (TimeoutError, LLMError) as exc:
        logger.warning(
            "vault_writer: Kimi falhou inferindo categoria/resumo (%r) — "
            "categoria vazia, descrição sem resumir", exc,
        )
        return "", descricao_bruta


async def _monta_linha(tipo: str, conteudo: str, quem: str) -> tuple[str, str]:
    """Devolve (linha_markdown, frase_de_confirmacao_do_conteudo)."""
    if tipo == "gasto":
        # Import tardio: `expenses.py` importa `_CATEGORIA_KEYWORDS` deste
        # módulo — import de topo aqui criaria ciclo.
        from app.services.expenses import _brl

        valor_num = _valor_float(conteudo)
        valor = f"{valor_num:.2f}" if valor_num is not None else "?"
        desc_bruta = _limpa_descricao(conteudo)
        cat, desc = await _inferir_categoria_e_resumo_kimi(desc_bruta)
        # Mesmo formato dos arquivos migrados (Sessão 19, `Gastos - YYYY-MM.md`):
        # data ISO, colunas Data | Pessoa | Categoria | Descrição | Valor.
        data = _data_br_gasto(conteudo)
        linha = f"| {data} | {quem} | {cat} | {desc} | {valor} |"
        rotulo_cat = cat or "sem categoria"
        valor_confirmacao = _brl(valor_num) if valor_num is not None else "?"
        return linha, f"{desc} — {valor_confirmacao} ({rotulo_cat})"

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


def _splice_linha(conteudo: str, heading: str | None, linha: str) -> str:
    """Insere `linha` logo após a última linha não-vazia da seção `## {heading}`
    (o fim da tabela ou da lista), com uma única quebra — sem linha em branco no
    meio, que quebraria a renderização da tabela no Obsidian.

    `heading=None` pula a busca e sempre anexa no fim do arquivo — usado por
    "gasto": a nota mensal não tem `## Registros`, só H1 + tabela (Sessão 19).
    """
    linhas = conteudo.splitlines()
    if heading is None:
        return conteudo.rstrip() + "\n" + linha + "\n"
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


async def _append_sob_heading(
    path: str, heading: str | None, linha: str,
    *, gerar_arquivo_novo: Callable[[str], str] | None = None,
) -> None:
    """Lê a nota via MCP, insere `linha` no fim da seção e regrava.

    Read-modify-write (em vez de patch append do connector, que insere uma
    linha em branco antes e quebra a tabela). Volume do casal é baixo, então
    a corrida de escrita concorrente é aceitável.

    `gerar_arquivo_novo(path)` (Sessão 19): se passado e a nota ainda não
    existir, gera o conteúdo base (frontmatter + header + cabeçalho de
    tabela) antes de inserir a linha, em vez de levantar erro. Usado por
    "gasto", cujo arquivo do mês pode não existir ainda.

    Limitação conhecida: o MCP do Obsidian não expõe uma checagem de
    existência separada da leitura — "arquivo não existe" e "MCP falhou por
    outro motivo" chegam pelo mesmo `VaultWriteError` de `get_vault_file`.
    Quando `gerar_arquivo_novo` está setado, qualquer falha de leitura vira
    "vamos criar do zero" — testado contra um mês novo real (ver diagnóstico
    desta sessão), mas não distingue com certeza os dois casos.
    """
    try:
        result = await _mcp_call("get_vault_file", {"path": path, "format": "text"})
    except VaultWriteError as exc:
        if gerar_arquivo_novo is None:
            raise
        logger.info("vault_writer: %s não encontrado (%r) — criando do zero", path, exc)
        conteudo = gerar_arquivo_novo(path)
    else:
        try:
            conteudo = result["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            # A leitura funcionou (arquivo existe), só a resposta veio num
            # formato inesperado — nunca tratar como "não existe" aqui, ou
            # sobrescreveríamos conteúdo real com um esqueleto vazio.
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
    não é mais usado aqui desde a Sessão 19 (dual-write SQL removido) —
    mantido na assinatura só pra não quebrar o call site em `webhook.py`.
    Levanta VaultWriteError se o MCP do Obsidian não estiver acessível.
    """
    if tipo not in _TARGETS:
        raise VaultWriteError(f"tipo desconhecido: {tipo!r}")

    path_or_fn, heading = _TARGETS[tipo]
    path = path_or_fn() if callable(path_or_fn) else path_or_fn
    linha, resumo = await _monta_linha(tipo, conteudo, quem)

    gerar_arquivo_novo = _gastos_arquivo_novo if tipo == "gasto" else None
    await _append_sob_heading(path, heading, linha, gerar_arquivo_novo=gerar_arquivo_novo)
    logger.info("save_to_vault: registro do tipo %s anexado em %s", tipo, path)

    onde = path
    rotulo = {
        "gasto": "Anotado",
        "data": "Anotado",
        "lembrete": "Anotado",
        "lista": "Adicionado à lista",
    }[tipo]
    return f"✅ {rotulo}{voc}! Registrei em _{onde}_:\n• {resumo}"
