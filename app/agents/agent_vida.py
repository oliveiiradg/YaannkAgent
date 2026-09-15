"""Agent Vida (Fase B) — unifica vida_casal + financeiro.

Sequência de decisão (D-07, Sessão 18): ordem real do código herdado do
`pipeline.py`, não a ordem originalmente esboçada no `design.md` — decidido
não reordenar, os regexes envolvidos são praticamente disjuntos.

    balanço → financeiro → bills → lista de compras → datas importantes → RAG

Escrita em contas fixas (marcar paga, remover, adicionar) não tem fast-path:
é decidida pelo ReAct em `_try_react()` (D-10), que chama
`bills.mark_bill_paid()`/`remove_bill()`/`add_bill()` direto.

`detect_save_intent()`/`save_to_vault()` NÃO são deste agente — já vivem em
`app/routes/webhook.py`, uma camada acima (D-08).

Fonte dos gastos variáveis (Sessão 19 → 20):
- Sessão 19 migrou do SQLite pro vault (`Gastos - YYYY-MM.md`, um por mês).
- A revisão 2 da Sessão 19 trocou leitura por path calculado por busca
  semântica — e o benchmark reprovou: 17 de 25 casos `casal` falharam, com
  o Kimi afirmando "não encontrei lançamento em agosto/2026" num mês que
  tem 7 lançamentos, e "setembro ainda não tem nada salvo" com setembro
  cheio. A busca não trazia o arquivo do mês certo de forma confiável.
- **Sessão 20 reverteu a LEITURA pro determinístico:** o período vira
  meses por cálculo de calendário, cada `Gastos - YYYY-MM.md` é lido por
  path, e a busca semântica sobrou só como fallback pra quando nenhum
  arquivo do período existir.
- **Sessão 21 tirou o Kimi também da RESPOSTA:** ele ainda agregava e
  formatava o texto final, e errou soma no benchmark (cas-03: R$ 2.265,00
  para um total de R$ 1.265,00), além de quebrar os 15 `reply_equals` de
  `casal` por formatar livre a cada rodada. Agora `_formata_expenses()`
  soma em Python e reusa os templates do `expenses.query_expenses()`
  original. O Kimi só entra no fallback de busca semântica, onde não há
  tabela pra somar.

Leitura é do disco (`safe_vault_path`), igual `bills.py` faz com
`Contas - YYYY-MM.md`; escrita continua via MCP em `vault_writer.py`.
Datas nas tabelas em `dd-mm-aa` (Sessão 20).
"""

import datetime
import json
import logging
import pathlib
import re
import time

from app.agents.base import AgentExecutor, AgentResponse
from app.config import settings
from app.services.agenda_bia import (
    add_agendamento,
    cancel_agendamento,
    format_agenda_dia_message,
    get_agenda_dia,
)
from app.services.bills import (
    BillsResult,
    _STATUS_ROTULO,
    add_bill,
    answer_bills_query,
    get_contas_do_mes,
    mark_bill_paid,
    remove_bill,
)
from app.services.expenses import (
    _brl,
    _CASAL_RE,
    _match_categoria,
    _MES_NOME,
    _parse_periodo,
    _PESSOAL_RE,
    _POR_PESSOA_RE,
)
from app.services.finance_patterns import (
    BALANCE_COMMAND_RE,
    FINANCIAL_QUERY_STRICT_RE,
)
from app.services.llm.base import LLMError, Message
from app.services.llm.registry import get_provider
from app.services.orchestrator import OrchestratorDecision
from app.services.query_decomposer import decompose_query
from app.services.vault_search import safe_vault_path
from app.services.vault_search_semantic import _strip_frontmatter, search_vault_hybrid
from app.services.vault_writer import VaultWriteError, _mcp_call, save_to_vault


logger = logging.getLogger(__name__)

_FINANCAS_FOLDER = "02 - Áreas/Finanças"

# Raiz alternativa SÓ para as notas de gasto. Existe para o benchmark: a
# fixture precisa de gastos determinísticos, mas repontar
# `vault_search.settings.vault_path` (tentativa da Sessão 20) apontava também
# o TF-IDF para o vault temporário — a perna TF-IDF do RRF voltava vazia e o
# RAG técnico regredia (tec-16, tec-17, adv-08 perderam `Bugs e Erros.md` do
# top-3). Em produção fica `None` e a leitura segue por `safe_vault_path()`.
GASTOS_ROOT_OVERRIDE: str | None = None

# Frases que pedem mais do que os 4 templates determinísticos de bills.py
# cobrem (D-09) — dispara o caminho lento (Kimi formata o BillsResult cru).
# Rascunho inicial: precisa de calibração com casos reais, como o
# BILLS_QUERY_RE precisou (Sessão 17).
_BILLS_DETAIL_MODIFIER_RE = re.compile(
    r"\b("
    r"detalhad[ao]s?|detalhe|discriminad[ao]s?|completa|na\s+[ií]ntegra|"
    r"item\s+por\s+item|cada\s+conta|todas?\s+as\s+contas|com\s+status|"
    r"s[óo]\s+as?\s+pag[ao]s?|somente\s+as?\s+pag[ao]s?|"
    r"s[óo]\s+as?\s+pendentes?|somente\s+as?\s+pendentes?"
    r")\b",
    re.IGNORECASE,
)

# Caminhos reais confirmados por busca no vault (Sessão 17/18) — não assumidos.
_LISTA_COMPRAS_PATH = "02 - Áreas/Finanças/Lista de Compras.md"
_DATAS_IMPORTANTES_PATH = "02 - Áreas/Pessoas/Datas Importantes.md"

# --- Agenda da Bia (V5) --------------------------------------------------
# Interação acontece no chat privado dela (BIA_JID) — sem gatilho de grupo.

# "Fulana, dia 03, quarta, 15h" — gramática fixa: nome, "dia N[/mês]",
# dia-da-semana opcional (só validação cruzada), hora (h/hmin/hh:mm).
_ADD_AGENDA_RE = re.compile(
    r"^(?P<nome>[^,]+),\s*"
    r"dia\s*0?(?P<dia>\d{1,2})(?:\s*/\s*0?(?P<mes>\d{1,2}))?\s*,?\s*"
    r"(?:(?P<diasemana>segunda(?:-feira)?|ter[cç]a(?:-feira)?|quarta(?:-feira)?|"
    r"quinta(?:-feira)?|sexta(?:-feira)?|s[áa]bado|domingo)\s*,?\s*)?"
    r"(?:[àa]s?\s*)?(?P<hora>\d{1,2})(?:[:h](?P<minuto>\d{2}))?\s*h?\b",
    re.IGNORECASE,
)

_DIA_SEMANA_NORM = {
    "segunda": 0, "segunda-feira": 0,
    "terca": 1, "terça": 1, "terca-feira": 1, "terça-feira": 1,
    "quarta": 2, "quarta-feira": 2,
    "quinta": 3, "quinta-feira": 3,
    "sexta": 4, "sexta-feira": 4,
    "sabado": 5, "sábado": 5,
    "domingo": 6,
}
_DIA_SEMANA_PT = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]


def _extrair_add_agenda(text: str, hoje: datetime.date) -> dict | None:
    m = _ADD_AGENDA_RE.match(text.strip())
    if not m:
        return None
    nome = m.group("nome").strip()
    if not nome:
        return None
    dia = int(m.group("dia"))
    mes = int(m.group("mes")) if m.group("mes") else hoje.month
    try:
        data = datetime.date(hoje.year, mes, dia)
    except ValueError:
        return None
    hora = int(m.group("hora"))
    minuto = int(m.group("minuto") or 0)
    if not (0 <= hora <= 23 and 0 <= minuto <= 59):
        return None

    diasemana_falada = m.group("diasemana")
    mismatch = False
    if diasemana_falada:
        esperado = _DIA_SEMANA_NORM.get(diasemana_falada.strip().lower())
        if esperado is not None and esperado != data.weekday():
            mismatch = True

    return {
        "nome": nome, "data": data, "hora": f"{hora:02d}:{minuto:02d}",
        "diasemana_falada": diasemana_falada, "mismatch": mismatch,
    }


# "cancela Fulana", "cancela o horário da Fulana dia 05", "cancela Fulana amanhã"
_CANCEL_AGENDA_RE = re.compile(
    r"\bcancela(?:r)?\b\s+(?:o\s+|a\s+)?(?:hor[áa]rio\s+(?:d[ae]\s+)?)?"
    r"(?P<nome>.+?)"
    r"(?:\s+dia\s*0?(?P<dia>\d{1,2})(?:\s*/\s*0?(?P<mes>\d{1,2}))?"
    r"|\s+(?P<rel>hoje|amanh[ãa]))?$",
    re.IGNORECASE,
)


def _extrair_cancel_agenda(text: str, hoje: datetime.date) -> dict | None:
    m = _CANCEL_AGENDA_RE.search(text.strip())
    if not m:
        return None
    nome = m.group("nome").strip().rstrip("?!.,").strip()
    if not nome:
        return None
    data: datetime.date | None = None
    if m.group("dia"):
        mes = int(m.group("mes")) if m.group("mes") else hoje.month
        try:
            data = datetime.date(hoje.year, mes, int(m.group("dia")))
        except ValueError:
            data = None
    elif m.group("rel"):
        rel = m.group("rel").lower()
        data = hoje + datetime.timedelta(days=1) if rel.startswith("amanh") else hoje
    return {"nome": nome, "data": data}


# "quais horários tenho amanhã?", "agenda de hoje", "clientes do dia 05"
_QUERY_AGENDA_RE = re.compile(
    r"\b(hor[áa]rios?|agenda|clientes?)\b.*?\b(hoje|amanh[ãa]|dia\s*0?\d{1,2})\b|"
    r"\b(hoje|amanh[ãa])\b.*?\b(hor[áa]rios?|agenda|clientes?)\b",
    re.IGNORECASE,
)


def _extrair_query_agenda_data(text: str, hoje: datetime.date) -> datetime.date | None:
    if not _QUERY_AGENDA_RE.search(text):
        return None
    if re.search(r"amanh[ãa]", text, re.IGNORECASE):
        return hoje + datetime.timedelta(days=1)
    if re.search(r"\bhoje\b", text, re.IGNORECASE):
        return hoje
    m = re.search(r"\bdia\s*0?(\d{1,2})\b", text, re.IGNORECASE)
    if m:
        try:
            return datetime.date(hoje.year, hoje.month, int(m.group(1)))
        except ValueError:
            return None
    return hoje


_SHOPPING_LIST_RE = re.compile(r"\blista\s+de\s+compras?\b", re.IGNORECASE)
_IMPORTANT_DATES_RE = re.compile(
    r"\bdatas?\s+importantes?\b|\bpr[óo]xim[ao]s?\s+datas?\b", re.IGNORECASE
)

# "últimos N meses" não é coberto por `expenses._parse_periodo()` (que cobre
# hoje/ontem/semana/mês passado/mês por nome) — tratado em `_meses_do_periodo`.
_ULTIMOS_MESES_RE = re.compile(r"[úu]ltim[oa]s?\s+(\d+)\s+m[êe]s(?:es)?", re.IGNORECASE)

_GASTOS_HEADER_RE = re.compile(r"^\|\s*data\s*\|", re.IGNORECASE)
_BR_DATA_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2,4})$")
_ISO_DATA_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _format_bills_template(result: BillsResult) -> str:
    """Reconstrói exatamente as 4 strings que `answer_bills_query()` devolvia
    antes do contrato mudar pra `BillsResult` (D-09) — caminho sem Kimi."""
    if result.kind == "specific":
        c = result.conta_alvo
        return f"{c['descricao']} ({_brl(c['valor'])}) {_STATUS_ROTULO[c['status']]}."

    if result.kind == "due_today":
        do_dia = [c for c in result.contas if c["dia_vencimento"] == result.hoje]
        if not do_dia:
            return f"Nenhuma conta fixa vence hoje (dia {result.hoje})."
        linhas = [f"Contas que vencem hoje (dia {result.hoje}):"]
        linhas += [f"- {c['descricao']}: {_brl(c['valor'])}" for c in do_dia]
        return "\n".join(linhas)

    if result.kind == "pending":
        if not result.pendentes:
            return "Nenhuma conta fixa pendente esse mês — tudo pago 🎉"
        linhas = [f"Contas fixas pendentes ({len(result.pendentes)}):"]
        linhas += [
            f"- {c['descricao']}: {_brl(c['valor'])}"
            + (f" (dia {c['dia_vencimento']})" if c["dia_vencimento"] else "")
            for c in result.pendentes
        ]
        linhas.append(f"Total pendente: {_brl(result.total_pendente)}")
        return "\n".join(linhas)

    # kind == "total"
    return (
        f"Total de contas fixas do mês: {_brl(result.total_geral)} "
        f"({len(result.contas)} contas)."
    )


async def _format_bills_via_kimi(text: str, result: BillsResult) -> str:
    """Passa o `BillsResult` cru pro Kimi formatar — usado quando a pergunta
    tem modificador de detalhe (`_BILLS_DETAIL_MODIFIER_RE`). Levanta
    `LLMError` em falha; quem chama decide o fallback."""
    contas_str = "\n".join(
        f"- {c['descricao']}: {_brl(c['valor'])}, vencimento dia "
        f"{c['dia_vencimento']}, status: {c['status']}"
        for c in result.contas
    )
    prompt = (
        f"Pergunta do usuário sobre as contas fixas de {result.mes_label}: "
        f"{text!r}\n\n"
        f"Dados de todas as contas do mês:\n{contas_str}\n\n"
        f"Total geral: {_brl(result.total_geral)}. "
        f"Total pendente: {_brl(result.total_pendente)}.\n\n"
        "Responda a pergunta usando só esses dados, em português, direto, "
        "formato WhatsApp. Se a pergunta pedir um filtro (ex.: só as pagas, "
        "só as pendentes), aplique o filtro você mesmo sobre os dados acima."
    )
    provider = get_provider("kimi")
    result_llm = await provider.generate(
        [{"role": "user", "content": prompt}], max_tokens=400, timeout=30.0,
    )
    return result_llm.text


def _gastos_mes_path(ano: int, mes: int) -> str:
    return f"{_FINANCAS_FOLDER}/Gastos - {ano:04d}-{mes:02d}.md"


async def _ler_gastos_mes(ano: int, mes: int) -> str | None:
    """Conteúdo de `Gastos - YYYY-MM.md`, lido **do disco** — mesmo padrão de
    `bills.py` com `Contas - YYYY-MM.md` (Sessão 20).

    Escrita continua indo por MCP (`vault_writer`), pra que o Obsidian saiba
    da mudança; leitura vem do disco porque:
    - não depende do índice do plugin (que na Sessão 19 não enxergou arquivos
      escritos direto no disco e quase causou perda de dados);
    - é testável — a fixture do benchmark pode apontar `vault_path` pra um
      diretório temporário, o que é impossível com o MCP real;
    - é mais rápido: sem round-trip de rede pra um caminho já conhecido.
    """
    if GASTOS_ROOT_OVERRIDE:
        path = pathlib.Path(GASTOS_ROOT_OVERRIDE) / _gastos_mes_path(ano, mes)
    else:
        path = safe_vault_path(_gastos_mes_path(ano, mes))
    if path is None or not path.is_file():
        logger.info("Agent Vida: %s não encontrado", _gastos_mes_path(ano, mes))
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.warning("Agent Vida: falha ao ler gastos de %04d-%02d (%r)", ano, mes, exc)
        return None


def _periodo_da_pergunta(
    hoje: datetime.date, texto: str
) -> tuple[list[tuple[int, int]], str, str, str]:
    """`(meses, data_ini_iso, data_fim_iso_exclusiva, rótulo)` da pergunta.

    Os meses dizem QUAIS `Gastos - YYYY-MM.md` ler; o intervalo de datas
    filtra as linhas dentro deles (uma pergunta de semana lê o mês inteiro
    mas só soma os dias certos). "últimos N meses" tem parsing próprio (não
    coberto por `expenses._parse_periodo`); o resto reusa esse parser, que
    também devolve o rótulo usado na resposta ("neste mês", "em agosto/2026").
    """
    m = _ULTIMOS_MESES_RE.search(texto)
    if m:
        n = max(1, int(m.group(1)))
        meses = []
        ano, mes = hoje.year, hoje.month
        for _ in range(n):
            meses.append((ano, mes))
            mes -= 1
            if mes == 0:
                mes, ano = 12, ano - 1
        meses.reverse()
        data_ini = datetime.date(meses[0][0], meses[0][1], 1).isoformat()
        data_fim = (hoje + datetime.timedelta(days=1)).isoformat()
        return meses, data_ini, data_fim, f"nos últimos {n} meses"

    data_ini_iso, data_fim_iso, rotulo = _parse_periodo(texto, hoje)
    ini = datetime.date.fromisoformat(data_ini_iso)
    fim = datetime.date.fromisoformat(data_fim_iso)  # exclusiva

    meses = []
    cursor = datetime.date(ini.year, ini.month, 1)
    while cursor < fim:
        meses.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = datetime.date(cursor.year + 1, 1, 1)
        else:
            cursor = datetime.date(cursor.year, cursor.month + 1, 1)
    return meses, data_ini_iso, data_fim_iso, rotulo


def _parse_gastos_md(conteudo: str) -> list[dict]:
    """Parseia a tabela `Data | Pessoa | Categoria | Descrição | Valor` de uma
    nota `Gastos - YYYY-MM.md`. Data em `dd-mm-aa` (Sessão 20). Tolerante:
    linha sem valor numérico é ignorada, nunca levanta exceção por formato
    inesperado (mesmo espírito de `bills.parse_contas_fixas`)."""
    registros: list[dict] = []
    in_table = False
    for ln in conteudo.splitlines():
        s = ln.strip()
        if not in_table:
            if _GASTOS_HEADER_RE.match(s):
                in_table = True
            continue
        if not s.startswith("|"):
            break
        cols = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cols):
            continue
        if len(cols) < 5:
            continue
        data, pessoa, categoria, descricao, valor_raw = cols[:5]
        try:
            valor = float(valor_raw)
        except ValueError:
            continue
        registros.append({
            "data": data, "pessoa": pessoa, "categoria": categoria or "Outros",
            "descricao": descricao, "valor": valor,
        })
    return registros


def _data_iso_registro(data_br: str) -> str | None:
    """`dd-mm-aa` (formato da coluna Data, Sessão 20) → ISO, ou `None` se a
    linha não tiver data reconhecível. Também aceita `dd-mm-aaaa` e ISO puro,
    porque notas antigas migradas podem ter sobrado em outro formato."""
    s = data_br.strip()
    if _ISO_DATA_RE.match(s):
        return s
    m = _BR_DATA_RE.match(s)
    if not m:
        return None
    dia, mes, ano = m.group(1), m.group(2), m.group(3)
    if len(ano) == 2:
        ano = f"20{ano}"
    try:
        return datetime.date(int(ano), int(mes), int(dia)).isoformat()
    except ValueError:
        return None


def _pessoa_citada(texto: str, registros: list[dict]) -> str | None:
    """Nome de pessoa citado na pergunta, procurado entre quem aparece na
    coluna Pessoa das notas lidas mais os nomes configurados.

    Substitui `expenses._named_person()`, que consultava a tabela SQL
    `expenses` — sem sentido agora que a fonte é o vault (Sessão 20)."""
    nomes = {r["pessoa"] for r in registros if r["pessoa"]}
    nomes |= set(settings.known_names.values())
    for n in (settings.owner_name, settings.partner_name):
        if n:
            nomes.add(n)
    low = texto.lower()
    for nome in sorted(nomes, key=len, reverse=True):
        if re.search(rf"\b{re.escape(nome.lower())}\b", low):
            return nome
    return None


def _formata_expenses(
    registros: list[dict], message: str, autor: str | None, rotulo: str
) -> str | None:
    """Agrega e formata os lançamentos — 100% determinístico (Sessão 20).

    Mesmos templates do `expenses.query_expenses()` original, que o benchmark
    valida por `reply_equals`. O Kimi formatava isso livremente desde a Fase B
    e além de quebrar todos os `reply_equals` errou soma (cas-03: R$ 2.265,00
    para um total de R$ 1.265,00) — aritmética de LLM não é aceitável aqui.

    Devolve `None` quando não sobra lançamento no filtro, pra que o chamador
    decida entre mensagem de "nada registrado" e fallback."""
    categoria = _match_categoria(message)
    if categoria:
        registros = [
            r for r in registros if r["categoria"].lower() == categoria.lower()
        ]
    if not registros:
        return None

    cab = rotulo[0].upper() + rotulo[1:]

    # --- "quanto cada um gastou?" → quebra por pessoa ----------------------
    if _POR_PESSOA_RE.search(message):
        por_pessoa: dict[str, list[float]] = {}
        for r in registros:
            por_pessoa.setdefault(r["pessoa"] or "sem nome", []).append(r["valor"])
        ordenado = sorted(
            por_pessoa.items(), key=lambda kv: sum(kv[1]), reverse=True
        )
        alvo = f" em {categoria}" if categoria else ""
        linhas = [f"{cab}{alvo}, por pessoa:"]
        linhas += [
            f"- {pessoa}: {_brl(sum(vs))} ({len(vs)} lanç.)"
            for pessoa, vs in ordenado
        ]
        linhas.append(f"Total: {_brl(sum(r['valor'] for r in registros))}")
        return "\n".join(linhas)

    # --- filtro por pessoa: nome citado > "eu gastei" > casal --------------
    citada = _pessoa_citada(message, registros)
    pessoal = bool(autor and _PESSOAL_RE.search(message) and not _CASAL_RE.search(message))
    if citada:
        registros = [r for r in registros if r["pessoa"] == citada]
        sujeito = f"{citada} gastou"
    elif pessoal:
        registros = [r for r in registros if r["pessoa"] == autor]
        sujeito = "você gastou"
    else:
        sujeito = "vocês gastaram"
    if not registros:
        return None

    total = sum(r["valor"] for r in registros)
    n = len(registros)
    lanc = "lançamento" if n == 1 else "lançamentos"
    if categoria:
        return f"{cab}, {sujeito} {_brl(total)} em {categoria} ({n} {lanc})."

    linha = f"{cab}, {sujeito} {_brl(total)} ({n} {lanc})."
    por_categoria: dict[str, float] = {}
    for r in registros:
        por_categoria[r["categoria"]] = por_categoria.get(r["categoria"], 0.0) + r["valor"]
    if len(por_categoria) > 1:
        itens = "\n".join(
            f"- {cat}: {_brl(v)}"
            for cat, v in sorted(por_categoria.items(), key=lambda kv: kv[1], reverse=True)
        )
        return f"{linha}\n{itens}"
    return linha


async def _kimi_texto(prompt: str) -> str:
    """Chamada Kimi simples (sem contexto estrutural/histórico) — usada por
    expenses e balance, que montam o próprio prompt a partir da busca
    semântica em vez do Bloco 3 genérico."""
    provider = get_provider("kimi")
    result = await provider.generate(
        [{"role": "user", "content": prompt}], max_tokens=400, timeout=30.0,
    )
    return result.text


# --- ReAct (D-10) --------------------------------------------------------
#
# Causa raiz do vault parado por 19 dias: os passos 1-11 de `try_fast_path`
# são regex, e linguagem natural variável ("faculdade paga", "TIM ✅", "já
# quitei o aluguel do salão") nunca bate — cai direto pro RAG genérico, que
# só LÊ o vault, nunca escreve. `_try_react()` entra DEPOIS de todo fast-path
# falhar (só no caminho lento, com orquestrador já rodado — nunca em
# `try_fast_path()`, que `pipeline.py` chama antes de pagar latência de LLM,
# D-05) e ANTES do fallback RAG: o Kimi vira o cérebro de decisão, recebe a
# mensagem + contexto real do vault (contas do mês) e devolve UMA ação
# estruturada em JSON. O agente executa a ferramenta e devolve o resultado
# pro Kimi, que decide continuar (nova ação) ou encerrar (`respond`).
#
# Ferramentas de dado puro (`_try_balance`, `_try_expenses`,
# `_try_query_agenda`, `_try_shopping_list`, `_try_important_dates`) e as
# ferramentas em si (`bills.py`, `vault_writer.py`) não mudam — só ganham
# mais um chamador.

_REACT_MAX_STEPS = 3

# O Kimi às vezes embrulha o JSON em ```json ... ``` mesmo pedindo JSON puro.
_CERCA_CODIGO_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _remove_cerca_codigo(texto: str) -> str:
    texto = texto.strip()
    m = _CERCA_CODIGO_RE.match(texto)
    return m.group(1) if m else texto


_REACT_ACOES = (
    "mark_paid", "remove_bill", "add_bill", "add_expense",
    "write_note", "read_vault", "respond",
)

_REACT_SYSTEM_PROMPT = (
    "Você é o Yaannk, assistente do casal, decidindo qual ação tomar diante "
    "de uma mensagem do WhatsApp que NÃO bateu com nenhum comando fixo.\n\n"
    "Responda APENAS em JSON, sem texto fora dele, no formato:\n"
    '{{"action": "<ação>", "params": {{...}}}}\n\n'
    "Ações possíveis:\n"
    '- mark_paid: {{"conta": "<nome como aparece no contexto>"}} — confirma '
    "pagamento de uma conta fixa já existente (ex.: \"faculdade paga\", "
    "\"TIM ✅\", \"já quitei o aluguel\").\n"
    '- remove_bill: {{"conta": "<nome>"}} — remove uma conta fixa.\n'
    '- add_bill: {{"nome": "<nome>", "valor": <número>, "dia": <1-31>}} — '
    "cria conta fixa nova.\n"
    '- add_expense: {{"valor": <número>, "descricao": "<texto>", '
    '"categoria": "<categoria ou vazio>"}} — registra gasto variável novo.\n'
    '- write_note: {{"path": "<caminho no vault>", "content": "<texto>"}} — '
    "só quando nenhuma outra ação serve.\n"
    '- read_vault: {{"query": "<o que procurar>"}} — busca no vault antes '
    "de decidir (ex.: pra conferir o nome exato de uma conta).\n"
    '- respond: {{"text": "<resposta final ao usuário, em português, '
    'formato WhatsApp>"}} — encerra sem executar nada (dúvida, saudação, '
    "pedido que nenhuma ação cobre).\n\n"
    "Regras:\n"
    "- \"conta\" em mark_paid/remove_bill deve ser o nome EXATO como aparece "
    "no contexto abaixo (não invente contas fora dessa lista).\n"
    "- Se a conta citada não aparecer no contexto, use `respond` avisando "
    "que não achou, em vez de chutar `mark_paid`.\n"
    "- Nunca invente valores: se a mensagem não tiver valor numérico e a "
    "ação exigir um, use `respond` pedindo o valor.\n"
    "- Quem mandou a mensagem: {autor}. Se for chamar a pessoa pelo nome, use "
    "exatamente esse; se for \"desconhecido\", não use nome nenhum.\n\n"
    "Contexto (contas fixas do mês corrente):\n{contexto}"
)


async def _react_contexto_contas(ano: int, mes: int) -> str:
    contas = get_contas_do_mes(ano, mes)
    if not contas:
        return "(nenhuma conta fixa cadastrada este mês)"
    # Ordenado por vencimento (sem dia vai pro fim) — o Kimi tende a repetir a
    # ordem do contexto quando lista as contas.
    contas = sorted(
        contas, key=lambda c: (c["dia_vencimento"] is None, c["dia_vencimento"] or 0)
    )
    linhas = []
    for c in contas:
        vence = f"vence dia {c['dia_vencimento']}" if c["dia_vencimento"] else "sem vencimento"
        linhas.append(
            f"- {c['descricao']} ({_brl(c['valor'])}, {vence}) — {_STATUS_ROTULO[c['status']]}"
        )
    return "\n".join(linhas)


async def _react_write_note(path: str, content: str) -> dict:
    """Anexa `content` no fim da nota em `path` (cria a nota se ela ainda não
    existir) — via MCP, mesmo transporte de `vault_writer._mcp_call`, sem
    reusar `save_to_vault` porque essa função só cobre os 4 `tipo` fixos com
    caminho pré-definido (D-10 pede caminho livre)."""
    try:
        try:
            result = await _mcp_call("get_vault_file", {"path": path, "format": "text"})
            conteudo = result["content"][0]["text"]
        except VaultWriteError:
            conteudo = ""
        novo = conteudo.rstrip() + ("\n\n" if conteudo.strip() else "") + content.strip() + "\n"
        await _mcp_call("create_vault_file", {"path": path, "content": novo})
        return {"success": True, "message": f"nota atualizada em {path}"}
    except VaultWriteError as exc:
        return {"success": False, "message": f"falha ao gravar em {path}: {exc}"}


async def _react_executa(
    action: str, params: dict, mes: str, autor: str | None
) -> str | None:
    """Executa a ação decidida pelo Kimi e devolve a observação em texto pro
    próximo turno do loop. `None` = ação inválida/sem params suficientes →
    quem chama aborta o ReAct e cai pro RAG."""
    if action == "mark_paid":
        conta = str(params.get("conta", "")).strip()
        if not conta:
            return None
        resultado = await mark_bill_paid(conta, mes)
        return json.dumps(resultado, ensure_ascii=False)

    if action == "remove_bill":
        conta = str(params.get("conta", "")).strip()
        if not conta:
            return None
        resultado = await remove_bill(conta, mes)
        return json.dumps(resultado, ensure_ascii=False)

    if action == "add_bill":
        nome = str(params.get("nome", "")).strip()
        try:
            valor = float(params["valor"])
            dia = int(params["dia"])
        except (KeyError, TypeError, ValueError):
            return None
        if not nome or not (1 <= dia <= 31):
            return None
        resultado = await add_bill(nome, valor, dia, mes)
        return json.dumps(resultado, ensure_ascii=False)

    if action == "add_expense":
        try:
            valor = float(params["valor"])
        except (KeyError, TypeError, ValueError):
            return None
        descricao = str(params.get("descricao", "")).strip()
        categoria = str(params.get("categoria", "")).strip()
        texto = f"gastei {valor:.2f} em {descricao or 'algo'}"
        if categoria:
            texto += f" ({categoria})"
        try:
            confirmacao = await save_to_vault("gasto", texto, autor or "?")
        except VaultWriteError as exc:
            return json.dumps({"success": False, "message": str(exc)}, ensure_ascii=False)
        return json.dumps({"success": True, "message": confirmacao}, ensure_ascii=False)

    if action == "write_note":
        path = str(params.get("path", "")).strip()
        content = str(params.get("content", "")).strip()
        if not path or not content:
            return None
        resultado = await _react_write_note(path, content)
        return json.dumps(resultado, ensure_ascii=False)

    if action == "read_vault":
        query = str(params.get("query", "")).strip()
        if not query:
            return None
        contexto = await search_vault_hybrid(query, priority_folder=_FINANCAS_FOLDER)
        return contexto or "(nada encontrado no vault para essa busca)"

    return None


async def _react_decide(
    message: str, contexto_contas: str, autor: str | None
) -> AgentResponse | None:
    """Loop ReAct: pergunta ao Kimi, executa a ação, devolve a observação, até
    `respond` ou `_REACT_MAX_STEPS`. Qualquer JSON inválido/ação desconhecida
    aborta o loop e devolve `None` (cai pro RAG — nunca trava a resposta)."""
    started = time.monotonic()
    hoje = datetime.date.today()
    mes = f"{hoje.year:04d}-{hoje.month:02d}"

    logger.info(
        "Agent Vida ReAct: iniciando loop (mês=%s, %d linha(s) de contexto)",
        mes, contexto_contas.count("\n") + 1,
    )

    messages: list[Message] = [
        {"role": "system", "content": _REACT_SYSTEM_PROMPT.format(
            contexto=contexto_contas, autor=autor or "desconhecido"
        )},
        {"role": "user", "content": message},
    ]

    for passo in range(_REACT_MAX_STEPS):
        chamada_started = time.monotonic()
        try:
            provider = get_provider("kimi")
            result = await provider.generate(messages, max_tokens=300, timeout=45.0)
        except LLMError as exc:
            logger.warning(
                "Agent Vida ReAct: Kimi falhou no passo %d após %dms (%r) — cai pro RAG",
                passo + 1, int((time.monotonic() - chamada_started) * 1000), exc,
            )
            return None

        choices = result.raw.get("choices") or [{}]
        logger.info(
            "Agent Vida ReAct: Kimi respondeu no passo %d em %dms "
            "(provider=%dms, finish_reason=%s, tokens_saida=%s, %d chars)",
            passo + 1, int((time.monotonic() - chamada_started) * 1000),
            result.latency_ms or 0, choices[0].get("finish_reason"),
            result.output_tokens, len(result.text),
        )

        try:
            decisao = json.loads(_remove_cerca_codigo(result.text))
        except json.JSONDecodeError as exc:
            logger.warning(
                "Agent Vida ReAct: JSON inválido do Kimi (%s, finish_reason=%s) "
                "raw=%r — cai pro RAG",
                exc, choices[0].get("finish_reason"), result.text[:500],
            )
            return None

        try:
            action = str(decisao["action"])
            params = decisao.get("params") or {}
        except (KeyError, TypeError, AttributeError) as exc:
            logger.warning(
                "Agent Vida ReAct: JSON sem `action`/`params` válidos (%r) raw=%r — cai pro RAG",
                exc, result.text[:500],
            )
            return None

        logger.info(
            "Agent Vida ReAct: passo %d/%d ação=%r params=%r",
            passo + 1, _REACT_MAX_STEPS, action, params,
        )

        if action not in _REACT_ACOES:
            logger.warning("Agent Vida ReAct: ação desconhecida %r — cai pro RAG", action)
            return None

        if action == "respond":
            texto = str(params.get("text", "")).strip()
            if not texto:
                logger.warning("Agent Vida ReAct: `respond` sem texto — cai pro RAG")
                return None
            return AgentResponse(
                text=texto, source="llm", agent="vida",
                latency_ms=int((time.monotonic() - started) * 1000),
            )

        observacao = await _react_executa(action, params, mes, autor)
        if observacao is None:
            logger.warning(
                "Agent Vida ReAct: ação %r com params insuficientes (%r) — cai pro RAG",
                action, params,
            )
            return None

        messages.append({"role": "assistant", "content": json.dumps(decisao, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Resultado da ação: {observacao}"})

    logger.info("Agent Vida ReAct: excedeu %d passos sem `respond` — cai pro RAG", _REACT_MAX_STEPS)
    return None


class AgentVida(AgentExecutor):
    async def handle(
        self,
        message: str,
        routing: OrchestratorDecision,
        history: list[Message],
        sender: str,
        *,
        conv_key: str,
    ) -> AgentResponse:
        """Entrada completa da ABC — fast-paths + fallback RAG. Usada no
        caminho lento do pipeline, depois que o orquestrador já rodou
        (`routing.agent == "vida"`). No caminho rápido, quem chama é
        `try_fast_path()` diretamente, sem esperar o orquestrador (D-05)."""
        response = await self.try_fast_path(message, conv_key, sender)
        if response is not None:
            return response
        response = await self._try_react(message, sender)
        if response is not None:
            return response
        return await self._fallback_rag(message, conv_key, history)

    async def _try_react(
        self, message: str, autor: str | None
    ) -> AgentResponse | None:
        """ReAct (D-10) — só no caminho lento (depois do orquestrador), nunca
        em `try_fast_path()`. Ver docstring de `_react_decide` para o loop
        completo; aqui só monta o contexto de contas do mês e delega."""
        logger.info("Agent Vida: _try_react chamado (mensagem=%r)", message[:120])
        hoje = datetime.date.today()
        contexto_contas = await _react_contexto_contas(hoje.year, hoje.month)
        resultado = await _react_decide(message, contexto_contas, autor)
        if resultado is None:
            logger.info("Agent Vida: _try_react não resolveu — cai pro RAG")
        else:
            logger.info(
                "Agent Vida: _try_react resolveu via ReAct (%d chars)", len(resultado.text)
            )
        return resultado

    async def try_fast_path(
        self, message: str, conv_key: str, autor: str | None = None
    ) -> AgentResponse | None:
        """Só os passos determinísticos (1-5) — sem RAG genérico, sem
        orquestrador. `pipeline.py` chama isso ANTES de instanciar o
        orquestrador, pra não pagar latência do Kimi em mensagens que já
        respondem por parser/leitura direta (mesma lógica de
        `_resolve_skill_name`, D-05).

        Sessão 21: todos os passos voltaram a ser 100% sem LLM no caminho
        normal. `_try_expenses()` só chama o Kimi quando nenhuma nota do
        período existe (fallback de busca semântica), e `_try_bills()` só
        quando a pergunta pede mais do que os templates cobrem (D-09)."""
        response = await self._try_balance(message)
        if response is not None:
            return response

        response = await self._try_expenses(message, autor)
        if response is not None:
            return response

        response = await self._try_bills(message)
        if response is not None:
            return response

        response = await self._try_cancel_agenda(message)
        if response is not None:
            return response

        response = await self._try_add_agenda(message)
        if response is not None:
            return response

        response = self._try_query_agenda(message)
        if response is not None:
            return response

        response = self._try_shopping_list(message)
        if response is not None:
            return response

        response = self._try_important_dates(message)
        if response is not None:
            return response

        return None

    # --- passos individuais -------------------------------------------------

    async def _try_balance(self, message: str) -> AgentResponse | None:
        """Balanço do mês corrente, 100% determinístico (Sessão 20 — revertido
        da busca semântica, que errava qual mês tinha dados: no benchmark ela
        respondeu "não encontrei lançamento em agosto" com agosto tendo 7).

        Lê `Gastos - YYYY-MM.md` por path calculado, parseia a tabela e soma
        em Python — sem Kimi, sem depender de recall de busca. NÃO inclui
        contas fixas (comportamento de sempre do `balance_report` original)."""
        if not BALANCE_COMMAND_RE.match(message):
            return None
        started = time.monotonic()
        hoje = datetime.date.today()
        titulo = f"{_MES_NOME[hoje.month]}/{hoje.year}"

        conteudo = await _ler_gastos_mes(hoje.year, hoje.month)
        registros = _parse_gastos_md(conteudo) if conteudo else []

        if not registros:
            text = f"📊 Balanço de {titulo}\nNenhum gasto variável registrado ainda."
        else:
            total = sum(r["valor"] for r in registros)
            por_categoria: dict[str, float] = {}
            for r in registros:
                por_categoria[r["categoria"]] = por_categoria.get(r["categoria"], 0.0) + r["valor"]
            breakdown = sorted(por_categoria.items(), key=lambda kv: kv[1], reverse=True)
            lanc = "lançamento" if len(registros) == 1 else "lançamentos"
            linhas = [
                f"📊 Balanço de {titulo}",
                f"Total: {_brl(total)} ({len(registros)} {lanc})",
            ]
            linhas += [f"- {cat}: {_brl(v)}" for cat, v in breakdown]
            text = "\n".join(linhas)

        latency_ms = int((time.monotonic() - started) * 1000)
        return AgentResponse(text=text, source="fast_path", agent="vida", latency_ms=latency_ms)

    async def _try_expenses(
        self, message: str, autor: str | None
    ) -> AgentResponse | None:
        """Consulta de gastos variáveis — determinística de ponta a ponta
        (Sessão 20, revisão 3).

        Ordem: (1) resolve período por cálculo de calendário, (2) lê os
        `Gastos - YYYY-MM.md` do período por path, (3) parseia, filtra e soma
        em Python, com os mesmos templates de resposta do
        `expenses.query_expenses()` original.

        O Kimi sobrou só no fallback de quando NENHUM arquivo do período
        existe — aí não há tabela pra somar, só trechos de busca semântica.
        No caminho normal ele não entra: na Fase B ele formatava a resposta e
        além de quebrar todo `reply_equals` do benchmark errou soma (cas-03).
        """
        if len(decompose_query(message)) != 1:
            return None
        if not FINANCIAL_QUERY_STRICT_RE.search(message):
            return None

        started = time.monotonic()
        hoje = datetime.date.today()
        meses, data_ini, data_fim, rotulo = _periodo_da_pergunta(hoje, message)

        registros: list[dict] = []
        blocos: list[str] = []
        for ano, mes in meses:
            conteudo = await _ler_gastos_mes(ano, mes)
            if not conteudo:
                continue
            blocos.append(f"### Gastos de {_MES_NOME[mes]}/{ano}\n{conteudo}")
            for r in _parse_gastos_md(conteudo):
                iso = _data_iso_registro(r["data"])
                if iso is None or not (data_ini <= iso < data_fim):
                    continue
                registros.append(r)

        def _resposta(text: str) -> AgentResponse:
            return AgentResponse(
                text=text, source="fast_path", agent="vida",
                latency_ms=int((time.monotonic() - started) * 1000),
            )

        if blocos:
            logger.info(
                "Agent Vida: gastos via path determinístico (%d bloco(s), "
                "%d lançamento(s) no período)", len(blocos), len(registros),
            )
            text = _formata_expenses(registros, message, autor, rotulo)
            if text is not None:
                return _resposta(text)
            cab = rotulo[0].upper() + rotulo[1:]
            return _resposta(f"{cab}, nenhum gasto registrado.")

        # Fallback: nenhum arquivo do período existe — talvez a pergunta não
        # fosse sobre um mês específico. Aí sim busca semântica + Kimi, que é
        # o único caminho sem tabela pra somar.
        contexto = await search_vault_hybrid(message, priority_folder=_FINANCAS_FOLDER)
        if not contexto:
            ano_ref, mes_ref = meses[-1] if meses else (hoje.year, hoje.month)
            return _resposta(
                f"Nenhum gasto registrado em {_MES_NOME[mes_ref]}/{ano_ref}."
            )

        logger.info("Agent Vida: gastos via busca semântica (fallback)")
        prompt = (
            f"Pergunta sobre gastos: {message!r}\n"
            f"Quem perguntou: {autor or 'desconhecido'}\n\n"
            "Trechos do vault sobre gastos (colunas Data | Pessoa | "
            f"Categoria | Descrição | Valor; data em dd-mm-aa):\n\n{contexto}\n\n"
            "Responda usando SÓ esses dados. Regras da resposta:\n"
            "- Português, formato WhatsApp, NO MÁXIMO 5 linhas.\n"
            "- SEM tabela markdown, SEM emoji decorativo, SEM pergunta de "
            "follow-up no final.\n"
            "- Se pedir filtro por pessoa, some só as linhas dessa pessoa "
            "(coluna Pessoa). Se pedir quebra por pessoa, agrupe por Pessoa. "
            "Se pedir total, some a coluna Valor.\n"
            "- Confira a soma com cuidado antes de responder."
        )
        try:
            text = await _kimi_texto(prompt)
        except LLMError as exc:
            logger.warning(
                "Agent Vida: Kimi falhou respondendo gastos (%r) — cai pro RAG", exc,
            )
            return None
        return _resposta(text)

    async def _try_bills(self, message: str) -> AgentResponse | None:
        """Sem mudança nesta revisão — path determinístico
        (`Contas - YYYY-MM.md`), continua igual ao que já estava validado."""
        if len(decompose_query(message)) != 1:
            return None
        started = time.monotonic()
        result = answer_bills_query(message)
        if result is None:
            return None

        if _BILLS_DETAIL_MODIFIER_RE.search(message):
            try:
                text = await _format_bills_via_kimi(message, result)
                source = "llm"
            except LLMError as exc:
                logger.warning(
                    "Agent Vida: Kimi falhou formatando bills detalhado (%r) "
                    "— caindo no template", exc,
                )
                text = _format_bills_template(result)
                source = "fast_path"
        else:
            text = _format_bills_template(result)
            source = "fast_path"

        latency_ms = int((time.monotonic() - started) * 1000)
        return AgentResponse(text=text, source=source, agent="vida", latency_ms=latency_ms)

    # --- Agenda da Bia (V5) --------------------------------------------------

    async def _try_add_agenda(self, message: str) -> AgentResponse | None:
        """"Fulana, dia 03, quarta, 15h" — registra cliente na agenda do mês
        (`agenda_bia.add_agendamento`). Conflito = mesmo dia+horário exato já
        ocupado por outra cliente não cancelada: não decide sozinho, só
        avisa. Dia da semana falado que não bate com a data calculada também
        não é decidido sozinho — pede confirmação em vez de gravar torto."""
        hoje = datetime.date.today()
        extraido = _extrair_add_agenda(message, hoje)
        if extraido is None:
            return None
        started = time.monotonic()

        if extraido["mismatch"]:
            dia_real = _DIA_SEMANA_PT[extraido["data"].weekday()]
            texto = (
                f"⚠️ Confere: {extraido['data'].strftime('%d/%m')} é {dia_real}, "
                f"não {extraido['diasemana_falada']}. Manda de novo com o dia "
                "certo se eu errei, ou confirma que é isso mesmo."
            )
            return AgentResponse(
                text=texto, source="fast_path", agent="vida",
                latency_ms=int((time.monotonic() - started) * 1000),
            )

        resultado = await add_agendamento(extraido["nome"], extraido["data"], extraido["hora"])
        prefixo = "✅" if resultado["success"] else "⚠️"
        texto = f"{prefixo} {resultado['message']}"
        return AgentResponse(
            text=texto, source="fast_path", agent="vida",
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    async def _try_cancel_agenda(self, message: str) -> AgentResponse | None:
        """"Cancela a Fulana" / "cancela Fulana dia 05" / "cancela Fulana
        amanhã" — marca o agendamento como cancelado (mantém a linha,
        histórico do mês) via `agenda_bia.cancel_agendamento`."""
        hoje = datetime.date.today()
        extraido = _extrair_cancel_agenda(message, hoje)
        if extraido is None:
            return None
        started = time.monotonic()
        resultado = await cancel_agendamento(extraido["nome"], extraido["data"])
        prefixo = "✅" if resultado["success"] else "⚠️"
        texto = f"{prefixo} {resultado['message']}"
        return AgentResponse(
            text=texto, source="fast_path", agent="vida",
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def _try_query_agenda(self, message: str) -> AgentResponse | None:
        """"Quais horários tenho amanhã?" / "agenda de hoje" — consulta
        determinística, sem LLM, mesmo espírito de `_try_bills`."""
        hoje = datetime.date.today()
        data = _extrair_query_agenda_data(message, hoje)
        if data is None:
            return None
        started = time.monotonic()
        entradas = get_agenda_dia(data)
        texto = format_agenda_dia_message(entradas, data)
        return AgentResponse(
            text=texto, source="fast_path", agent="vida",
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    def _try_shopping_list(self, message: str) -> AgentResponse | None:
        if not _SHOPPING_LIST_RE.search(message):
            return None
        return self._read_note_direct(_LISTA_COMPRAS_PATH)

    def _try_important_dates(self, message: str) -> AgentResponse | None:
        if not _IMPORTANT_DATES_RE.search(message):
            return None
        return self._read_note_direct(_DATAS_IMPORTANTES_PATH)

    def _read_note_direct(self, relpath: str) -> AgentResponse | None:
        """Leitura direta do disco (mesmo padrão de `bills.py`/`vault_search.py`
        — sem MCP). Um caminho já conhecido não precisa do round-trip de rede
        do MCP; ele existe pra busca semântica, não pra leitura de nota fixa."""
        started = time.monotonic()
        path = safe_vault_path(relpath)
        if path is None or not path.is_file():
            logger.info("Agent Vida: nota não encontrada em %s — cai pro RAG", relpath)
            return None
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("Agent Vida: falha ao ler %s (%r) — cai pro RAG", relpath, exc)
            return None
        text = _strip_frontmatter(content)
        latency_ms = int((time.monotonic() - started) * 1000)
        return AgentResponse(text=text, source="fast_path", agent="vida", latency_ms=latency_ms)

    async def _fallback_rag(
        self, message: str, conv_key: str, history: list[Message]
    ) -> AgentResponse:
        # Import tardio: `pipeline.py` importa este módulo (passo 6 da Fase B)
        # e este módulo precisa de `retrieve()`/`_block3()` de volta —
        # dependência circular se fosse import de topo (mesmo padrão já usado
        # em vault_search_semantic._catalog_snippet).
        from app.services.pipeline import _block3, retrieve
        from app.services.skills import get_skill

        logger.info("Agent Vida: _fallback_rag chamado (mensagem=%r)", message[:120])
        started = time.monotonic()
        r = await retrieve(message, conv_key=conv_key, skill_name="vida_casal")
        text, succeeded = await _block3(message, r, get_skill("vida_casal"), history)
        latency_ms = int((time.monotonic() - started) * 1000)
        return AgentResponse(
            text=text, source="rag", agent="vida", latency_ms=latency_ms,
            decision=r.decision, rag_files=r.rag_files,
            vault_context=r.vault_context, decomposed=r.decomposed,
            n_docs=r.n_docs, intent=r.decision.intent, succeeded=succeeded,
        )
