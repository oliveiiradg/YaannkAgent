"""Agente Yaannk (D-11) — Kimi com ferramentas nativas como caminho principal.

Com `AGENT_ENABLED=true`, substitui a cadeia detect_save_intent → fast-paths
→ orquestrador → ReAct. O modelo recebe a conversa recente (cada fala com o
nome de quem mandou), a data de hoje e dois grupos de ferramentas:

- **MCP do Obsidian** — buscar, ler, criar, editar, mover e apagar notas, com
  o schema vindo do próprio servidor (`tools/list`).
- **Domínio** — contas, gastos e agenda da Bia. Devolvem dado exato (totais
  já somados, contas ordenadas por vencimento) e escrevem no formato que os
  parsers de `bills.py`/`agent_vida.py` esperam. Conta e soma não ficam a
  cargo do LLM.

O loop roda até o modelo responder sem pedir ferramenta, até
`AGENT_MAX_STEPS` (a última chamada é forçada a responder) ou até
`AGENT_TIMEOUT_S`.
"""

import asyncio
import datetime
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.agents.agent_vida import _data_iso_registro, _ler_gastos_mes, _parse_gastos_md
from app.config import settings
from app.services import bills
from app.services.agenda_bia import add_agendamento, cancel_agendamento, get_agenda_dia
from app.services.conversation_store import get_recent_messages
from app.services.llm.base import LLMError
from app.services.llm.registry import get_provider
from app.services.vault_search import safe_vault_path
from app.services.vault_search_semantic import search_vault_hybrid
from app.services.vault_writer import (
    VaultWriteError,
    _append_sob_heading,
    _gastos_arquivo_novo,
    _gastos_path,
    _mcp_call,
    _mcp_rpc,
)

logger = logging.getLogger(__name__)

_MAX_TOOL_CHARS = 15_000
_MCP_TIMEOUT_S = 30.0
# Com menos que isso de tempo sobrando, para de explorar e responde com o que
# já juntou (sem raciocínio, que é o que mais demora).
_RESERVA_RESPOSTA_S = 40.0
_MAX_ARGS_LOG_CHARS = 200
_MAX_DESCRICAO_MCP_CHARS = 1000

_PERFIL_BIA_PATH = "02 - Áreas/Bia/Beatriz - Perfil.md"

_PEDIDO_RESPOSTA_FINAL = (
    "Responda agora ao usuário, em texto, com o que você já levantou. Não chame "
    "mais ferramentas; se faltou verificar algo, diga o que faltou."
)

# Tokens internos de tool call do Kimi que às vezes saem no `content` em vez de
# virar `tool_calls` (achado real 15/09: mensagem com
# `<|tool_calls_section_begin|>…` chegou crua no WhatsApp).
_BLOCO_FERRAMENTA_RE = re.compile(
    r"<\|tool_calls_section_begin\|>.*?(?:<\|tool_calls_section_end\|>|$)", re.DOTALL
)
_TOKEN_FERRAMENTA_RE = re.compile(r"<\|tool_calls?_[a-z_]+\|>")

_FALHA = "Tive um problema pra processar isso agora. Tenta de novo em instantes."
_FALHA_PARCIAL = (
    "Demorei demais e não consegui terminar. Parte do que você pediu pode já "
    "ter sido feita, confere antes de pedir de novo."
)

_DIAS_SEMANA = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]

# Ferramentas do MCP expostas ao modelo. Ficam de fora as de "nota ativa" (o
# Obsidian do notebook não tem ninguém olhando), canvas, comandos do
# Obsidian, `fetch` (internet) e `search_and_replace` (regex no vault inteiro).
_MCP_TOOLS = frozenset({
    "get_vault_overview", "list_vault_files", "get_recent_files",
    "search_vault_simple", "search_vault_smart", "execute_dataview_query",
    "get_vault_file", "get_vault_files", "get_vault_file_partial", "get_note_outline",
    "list_tags", "get_files_by_tag", "get_backlinks", "get_outgoing_links",
    "get_note_property", "set_note_property", "delete_note_property",
    "create_vault_file", "append_to_vault_file", "patch_vault_file",
    "rename_vault_file", "delete_vault_file", "create_vault_directory",
})
_MCP_WRITE_TOOLS = frozenset({
    "set_note_property", "delete_note_property", "create_vault_file",
    "append_to_vault_file", "patch_vault_file", "rename_vault_file",
    "delete_vault_file", "create_vault_directory",
})

_mcp_tools_cache: list[dict] | None = None


@dataclass
class AgentResult:
    text: str
    succeeded: bool
    steps: int = 0
    tools: list[str] = field(default_factory=list)


def _resultado_falha(passos: int, usadas: list[str]) -> AgentResult:
    # Se alguma ferramenta já rodou, pode ter escrito no vault antes da falha.
    return AgentResult(_FALHA_PARCIAL if usadas else _FALHA, False, passos, usadas)


@dataclass(frozen=True)
class _Contexto:
    autor: str | None


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def _obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


_MES = {"type": "string", "description": "Mês no formato AAAA-MM. Vazio = mês atual."}
_DATA = {"type": "string", "description": "Data no formato AAAA-MM-DD."}


# --- ferramentas de domínio ------------------------------------------------

def _ano_mes(mes: str | None) -> tuple[int, int]:
    if not mes:
        hoje = datetime.date.today()
        return hoje.year, hoje.month
    ano, m = str(mes).split("-")
    return int(ano), int(m)


async def _contas_do_mes(args: dict, ctx: _Contexto) -> dict:
    ano, mes = _ano_mes(args.get("mes"))
    contas = bills.get_contas_do_mes(ano, mes)
    if contas is None:
        return {"erro": f"a nota de contas de {ano:04d}-{mes:02d} não existe"}

    hoje = datetime.date.today()
    contas = sorted(
        contas, key=lambda c: (c["dia_vencimento"] is None, c["dia_vencimento"] or 0)
    )
    for c in contas:
        c["vencida"] = bool(
            c["status"] == "pendente"
            and c["dia_vencimento"]
            and (ano, mes, c["dia_vencimento"]) < (hoje.year, hoje.month, hoje.day)
        )
    pagas = [c for c in contas if c["status"] == "pago"]
    pendentes = [c for c in contas if c["status"] == "pendente"]
    return {
        "mes": f"{ano:04d}-{mes:02d}",
        "contas": contas,
        "totais": {
            "quantidade": len(contas),
            "pagas": len(pagas),
            "pendentes": len(pendentes),
            "vencidas": sum(1 for c in contas if c["vencida"]),
            "valor_total": round(sum(c["valor"] for c in contas), 2),
            "valor_pago": round(sum(c["valor"] for c in pagas), 2),
            "valor_pendente": round(sum(c["valor"] for c in pendentes), 2),
        },
    }


async def _marcar_conta_paga(args: dict, ctx: _Contexto) -> dict:
    ano, mes = _ano_mes(args.get("mes"))
    return await bills.mark_bill_paid(str(args["conta"]), f"{ano:04d}-{mes:02d}")


async def _adicionar_conta(args: dict, ctx: _Contexto) -> dict:
    ano, mes = _ano_mes(args.get("mes"))
    dia = int(args["dia"])
    if not 1 <= dia <= 31:
        raise ValueError(f"dia de vencimento inválido: {dia}")
    return await bills.add_bill(
        str(args["nome"]).strip(), float(args["valor"]), dia, f"{ano:04d}-{mes:02d}"
    )


async def _remover_conta(args: dict, ctx: _Contexto) -> dict:
    ano, mes = _ano_mes(args.get("mes"))
    return await bills.remove_bill(str(args["conta"]), f"{ano:04d}-{mes:02d}")


async def _gastos_do_periodo(args: dict, ctx: _Contexto) -> dict:
    ini = datetime.date.fromisoformat(args["data_inicio"])
    fim = datetime.date.fromisoformat(args["data_fim"])
    if fim < ini:
        ini, fim = fim, ini
    pessoa = str(args.get("pessoa") or "").strip().lower()
    categoria = str(args.get("categoria") or "").strip().lower()

    registros: list[dict] = []
    meses_sem_nota: list[str] = []
    ano, mes = ini.year, ini.month
    while (ano, mes) <= (fim.year, fim.month):
        conteudo = await _ler_gastos_mes(ano, mes)
        if conteudo is None:
            meses_sem_nota.append(f"{ano:04d}-{mes:02d}")
        else:
            for r in _parse_gastos_md(conteudo):
                iso = _data_iso_registro(r["data"])
                if iso is None or not ini.isoformat() <= iso <= fim.isoformat():
                    continue
                if pessoa and r["pessoa"].lower() != pessoa:
                    continue
                if categoria and r["categoria"].lower() != categoria:
                    continue
                registros.append({**r, "data": iso})
        ano, mes = (ano + 1, 1) if mes == 12 else (ano, mes + 1)

    por_pessoa: dict[str, float] = {}
    por_categoria: dict[str, float] = {}
    for r in registros:
        por_pessoa[r["pessoa"]] = por_pessoa.get(r["pessoa"], 0.0) + r["valor"]
        por_categoria[r["categoria"]] = por_categoria.get(r["categoria"], 0.0) + r["valor"]
    return {
        "periodo": {"inicio": ini.isoformat(), "fim": fim.isoformat()},
        "registros": registros,
        "quantidade": len(registros),
        "total": round(sum(r["valor"] for r in registros), 2),
        "por_pessoa": {k: round(v, 2) for k, v in por_pessoa.items()},
        "por_categoria": {k: round(v, 2) for k, v in por_categoria.items()},
        "meses_sem_nota": meses_sem_nota,
    }


async def _registrar_gasto(args: dict, ctx: _Contexto) -> dict:
    valor = float(args["valor"])
    # `|` quebraria a linha da tabela markdown.
    descricao = str(args["descricao"]).strip().replace("|", "/")
    categoria = str(args.get("categoria") or "Outros").strip().replace("|", "/")
    data = (
        datetime.date.fromisoformat(args["data"]) if args.get("data")
        else datetime.date.today()
    )
    pessoa = str(args.get("pessoa") or ctx.autor or "?").strip()
    path = _gastos_path(data)
    linha = f"| {data.strftime('%d-%m-%y')} | {pessoa} | {categoria} | {descricao} | {valor:.2f} |"
    await _append_sob_heading(path, None, linha, gerar_arquivo_novo=_gastos_arquivo_novo)
    return {"success": True, "arquivo": path, "linha": linha}


async def _agenda_bia_do_dia(args: dict, ctx: _Contexto) -> dict:
    data = datetime.date.fromisoformat(args["data"])
    return {"data": data.isoformat(), "agendamentos": get_agenda_dia(data)}


async def _agendar_cliente_bia(args: dict, ctx: _Contexto) -> dict:
    data = datetime.date.fromisoformat(args["data"])
    h, _, m = str(args["hora"]).lower().replace("h", ":").partition(":")
    hora, minuto = int(h), int(m or 0)
    if not (0 <= hora <= 23 and 0 <= minuto <= 59):
        raise ValueError(f"hora inválida: {args['hora']}")
    return await add_agendamento(str(args["cliente"]).strip(), data, f"{hora:02d}:{minuto:02d}")


async def _cancelar_agendamento_bia(args: dict, ctx: _Contexto) -> dict:
    data = datetime.date.fromisoformat(args["data"]) if args.get("data") else None
    return await cancel_agendamento(str(args["cliente"]).strip(), data)


async def _buscar_no_vault(args: dict, ctx: _Contexto) -> str:
    contexto = await search_vault_hybrid(str(args["consulta"]))
    return contexto or "(nada encontrado)"


@dataclass(frozen=True)
class _Ferramenta:
    schema: dict
    executa: Callable[[dict, _Contexto], Awaitable[object]]


_FERRAMENTAS_DOMINIO: dict[str, _Ferramenta] = {
    f.schema["function"]["name"]: f
    for f in (
        _Ferramenta(_tool(
            "contas_do_mes",
            "Contas fixas do mês (nota Contas - AAAA-MM), já ordenadas por dia de "
            "vencimento, com status, se está vencida e os totais somados. Use "
            "sempre que a pergunta envolver contas fixas.",
            _obj({"mes": _MES}),
        ), _contas_do_mes),
        _Ferramenta(_tool(
            "marcar_conta_paga",
            "Marca uma conta fixa como paga. `conta` deve ser o nome como aparece "
            "em contas_do_mes.",
            _obj({"conta": {"type": "string"}, "mes": _MES}, ["conta"]),
        ), _marcar_conta_paga),
        _Ferramenta(_tool(
            "adicionar_conta",
            "Cria uma conta fixa nova (pendente) na nota do mês.",
            _obj({
                "nome": {"type": "string"},
                "valor": {"type": "number"},
                "dia": {"type": "integer", "description": "Dia de vencimento, 1-31."},
                "mes": _MES,
            }, ["nome", "valor", "dia"]),
        ), _adicionar_conta),
        _Ferramenta(_tool(
            "remover_conta",
            "Remove uma conta fixa da nota do mês. `conta` deve ser o nome como "
            "aparece em contas_do_mes.",
            _obj({"conta": {"type": "string"}, "mes": _MES}, ["conta"]),
        ), _remover_conta),
        _Ferramenta(_tool(
            "gastos_do_periodo",
            "Gastos variáveis entre duas datas (inclusive), com filtro opcional por "
            "pessoa e categoria. Devolve os registros e os totais já somados (geral, "
            "por pessoa e por categoria). Use para qualquer pergunta de quanto foi gasto.",
            _obj({
                "data_inicio": _DATA,
                "data_fim": _DATA,
                "pessoa": {"type": "string"},
                "categoria": {"type": "string"},
            }, ["data_inicio", "data_fim"]),
        ), _gastos_do_periodo),
        _Ferramenta(_tool(
            "registrar_gasto",
            "Registra um gasto variável na nota Gastos do mês da data. `pessoa` "
            "vazio = quem está falando. Categorias usadas: Mercado, Alimentação, "
            "Transporte, Moradia, Saúde, Lazer, Contas, Outros.",
            _obj({
                "valor": {"type": "number"},
                "descricao": {"type": "string", "description": "Curta, sem valor nem data."},
                "categoria": {"type": "string"},
                "data": {**_DATA, "description": "AAAA-MM-DD. Vazio = hoje."},
                "pessoa": {"type": "string"},
            }, ["valor", "descricao", "categoria"]),
        ), _registrar_gasto),
        _Ferramenta(_tool(
            "agenda_bia_do_dia",
            "Atendimentos de clientes da Bia marcados num dia.",
            _obj({"data": _DATA}, ["data"]),
        ), _agenda_bia_do_dia),
        _Ferramenta(_tool(
            "agendar_cliente_bia",
            "Marca atendimento de cliente na agenda da Bia. Avisa se já houver "
            "cliente no mesmo horário.",
            _obj({
                "cliente": {"type": "string"},
                "data": _DATA,
                "hora": {"type": "string", "description": "HH:MM"},
            }, ["cliente", "data", "hora"]),
        ), _agendar_cliente_bia),
        _Ferramenta(_tool(
            "cancelar_agendamento_bia",
            "Cancela atendimento de cliente na agenda da Bia.",
            _obj({"cliente": {"type": "string"}, "data": _DATA}, ["cliente"]),
        ), _cancelar_agendamento_bia),
        _Ferramenta(_tool(
            "buscar_no_vault",
            "Busca híbrida (semântica + palavras-chave + reranker) no vault inteiro. "
            "Devolve trechos das notas mais relevantes com o caminho de cada uma. "
            "Boa para perguntas abertas quando você não sabe em que nota está.",
            _obj({"consulta": {"type": "string"}}, ["consulta"]),
        ), _buscar_no_vault),
    )
}


# --- ferramentas do MCP ----------------------------------------------------

async def _mcp_tools() -> list[dict]:
    """Schemas das ferramentas do MCP permitidas, lidos do servidor uma vez.
    MCP fora do ar não derruba o agente: segue só com as de domínio e tenta
    listar de novo na próxima mensagem."""
    global _mcp_tools_cache
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    try:
        result = await _mcp_rpc("tools/list", {})
    except VaultWriteError as exc:
        logger.warning("Agente: MCP indisponível, seguindo só com ferramentas de domínio (%r)", exc)
        return []

    tools = []
    for t in result.get("tools", []):
        if t.get("name") not in _MCP_TOOLS:
            continue
        schema = {k: v for k, v in (t.get("inputSchema") or {}).items() if k != "$schema"}
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        tools.append(_tool(t["name"], (t.get("description") or "")[:_MAX_DESCRICAO_MCP_CHARS], schema))
    _mcp_tools_cache = tools
    logger.info("Agente: %d ferramentas do MCP carregadas", len(tools))
    return tools


def _caminho_proibido(args: dict) -> str | None:
    for key in ("path", "from", "to"):
        p = str(args.get(key) or "")
        if p.startswith(("/", ".obsidian")) or ".." in p.split("/"):
            return p
    return None


def _formato_quebrado(args: dict) -> str | None:
    """Reescrita inteira de nota de contas/gastos precisa continuar legível
    pelos parsers, senão contas_do_mes/gastos_do_periodo passam a ver a nota
    vazia. Achado real (15/09): o agente leu uma auditoria antiga que falava
    do formato de tabela e propôs "consertar" a nota de contas pra tabela."""
    nome = str(args.get("path") or "").rsplit("/", 1)[-1]
    content = str(args.get("content") or "")
    if nome.startswith("Contas - ") and not bills.parse_contas_fixas(content):
        return (
            "escrita recusada: nenhuma conta reconhecida no conteúdo novo. A nota "
            "de contas precisa manter uma linha por conta no formato "
            "'- [ ] Nome — R$ 0,00 — dia N' ([x] = paga)"
        )
    if nome.startswith("Gastos - ") and not _parse_gastos_md(content):
        return (
            "escrita recusada: nenhum gasto reconhecido no conteúdo novo. A nota de "
            "gastos precisa manter a tabela '| Data | Pessoa | Categoria | Descrição "
            "| Valor |' com data em dd-mm-aa"
        )
    return None


async def _executa_mcp(name: str, args: dict) -> object:
    if name in _MCP_WRITE_TOOLS and (proibido := _caminho_proibido(args)) is not None:
        return {"erro": f"caminho não permitido: {proibido}"}
    if name == "create_vault_file" and (erro := _formato_quebrado(args)) is not None:
        return {"erro": erro}
    result = await _mcp_call(name, args, timeout=_MCP_TIMEOUT_S)
    textos = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    return "\n".join(textos) if textos else result


# --- loop ------------------------------------------------------------------

def _para_whatsapp(texto: str) -> str:
    """O modelo às vezes escapa pro markdown comum, que no WhatsApp aparece
    cru: `**x**` vira `*x*`, `## título` vira `*título*`, `---` some."""
    linhas = []
    for linha in texto.replace("**", "*").splitlines():
        s = linha.strip()
        if len(s) >= 3 and set(s) <= {"-", "—", "_"}:
            continue
        if s.startswith("#"):
            titulo = s.lstrip("#").strip()
            linha = f"*{titulo}*" if titulo else ""
        linhas.append(linha)
    return "\n".join(linhas).strip()


def _vazou_ferramenta(texto: str) -> bool:
    return "<|tool_call" in texto


def _remove_ferramenta_vazada(texto: str) -> str:
    return _TOKEN_FERRAMENTA_RE.sub("", _BLOCO_FERRAMENTA_RE.sub("", texto)).strip()


async def _chama(provider, messages, tools, reasoning, deadline, tool_choice="auto"):
    # `wait_for` é o limite de verdade: o timeout do cliente HTTP só estoura
    # com a conexão muda, e o OpenRouter manda keep-alive enquanto o modelo
    # pensa (achado real: chamada de 64s com 22s de prazo).
    restante = max(deadline - time.monotonic(), 1.0)
    return await asyncio.wait_for(
        provider.chat_tools(
            messages, tools, timeout=restante, reasoning=reasoning, tool_choice=tool_choice,
        ),
        timeout=restante,
    )


async def _resposta_final(provider, messages, tools, deadline):
    """Pede a resposta em texto com o que já foi levantado. Só `tool_choice=none`
    não basta: o Kimi tenta chamar ferramenta mesmo assim e o pedido sai como
    texto cru — 3 de 3 vezes no teste de 15/09; com o pedido explícito, 0 de 3.
    O pedido não entra no histórico da conversa."""
    pedido = [*messages, {"role": "user", "content": _PEDIDO_RESPOSTA_FINAL}]
    return await _chama(provider, pedido, tools, "off", deadline, tool_choice="none")


def _resumo_args(args: dict) -> str:
    """Args pro log sem o conteúdo das notas (pode ser grande ou pessoal)."""
    visivel = {k: v for k, v in args.items() if k not in ("content", "expectedContent")}
    return json.dumps(visivel, ensure_ascii=False, default=str)[:_MAX_ARGS_LOG_CHARS]


async def _executa_ferramenta(nome: str, argumentos: str, ctx: _Contexto) -> str:
    """Resultado da ferramenta em texto pro modelo. Erro nunca sobe: vira
    `{"erro": ...}` pro modelo decidir (tentar de novo, avisar o usuário)."""
    started = time.monotonic()
    try:
        args = json.loads(argumentos or "{}")
        if not isinstance(args, dict):
            raise ValueError("os argumentos precisam ser um objeto JSON")
    except ValueError as exc:
        resultado: object = {"erro": f"argumentos inválidos: {exc}"}
        args = {}
    else:
        try:
            if nome in _FERRAMENTAS_DOMINIO:
                resultado = await _FERRAMENTAS_DOMINIO[nome].executa(args, ctx)
            elif nome in _MCP_TOOLS:
                resultado = await _executa_mcp(nome, args)
            else:
                resultado = {"erro": f"ferramenta desconhecida: {nome}"}
        except (KeyError, TypeError, ValueError) as exc:
            resultado = {"erro": f"parâmetros inválidos para {nome}: {exc!r}"}
        except VaultWriteError as exc:
            resultado = {"erro": f"falha no vault: {exc}"}

    texto = (
        resultado if isinstance(resultado, str)
        else json.dumps(resultado, ensure_ascii=False, default=str)
    )
    duracao_ms = int((time.monotonic() - started) * 1000)
    if isinstance(resultado, dict) and "erro" in resultado:
        logger.warning(
            "Agente: ferramenta %s %s falhou em %dms (%s)",
            nome, _resumo_args(args), duracao_ms, resultado["erro"],
        )
    else:
        logger.info(
            "Agente: ferramenta %s %s → %d chars em %dms",
            nome, _resumo_args(args), len(texto), duracao_ms,
        )
    if len(texto) > _MAX_TOOL_CHARS:
        texto = texto[:_MAX_TOOL_CHARS] + "\n…(resultado cortado)"
    return texto


def _is_bia(sender_number: str | None) -> bool:
    return bool(sender_number) and sender_number == settings.bia_jid.split("@")[0]


def _ler_perfil_bia() -> str | None:
    """Lido do disco a cada mensagem dela: o perfil entra no prompt sem
    depender do modelo lembrar de abrir a nota, e edição no Obsidian vale na
    hora."""
    path = safe_vault_path(_PERFIL_BIA_PATH)
    if path is None or not path.is_file():
        logger.warning("Agente: perfil da Bia não encontrado em %s", _PERFIL_BIA_PATH)
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip() or None
    except OSError as exc:
        logger.warning("Agente: falha ao ler o perfil da Bia (%r)", exc)
        return None


def _system_prompt(autor: str | None, perfil: str | None = None) -> str:
    agora = datetime.datetime.now()
    nomes = " e ".join(n for n in (settings.owner_name, settings.partner_name) if n) or "o casal"
    prompt = (
        f"Você é o Yaannk, assistente pessoal de {nomes} no WhatsApp, com acesso "
        "completo ao vault do Obsidian deles.\n\n"
        f"Agora: {_DIAS_SEMANA[agora.weekday()]}, {agora:%d/%m/%Y %H:%M}.\n"
        f"Quem está falando agora: {autor or 'desconhecido'}. Cada mensagem da "
        "conversa começa com o nome de quem mandou. Chame a pessoa só pelo nome "
        "que aparece ali; se for desconhecido, não use nome.\n\n"
        "Onde as coisas ficam no vault:\n"
        "- Contas fixas: 02 - Áreas/Finanças/Contas - AAAA-MM.md\n"
        "- Gastos variáveis: 02 - Áreas/Finanças/Gastos - AAAA-MM.md\n"
        "- Lista de compras: 02 - Áreas/Finanças/Lista de Compras.md\n"
        "- Datas importantes: 02 - Áreas/Pessoas/Datas Importantes.md\n"
        "- Agenda de clientes da Bia: 02 - Áreas/Bia/Agenda - AAAA-MM.md\n"
        f"- Perfil da Bia: {_PERFIL_BIA_PATH}\n"
        "- Projetos: 01 - Projetos/. Para o resto, use get_vault_overview, "
        "list_vault_files ou as buscas.\n\n"
        "Formato ATUAL das notas de finanças (é o que o sistema lê e está "
        "funcionando; notas antigas de decisão, auditoria ou template podem "
        "descrever formatos que já mudaram, não use elas como referência):\n"
        "- Contas: uma linha por conta, `- [x] Nome — R$ 1.650,00 — dia 5` "
        "([x] paga, [ ] pendente).\n"
        "- Gastos: tabela `| Data | Pessoa | Categoria | Descrição | Valor |`, data "
        "em dd-mm-aa, valor com ponto (20.00).\n"
        "Nunca converta essas notas para outro formato.\n\n"
        "Como trabalhar:\n"
        "- Busque o dado real com as ferramentas antes de responder. Nunca "
        "invente conta, valor, data, nome de nota ou conteúdo.\n"
        "- Para contas fixas, gastos e agenda da Bia, use as ferramentas "
        "específicas (contas_do_mes, gastos_do_periodo, registrar_gasto etc.). "
        "Elas já somam e mantêm o formato das notas. Não some valores por conta "
        "própria: use os totais que a ferramenta devolve.\n"
        "- Liste contas por data de vencimento, da mais próxima para a mais "
        "distante, a não ser que peçam outra ordem.\n"
        "- Datas relativas (hoje, ontem, essa semana, mês passado) são a partir "
        "de agora. Semana vai de segunda a domingo.\n"
        "- Antes de editar uma nota, leia ela. Mude só o necessário e mantenha a "
        "formatação. Prefira patch_vault_file ou append_to_vault_file; ao "
        "reescrever a nota inteira com create_vault_file, passe expectedContent "
        "com o conteúdo que você leu.\n"
        "- Apagar, mover ou reescrever uma nota inteira só com pedido explícito. "
        "Se o pedido for ambíguo (qual nota, qual conta, qual valor), pergunte "
        "antes de mexer.\n"
        "- Se uma ferramenta devolver erro, tente corrigir (nome exato, outro "
        "caminho) ou explique o que aconteceu. Nunca diga que fez algo que a "
        "ferramenta não confirmou.\n"
        "- Considere o que já foi dito na conversa.\n"
        "- Em varredura ou revisão do vault, confira com os dados reais (as "
        "ferramentas de contas e gastos) antes de apontar algo como quebrado, e "
        "só proponha: não altere nada sem confirmação.\n\n"
        "Formato da resposta (WhatsApp):\n"
        "- Português do Brasil, direto, tom de conversa.\n"
        "- Negrito com *um asterisco*. Sem tabelas, sem títulos com #, sem "
        "blocos de código. Listas com • ou -.\n"
        "- Depois de alterar algo, confirme o que mudou e em qual nota.\n"
        "- Não fale de ferramentas, JSON ou de como você funciona por dentro."
    )
    if perfil:
        prompt += (
            "\n\nPerfil de quem está falando (Beatriz, a Bia), da nota "
            f"{_PERFIL_BIA_PATH}. Use para entender quem ela é, o salão e como ela "
            "prefere as respostas:\n" + perfil
        )
    return prompt


async def answer(
    text: str,
    *,
    conv_key: str,
    autor: str | None = None,
    history: list[dict] | None = None,
    sender_number: str | None = None,
) -> AgentResult:
    started = time.monotonic()
    deadline = started + settings.agent_timeout_s
    ctx = _Contexto(autor=autor)
    if history is None:
        history = get_recent_messages(conv_key)

    perfil = None
    if _is_bia(sender_number):
        perfil = _ler_perfil_bia()
        if perfil:
            logger.info("Agente: perfil da Bia no prompt (%d chars)", len(perfil))

    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(autor, perfil)},
        *history,
        {"role": "user", "content": f"{autor}: {text}" if autor else text},
    ]
    tools = [f.schema for f in _FERRAMENTAS_DOMINIO.values()] + await _mcp_tools()
    provider = get_provider("kimi")
    usadas: list[str] = []
    logger.info(
        "Agente: iniciando (%d msg(s) de histórico, %d ferramentas)", len(history), len(tools)
    )

    for passo in range(settings.agent_max_steps):
        restante = deadline - time.monotonic()
        if restante <= 0:
            break
        forcar_resposta = (
            passo == settings.agent_max_steps - 1 or restante < _RESERVA_RESPOSTA_S
        )
        if forcar_resposta and passo > 0:
            logger.info(
                "Agente: passo %d forçado a responder (%.0fs restantes)", passo + 1, restante
            )
        try:
            if forcar_resposta:
                turno = await _resposta_final(provider, messages, tools, deadline)
            else:
                # O passo normal não pode usar a reserva: uma chamada lenta sozinha
                # consumia o prazo inteiro e não sobrava tempo pra responder (achado
                # real 15/09: passo 5 começou com 109s e estourou os 180s).
                try:
                    turno = await _chama(
                        provider, messages, tools, settings.agent_reasoning,
                        deadline - _RESERVA_RESPOSTA_S,
                    )
                except TimeoutError:
                    logger.warning(
                        "Agente: passo %d passou do prazo — respondendo com o que já levantou",
                        passo + 1,
                    )
                    turno = await _resposta_final(provider, messages, tools, deadline)
                if not turno.tool_calls and (
                    not turno.content or _vazou_ferramenta(turno.content)
                ):
                    # Vazio: o K2.6 às vezes gasta a saída inteira raciocinando.
                    # Token vazado: quis chamar ferramenta fora do formato.
                    logger.warning(
                        "Agente: resposta %s no passo %d (finish_reason=%s) — "
                        "pedindo resposta final",
                        "com ferramenta vazada" if turno.content else "vazia",
                        passo + 1, turno.finish_reason,
                    )
                    turno = await _resposta_final(provider, messages, tools, deadline)
        except TimeoutError:
            logger.warning(
                "Agente: estourou o limite de %.0fs no passo %d",
                settings.agent_timeout_s, passo + 1,
            )
            return _resultado_falha(passo + 1, usadas)
        except LLMError as exc:
            logger.warning("Agente: Kimi falhou no passo %d (%r)", passo + 1, exc)
            return _resultado_falha(passo + 1, usadas)

        logger.info(
            "Agente: passo %d em %dms (finish_reason=%s, %d ferramenta(s))",
            passo + 1, turno.latency_ms, turno.finish_reason, len(turno.tool_calls),
        )
        messages.append(turno.message)

        if not turno.tool_calls:
            if _vazou_ferramenta(turno.content):
                logger.warning(
                    "Agente: ferramenta vazada de novo no passo %d — limpando o texto", passo + 1
                )
            texto = _para_whatsapp(_remove_ferramenta_vazada(turno.content))
            if not texto:
                logger.warning("Agente: sem resposta aproveitável no passo %d", passo + 1)
                return _resultado_falha(passo + 1, usadas)
            logger.info(
                "Agente: respondeu em %d passo(s), %dms total, ferramentas=%s",
                passo + 1, int((time.monotonic() - started) * 1000), usadas,
            )
            return AgentResult(texto, True, passo + 1, usadas)

        for call in turno.tool_calls:
            usadas.append(call.name)
            saida = await _executa_ferramenta(call.name, call.arguments, ctx)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": saida})

    logger.warning(
        "Agente: parou sem resposta final (%dms, ferramentas=%s)",
        int((time.monotonic() - started) * 1000), usadas,
    )
    return _resultado_falha(settings.agent_max_steps, usadas)
