"""Loop do agente Yaannk (D-11) com o Kimi substituído por turnos roteirizados.

Nada aqui chama LLM, MCP ou vault reais.
"""

import asyncio
import dataclasses
import json

import pytest

from app.agents import yaannk_agent
from app.services.llm.base import LLMError
from app.services.llm.kimi import ToolCall, ToolTurn


class _FakeKimi:
    def __init__(self, turnos):
        self.turnos = list(turnos)
        self.chamadas = []

    async def chat_tools(self, messages, tools, **kwargs):
        self.chamadas.append({"messages": [dict(m) for m in messages], "tools": tools, **kwargs})
        turno = self.turnos.pop(0)
        if isinstance(turno, Exception):
            raise turno
        return turno


def _pede_ferramenta(nome, args, call_id="c1"):
    arguments = json.dumps(args)
    return ToolTurn(
        message={
            "role": "assistant",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": nome, "arguments": arguments}}],
        },
        content="",
        tool_calls=[ToolCall(id=call_id, name=nome, arguments=arguments)],
        finish_reason="tool_calls",
        latency_ms=1,
    )


def _responde(texto):
    return ToolTurn(
        message={"role": "assistant", "content": texto},
        content=texto, tool_calls=[], finish_reason="stop", latency_ms=1,
    )


@pytest.fixture
def kimi(monkeypatch):
    def _instala(turnos):
        fake = _FakeKimi(turnos)
        monkeypatch.setattr(yaannk_agent, "get_provider", lambda name=None: fake)

        async def _sem_mcp():
            return []

        monkeypatch.setattr(yaannk_agent, "_mcp_tools", _sem_mcp)
        return fake

    return _instala


_JUIZ_REAL = yaannk_agent._acoes_confirmadas


@pytest.fixture(autouse=True)
def _sem_juiz(monkeypatch):
    """Por padrão o juiz da confirmação (D-15) não gasta turno roteirizado."""
    async def _zero(provider, texto, deadline):
        return 0

    monkeypatch.setattr(yaannk_agent, "_acoes_confirmadas", _zero)


@pytest.fixture
def com_juiz(monkeypatch):
    monkeypatch.setattr(yaannk_agent, "_acoes_confirmadas", _JUIZ_REAL)


def _tool_msgs(chamada):
    return [m for m in chamada["messages"] if m["role"] == "tool"]


def test_resultado_da_ferramenta_volta_pro_modelo_ordenado_e_somado(kimi, monkeypatch):
    contas = [
        {"descricao": "Luz", "valor": 200.0, "dia_vencimento": 18, "status": "pendente"},
        {"descricao": "Avulsa", "valor": 57.0, "dia_vencimento": None, "status": "pago"},
        {"descricao": "Aluguel", "valor": 1650.0, "dia_vencimento": 5, "status": "pago"},
    ]
    monkeypatch.setattr(yaannk_agent.bills, "get_contas_do_mes", lambda ano, mes: contas)
    fake = kimi([_pede_ferramenta("contas_do_mes", {"mes": "2026-09"}), _responde("pronto")])

    result = asyncio.run(yaannk_agent.answer("contas?", conv_key="g", autor="Douglas", history=[]))

    assert result.succeeded
    assert result.tools == ["contas_do_mes"]
    [tool_msg] = _tool_msgs(fake.chamadas[1])
    assert tool_msg["tool_call_id"] == "c1"
    dados = json.loads(tool_msg["content"])
    assert [c["descricao"] for c in dados["contas"]] == ["Aluguel", "Luz", "Avulsa"]
    assert dados["totais"]["valor_pendente"] == 200.0
    assert dados["totais"]["pagas"] == 2


def test_historico_e_autor_entram_na_conversa(kimi):
    historico = [
        {"role": "user", "content": "Bia: quanto falta pagar?"},
        {"role": "assistant", "content": "Faltam 3 contas."},
    ]
    fake = kimi([_responde("oi")])

    asyncio.run(yaannk_agent.answer(
        "e quais são?", conv_key="g", autor="Douglas", history=historico
    ))

    msgs = fake.chamadas[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert "Quem está falando agora: Douglas" in msgs[0]["content"]
    assert msgs[1:3] == historico
    assert msgs[-1] == {"role": "user", "content": "Douglas: e quais são?"}


def test_resposta_vazia_tenta_de_novo_sem_raciocinio(kimi):
    fake = kimi([_responde(""), _responde("Você é o Douglas.")])

    result = asyncio.run(yaannk_agent.answer(
        "consegue identificar quem está falando?", conv_key="g", autor="Douglas", history=[]
    ))

    assert result.succeeded
    assert [c["reasoning"] for c in fake.chamadas] == [yaannk_agent.settings.agent_reasoning, "off"]
    # a tentativa vazia não entra na conversa
    assert not any(m["role"] == "assistant" for m in fake.chamadas[1]["messages"])


_VAZADO = (
    '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_vault_file:10'
    '<|tool_call_argument_begin|>{"path": "CURRENT.md"}<|tool_call_end|><|tool_calls_section_end|>'
)


def test_ferramenta_vazada_no_texto_pede_resposta_final_e_nao_chega_no_usuario(kimi):
    fake = kimi([_responde(_VAZADO), _responde("Encontrei 3 pontos pra arrumar.")])

    result = asyncio.run(yaannk_agent.answer("faz a varredura", conv_key="g", history=[]))

    assert result.succeeded
    assert result.text == "Encontrei 3 pontos pra arrumar."
    final = fake.chamadas[1]
    assert final["tool_choice"] == "none"
    assert final["messages"][-1] == {"role": "user", "content": yaannk_agent._PEDIDO_RESPOSTA_FINAL}


def test_vazamento_que_persiste_e_limpo_do_texto(kimi):
    kimi([_responde(_VAZADO), _responde(f"Resumo parcial do vault.{_VAZADO}")])

    result = asyncio.run(yaannk_agent.answer("faz a varredura", conv_key="g", history=[]))

    assert result.succeeded
    assert result.text == "Resumo parcial do vault."


def test_so_ferramenta_vazada_sem_texto_vira_falha_honesta(kimi):
    kimi([_responde(_VAZADO), _responde(_VAZADO)])

    result = asyncio.run(yaannk_agent.answer("faz a varredura", conv_key="g", history=[]))

    assert not result.succeeded
    assert "<|" not in result.text


def test_resposta_forcada_leva_o_pedido_de_resposta_final(kimi, monkeypatch):
    monkeypatch.setattr(
        yaannk_agent, "settings", dataclasses.replace(yaannk_agent.settings, agent_max_steps=1)
    )
    fake = kimi([_responde("ok")])

    asyncio.run(yaannk_agent.answer("x", conv_key="g", history=[]))

    assert fake.chamadas[0]["messages"][-1]["content"] == yaannk_agent._PEDIDO_RESPOSTA_FINAL


def test_negrito_markdown_vira_negrito_do_whatsapp(kimi):
    kimi([_responde("**Pendentes:** *Luz* — R$ 200,00")])

    result = asyncio.run(yaannk_agent.answer("x", conv_key="g", history=[]))

    assert result.text == "*Pendentes:* *Luz* — R$ 200,00"


def test_mensagem_da_bia_leva_o_perfil_dela_no_prompt(kimi, monkeypatch, tmp_path):
    perfil = tmp_path / "Beatriz - Perfil.md"
    perfil.write_text("Beatriz, cabeleireira. Prefere respostas curtas.", encoding="utf-8")
    monkeypatch.setattr(
        yaannk_agent, "settings",
        dataclasses.replace(yaannk_agent.settings, bia_jid="5521900000000@s.whatsapp.net"),
    )
    monkeypatch.setattr(yaannk_agent, "safe_vault_path", lambda relpath: perfil)
    fake = kimi([_responde("oi, Bia"), _responde("oi, Douglas")])

    asyncio.run(yaannk_agent.answer(
        "oi", conv_key="bia", autor="Bia", history=[], sender_number="5521900000000"
    ))
    asyncio.run(yaannk_agent.answer(
        "oi", conv_key="douglas", autor="Douglas", history=[], sender_number="5521911111111"
    ))

    assert "Prefere respostas curtas" in fake.chamadas[0]["messages"][0]["content"]
    assert "Prefere respostas curtas" not in fake.chamadas[1]["messages"][0]["content"]


def test_ultimo_passo_obriga_a_responder_sem_ferramenta(kimi, monkeypatch):
    monkeypatch.setattr(
        yaannk_agent, "settings", dataclasses.replace(yaannk_agent.settings, agent_max_steps=2)
    )
    fake = kimi([_pede_ferramenta("ferramenta_que_nao_existe", {}), _responde("ok")])

    asyncio.run(yaannk_agent.answer("x", conv_key="g", history=[]))

    assert fake.chamadas[0]["tool_choice"] == "auto"
    assert fake.chamadas[1]["tool_choice"] == "none"


def test_perto_do_limite_responde_sem_explorar_nem_raciocinar(kimi, monkeypatch):
    monkeypatch.setattr(
        yaannk_agent, "settings", dataclasses.replace(yaannk_agent.settings, agent_timeout_s=30.0)
    )
    fake = kimi([_responde("com o que eu tenho: ...")])

    asyncio.run(yaannk_agent.answer("x", conv_key="g", history=[]))

    assert fake.chamadas[0]["tool_choice"] == "none"
    assert fake.chamadas[0]["reasoning"] == "off"


def test_chamada_lenta_nao_passa_do_limite(kimi, monkeypatch):
    monkeypatch.setattr(
        yaannk_agent, "settings", dataclasses.replace(yaannk_agent.settings, agent_timeout_s=0.2)
    )
    fake = kimi([])

    async def _lenta(*a, **kw):
        await asyncio.sleep(5)

    fake.chat_tools = _lenta

    async def _roda():
        started = asyncio.get_running_loop().time()
        result = await yaannk_agent.answer("x", conv_key="g", history=[])
        return result, asyncio.get_running_loop().time() - started

    result, duracao = asyncio.run(_roda())

    assert not result.succeeded
    # a chamada leva 5s; o limite corta bem antes (piso de 1s por chamada em `_chama`)
    assert duracao < 2.0


def test_passo_lento_nao_come_a_reserva_e_ainda_responde(kimi, monkeypatch):
    # 41.5s de limite com 40s de reserva: o passo normal tem ~1.5s; a chamada
    # lenta estoura esse prazo e a resposta final usa a reserva.
    monkeypatch.setattr(
        yaannk_agent, "settings", dataclasses.replace(yaannk_agent.settings, agent_timeout_s=41.5)
    )
    fake = kimi([])
    chamadas = []

    async def _primeira_lenta(messages, tools, **kw):
        chamadas.append(kw)
        if len(chamadas) == 1:
            await asyncio.sleep(5)
        return _responde("com o que eu já levantei: ...")

    fake.chat_tools = _primeira_lenta

    result = asyncio.run(yaannk_agent.answer("x", conv_key="g", history=[]))

    assert result.succeeded
    assert result.text == "com o que eu já levantei: ..."
    assert [c["tool_choice"] for c in chamadas] == ["auto", "none"]


def test_ferramenta_desconhecida_vira_erro_pro_modelo(kimi):
    fake = kimi([_pede_ferramenta("apaga_tudo", {}), _responde("não deu")])

    result = asyncio.run(yaannk_agent.answer("x", conv_key="g", history=[]))

    assert result.succeeded
    [tool_msg] = _tool_msgs(fake.chamadas[1])
    assert "ferramenta desconhecida" in json.loads(tool_msg["content"])["erro"]


def test_argumentos_invalidos_viram_erro_pro_modelo(kimi):
    fake = kimi([
        _pede_ferramenta("gastos_do_periodo", {"data_inicio": "ontem", "data_fim": "hoje"}),
        _responde("qual período?"),
    ])

    asyncio.run(yaannk_agent.answer("x", conv_key="g", history=[]))

    [tool_msg] = _tool_msgs(fake.chamadas[1])
    assert "parâmetros inválidos" in json.loads(tool_msg["content"])["erro"]


@pytest.mark.parametrize("args", [
    {"path": "../fora.md"},
    {"path": "/etc/passwd"},
    {"path": ".obsidian/plugins/x.json"},
    {"from": "ok.md", "to": "../fora.md"},
])
def test_escrita_fora_do_vault_nao_chega_no_mcp(monkeypatch, args):
    async def _mcp_nao_pode(*a, **kw):
        raise AssertionError("não devia chamar o MCP")

    monkeypatch.setattr(yaannk_agent, "_mcp_call", _mcp_nao_pode)
    nome = "rename_vault_file" if "from" in args else "delete_vault_file"

    saida = asyncio.run(yaannk_agent._executa_ferramenta(
        nome, json.dumps(args), yaannk_agent._Contexto(autor="Douglas")
    ))

    assert "caminho não permitido" in json.loads(saida)["erro"]


def test_markdown_de_titulo_e_separador_vira_formato_whatsapp(kimi):
    kimi([_responde("## 🔴 Crítico\n\n---\n\n- item um\n### Leves\ntexto")])

    result = asyncio.run(yaannk_agent.answer("x", conv_key="g", history=[]))

    assert result.text == "*🔴 Crítico*\n\n\n- item um\n*Leves*\ntexto"


@pytest.mark.parametrize("path,content", [
    (
        "02 - Áreas/Finanças/Contas - 2026-09.md",
        "| Descrição | Valor | Vencimento | Status |\n|---|---|---|---|\n| Aluguel | 1650 | 5 | pago |",
    ),
    ("02 - Áreas/Finanças/Gastos - 2026-09.md", "# Gastos\n\n- 09/09 mercado R$ 20"),
])
def test_reescrita_que_quebra_formato_de_financas_e_recusada(monkeypatch, path, content):
    async def _mcp_nao_pode(*a, **kw):
        raise AssertionError("não devia chamar o MCP")

    monkeypatch.setattr(yaannk_agent, "_mcp_call", _mcp_nao_pode)

    saida = asyncio.run(yaannk_agent._executa_ferramenta(
        "create_vault_file", json.dumps({"path": path, "content": content}),
        yaannk_agent._Contexto(autor="Douglas"),
    ))

    assert "escrita recusada" in json.loads(saida)["erro"]


def test_reescrita_de_contas_no_formato_certo_chega_no_mcp(monkeypatch):
    chamadas = []

    async def _fake_mcp(name, args, **kw):
        chamadas.append(name)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(yaannk_agent, "_mcp_call", _fake_mcp)
    content = "# Contas\n\n- [x] Aluguel — R$ 1.650,00 — dia 5\n- [ ] Luz — R$ 200,00 — dia 18\n"

    saida = asyncio.run(yaannk_agent._executa_ferramenta(
        "create_vault_file",
        json.dumps({"path": "02 - Áreas/Finanças/Contas - 2026-09.md", "content": content}),
        yaannk_agent._Contexto(autor="Douglas"),
    ))

    assert saida == "ok"
    assert chamadas == ["create_vault_file"]


def test_registrar_gasto_escreve_linha_no_formato_da_tabela(monkeypatch):
    gravado = {}

    async def _fake_append(path, heading, linha, *, gerar_arquivo_novo=None):
        gravado.update(path=path, heading=heading, linha=linha)

    monkeypatch.setattr(yaannk_agent, "_append_sob_heading", _fake_append)
    args = {"valor": 23.5, "descricao": "pão | leite", "categoria": "Mercado", "data": "2026-09-14"}

    asyncio.run(yaannk_agent._executa_ferramenta(
        "registrar_gasto", json.dumps(args), yaannk_agent._Contexto(autor="Douglas")
    ))

    assert gravado["path"] == "02 - Áreas/Finanças/Gastos - 2026-09.md"
    assert gravado["heading"] is None
    assert gravado["linha"] == "| 14-09-26 | Douglas | Mercado | pão / leite | 23.50 |"


def test_falha_do_kimi_depois_de_escrever_avisa_que_pode_ter_feito_algo(kimi, monkeypatch):
    async def _fake_append(*a, **kw):
        return None

    monkeypatch.setattr(yaannk_agent, "_append_sob_heading", _fake_append)
    kimi([
        _pede_ferramenta("registrar_gasto", {"valor": 10, "descricao": "café", "categoria": "Alimentação"}),
        LLMError("timeout"),
    ])

    result = asyncio.run(yaannk_agent.answer("gastei 10 no café", conv_key="g", history=[]))

    assert not result.succeeded
    assert result.text == yaannk_agent._FALHA_PARCIAL


def test_historico_mostra_ferramentas_que_rodaram_em_cada_resposta(kimi):
    historico = [
        {"role": "user", "content": "Bia: marca a Thayna amanhã 8h"},
        {"role": "assistant", "content": "Marquei a Thayna.", "tools": [
            {"nome": "agendar_cliente_bia", "args": '{"cliente": "Thayna"}', "ok": True},
        ]},
        {"role": "user", "content": "Bia: e a Dalmacia 9h30"},
        {"role": "assistant", "content": "Marquei a Dalmacia.", "tools": []},
        {"role": "assistant", "content": "resposta antiga", "tools": None},
    ]
    fake = kimi([_responde("oi")])

    asyncio.run(yaannk_agent.answer("oi", conv_key="g", autor="Bia", history=historico))

    msgs = fake.chamadas[0]["messages"]
    assert "agendar_cliente_bia" in msgs[2]["content"] and "→ ok" in msgs[2]["content"]
    assert msgs[2]["content"].endswith("Marquei a Thayna.")
    assert "nenhuma ferramenta" in msgs[4]["content"]
    assert msgs[5] == {"role": "assistant", "content": "resposta antiga"}
    assert all("tools" not in m for m in msgs)


def test_resposta_registra_acoes_com_sucesso_e_falha(kimi, monkeypatch):
    async def _agenda(args, ctx):
        return {"success": False, "conflict": True, "message": "conflito"}

    ferramenta = yaannk_agent._FERRAMENTAS_DOMINIO["agendar_cliente_bia"]
    monkeypatch.setitem(
        yaannk_agent._FERRAMENTAS_DOMINIO, "agendar_cliente_bia",
        dataclasses.replace(ferramenta, executa=_agenda),
    )
    kimi([
        _pede_ferramenta("agendar_cliente_bia", {"cliente": "Dalmacia", "data": "2026-09-16", "hora": "09:30"}),
        _responde("[registro interno: ferramentas chamadas nesta resposta: x → ok]\nDeu conflito."),
    ])

    result = asyncio.run(yaannk_agent.answer("marca", conv_key="g", autor="Bia", history=[]))

    assert result.acoes == [{
        "nome": "agendar_cliente_bia",
        "args": '{"cliente": "Dalmacia", "data": "2026-09-16", "hora": "09:30"}',
        "ok": False,
    }]
    assert result.text == "Deu conflito."


def test_conversation_store_grava_e_le_ferramentas(monkeypatch, tmp_path):
    from app.services import conversation_store, db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "c.db")
    conversation_store.add_message("k", "user", "oi")
    conversation_store.add_message("k", "assistant", "feito", tools=[{"nome": "x", "args": "{}", "ok": True}])
    conversation_store.add_message("k", "assistant", "só conversa", tools=[])

    assert conversation_store.get_recent_messages("k") == [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "feito"},
        {"role": "assistant", "content": "só conversa"},
    ]
    com = conversation_store.get_recent_messages("k", with_tools=True)
    assert com[1]["tools"] == [{"nome": "x", "args": "{}", "ok": True}]
    assert com[2]["tools"] == []


# --- gate da confirmação falsa (D-15) --------------------------------------

def _escrita_falsa(monkeypatch):
    chamadas = []

    async def _agenda(args, ctx):
        chamadas.append(args)
        return {"success": True}

    f = yaannk_agent._FERRAMENTAS_DOMINIO["agendar_cliente_bia"]
    monkeypatch.setitem(
        yaannk_agent._FERRAMENTAS_DOMINIO, "agendar_cliente_bia",
        dataclasses.replace(f, executa=_agenda),
    )
    return chamadas


def test_confirmacao_sem_ferramenta_e_descartada_e_reexecutada_com_tool_required(
    kimi, com_juiz, monkeypatch
):
    escritas = _escrita_falsa(monkeypatch)
    fake = kimi([
        _responde("Agendado! ✅ Luiza às 15h."),           # confirmação falsa
        _responde("1"),                                    # juiz
        _pede_ferramenta("agendar_cliente_bia", {"cliente": "Luiza", "data": "2026-09-24", "hora": "15:00"}),
        _responde("Agendei a Luiza às 15h."),
        _responde("1"),                                    # juiz
    ])
    result = asyncio.run(yaannk_agent.answer("agenda a Luiza 3h", conv_key="g", history=[]))

    assert escritas == [{"cliente": "Luiza", "data": "2026-09-24", "hora": "15:00"}]
    assert result.succeeded and result.text == "Agendei a Luiza às 15h."
    assert result.tools == ["agendar_cliente_bia"]
    retry = fake.chamadas[2]
    assert retry["tool_choice"] == "required"
    assert retry["messages"][-1] == {"role": "system", "content": yaannk_agent._aviso_sem_ferramenta(1, 0)}
    assert all("Agendado! ✅" not in str(m.get("content")) for m in retry["messages"])


def test_pergunta_de_am_pm_passa_direto_sem_retry(kimi, com_juiz):
    fake = kimi([_responde("Às 3h da manhã ou da tarde?"), _responde("0")])
    result = asyncio.run(yaannk_agent.answer("agenda a Luiza às 3", conv_key="g", history=[]))

    assert result.text == "Às 3h da manhã ou da tarde?"
    assert len(fake.chamadas) == 2
    assert all(c.get("tool_choice") != "required" for c in fake.chamadas)


def test_confirmacao_sem_ferramenta_de_novo_vira_falha_honesta(kimi, com_juiz):
    kimi([
        _responde("Agendado ✅"), _responde("1"),
        _responde("Agendado ✅ mesmo"), _responde("1"),
    ])
    result = asyncio.run(yaannk_agent.answer("agenda", conv_key="g", history=[]))

    assert not result.succeeded
    assert result.text == yaannk_agent._FALHA


def test_confirmacao_igual_as_escritas_feitas_passa_sem_retry(kimi, com_juiz, monkeypatch):
    _escrita_falsa(monkeypatch)
    fake = kimi([
        _pede_ferramenta("agendar_cliente_bia", {"cliente": "Luiza"}),
        _responde("Agendei a Luiza."),
        _responde("1"),
    ])
    result = asyncio.run(yaannk_agent.answer("agenda a Luiza", conv_key="g", history=[]))

    assert result.text == "Agendei a Luiza."
    assert len(fake.chamadas) == 3
    assert all(c.get("tool_choice") != "required" for c in fake.chamadas)


def test_confirma_duas_e_executa_uma_o_gate_pega_e_reexecuta(kimi, com_juiz, monkeypatch):
    escritas = _escrita_falsa(monkeypatch)
    fake = kimi([
        _pede_ferramenta("agendar_cliente_bia", {"cliente": "Luiza"}, "c1"),
        _responde("Agendei a Luiza e a mãe dela."),         # confirma 2, fez 1
        _responde("2"),                                     # juiz
        _pede_ferramenta("agendar_cliente_bia", {"cliente": "Mãe da Luiza"}, "c2"),
        _responde("Agendei a Luiza e a mãe dela."),
        _responde("2"),                                     # juiz: 2 == 2
    ])
    result = asyncio.run(yaannk_agent.answer("agenda Luiza e a mãe", conv_key="g", history=[]))

    assert [e["cliente"] for e in escritas] == ["Luiza", "Mãe da Luiza"]
    assert result.succeeded and result.tools == ["agendar_cliente_bia"] * 2
    retry = fake.chamadas[3]
    assert retry["tool_choice"] == "required"
    assert retry["messages"][-1] == {
        "role": "system", "content": yaannk_agent._aviso_sem_ferramenta(2, 1),
    }
    assert "faltam 1" in retry["messages"][-1]["content"]


def test_escrita_que_falhou_nao_conta_como_feita(kimi, com_juiz, monkeypatch):
    async def _falha(args, ctx):
        return {"erro": "conflito"}

    f = yaannk_agent._FERRAMENTAS_DOMINIO["agendar_cliente_bia"]
    monkeypatch.setitem(
        yaannk_agent._FERRAMENTAS_DOMINIO, "agendar_cliente_bia",
        dataclasses.replace(f, executa=_falha),
    )
    fake = kimi([
        _pede_ferramenta("agendar_cliente_bia", {"cliente": "Luiza"}),
        _responde("Agendei a Luiza."),
        _responde("1"),
        _responde("Deu conflito, não agendei."),
        _responde("0"),
    ])
    result = asyncio.run(yaannk_agent.answer("agenda a Luiza", conv_key="g", history=[]))

    assert result.text == "Deu conflito, não agendei."
    assert fake.chamadas[3]["tool_choice"] == "required"


def test_juiz_que_falha_nao_trava_a_resposta(kimi, com_juiz):
    kimi([_responde("Agendado ✅"), LLMError("fora do ar")])
    result = asyncio.run(yaannk_agent.answer("agenda", conv_key="g", history=[]))
    assert result.text == "Agendado ✅"
