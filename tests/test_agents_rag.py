"""Agentes RAG das Fases C e D — `app/agents/rag_agent.py`.

Tudo offline: `retrieve()`/`_block3()` entram como dublês. O que se testa é
o contrato do wrapper — que cada agente amarra a skill certa, que os
metadados de retrieval sobrevivem até o `AgentResponse` (sem isso o
benchmark perde `rag_contains`/`tier`/`n_docs`) e que o registry resolve
cada nome pro agente certo.
"""

import asyncio
import types

import pytest

from app.agents.base import AgentResponse
from app.agents.rag_agent import AgentCarreira, AgentConhecimento, AgentYaannk
from app.agents.registry import get_agent
from app.services.skills import SKILLS


def _fake_retrieval(skill_name: str):
    """`RetrievalResult` mínimo — só os campos que o wrapper repassa."""
    decision = types.SimpleNamespace(intent="technical_rag", provider="kimi")
    return types.SimpleNamespace(
        skill_name=skill_name, decision=decision,
        rag_files=["01 - Projetos/Pessoal/YaannkAgent/YaannkAgent - STATE.md"],
        vault_context="### nota.md\nconteúdo", decomposed=False, n_docs=1,
    )


def _run_agent(monkeypatch, agent, message="pergunta qualquer"):
    """Roda `agent.handle()` com `retrieve`/`_block3` dublados, devolvendo
    (resposta, skill_name que o agente pediu ao `retrieve`)."""
    from app.services import pipeline

    pedido = {}

    async def fake_retrieve(text, *, conv_key, skill_name=None):
        pedido["skill_name"] = skill_name
        pedido["conv_key"] = conv_key
        return _fake_retrieval(skill_name)

    async def fake_block3(text, r, skill, history):
        pedido["prompt_injection"] = skill["prompt_injection"]
        return "resposta do bloco 3", True

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "_block3", fake_block3)

    resposta = asyncio.run(
        agent.handle(message, routing=None, history=[], sender="Douglas", conv_key="chat-1")
    )
    return resposta, pedido


# --- cada agente amarra a skill certa --------------------------------------

@pytest.mark.parametrize(
    "agent,skill_esperada,nome_esperado",
    [
        (AgentYaannk(), "yaannk_tecnico", "yaannk"),
        (AgentConhecimento(), "conhecimento", "conhecimento"),
        (AgentCarreira(), "carreira", "carreira"),
    ],
)
def test_agente_usa_a_skill_do_seu_dominio(monkeypatch, agent, skill_esperada, nome_esperado):
    resposta, pedido = _run_agent(monkeypatch, agent)
    assert pedido["skill_name"] == skill_esperada
    assert resposta.agent == nome_esperado
    assert resposta.source == "rag"
    assert resposta.text == "resposta do bloco 3"


def test_prompt_injection_vem_da_skill_do_agente(monkeypatch):
    _, pedido = _run_agent(monkeypatch, AgentConhecimento())
    assert pedido["prompt_injection"] == SKILLS["conhecimento"]["prompt_injection"]


# --- metadados de retrieval sobrevivem (gate do benchmark) -----------------

def test_metadados_de_retrieval_chegam_no_agent_response(monkeypatch):
    """Sem isso, despachar o caminho lento por agente apagaria
    `rag_contains`/`tier`/`n_docs` do `PipelineResult` — e o benchmark
    inteiro de retrieval iria junto."""
    resposta, _ = _run_agent(monkeypatch, AgentYaannk())
    assert resposta.rag_files == [
        "01 - Projetos/Pessoal/YaannkAgent/YaannkAgent - STATE.md"
    ]
    assert resposta.n_docs == 1
    assert resposta.vault_context == "### nota.md\nconteúdo"
    assert resposta.decomposed is False
    assert resposta.intent == "technical_rag"
    assert resposta.decision is not None


# --- registry ---------------------------------------------------------------

@pytest.mark.parametrize(
    "nome,classe",
    [
        ("yaannk", AgentYaannk),
        ("conhecimento", AgentConhecimento),
        ("carreira", AgentCarreira),
    ],
)
def test_registry_resolve_os_agentes_novos(nome, classe):
    assert isinstance(get_agent(nome), classe)


def test_registry_devolve_none_para_agente_sem_registro():
    assert get_agent("default") is None
    assert get_agent("inexistente") is None


# --- skills novas da Fase D -------------------------------------------------

@pytest.mark.parametrize(
    "skill,pasta",
    [
        ("conhecimento", "05 - Faculdade"),
        ("carreira", "02 - Áreas/Trabalho"),
    ],
)
def test_skills_da_fase_d_apontam_para_pastas_reais_do_vault(skill, pasta):
    assert SKILLS[skill]["priority_folder"] == pasta


def test_todas_as_skills_tem_os_quatro_campos():
    for nome, skill in SKILLS.items():
        assert set(skill) == {
            "keywords", "priority_folder", "prompt_injection", "excluded_subfolders"
        }, f"skill {nome} com campos fora do contrato"
