"""Orquestrador (Fase A) — `app/services/orchestrator.py`.

Tudo aqui é offline: o Kimi entra como dublê. O que se testa é o contrato —
parse da resposta, mapeamento skill_name -> agent, e cada caminho de queda
para o fallback determinístico (`classify_intent()`) — porque é isso que
garante a invariante: o usuário nunca vê uma falha do orquestrador, só uma
resposta um pouco menos precisa.

`orch-06`/`orch-09` (flag desligada) testam o `pipeline.py`, não o
`orchestrator.py` isoladamente — a decisão de nunca chamar `route()` com
`ORCHESTRATOR_ENABLED=false` é do pipeline (Q1, sessão 17), então é lá que
a garantia precisa ser verificada.
"""

import asyncio
import dataclasses
import json

import pytest

from app.services import orchestrator, pipeline
from app.services.llm.base import LLMError
from app.services.orchestrator import OrchestratorDecision, _fallback, route


class _FakeProvider:
    def __init__(self, *, text: str = "", raises: Exception | None = None, delay: float = 0.0):
        self._text = text
        self._raises = raises
        self._delay = delay

    async def generate(self, messages, **kwargs):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises:
            raise self._raises
        return type("R", (), {"text": self._text})()


def _use_provider(monkeypatch, provider):
    monkeypatch.setattr(orchestrator, "get_provider", lambda name: provider)


def _use_timeout(monkeypatch, seconds: float):
    monkeypatch.setattr(
        orchestrator, "settings",
        dataclasses.replace(orchestrator.settings, orchestrator_timeout_s=seconds),
    )


def _kimi_json(**fields) -> str:
    base = {
        "skill_name": "vida_casal",
        "intent": "consulta de gasto",
        "context_hint": "gastos do mês",
        "confidence": 0.9,
    }
    base.update(fields)
    return json.dumps(base)


# --- roteamento correto via Kimi -------------------------------------------

def test_orch_01_roteamento_correto_vida(monkeypatch):
    _use_provider(monkeypatch, _FakeProvider(text=_kimi_json(skill_name="vida_casal")))
    _use_timeout(monkeypatch, 3.0)
    decision = asyncio.run(route("quanto gastei esse mês"))
    assert decision.agent == "vida"
    assert decision.skill_name == "vida_casal"
    assert decision.via_fallback is False


def test_orch_02_roteamento_correto_yaannk(monkeypatch):
    _use_provider(
        monkeypatch, _FakeProvider(text=_kimi_json(skill_name="yaannk_tecnico"))
    )
    _use_timeout(monkeypatch, 3.0)
    decision = asyncio.run(route("como funciona o agentic rag"))
    assert decision.agent == "yaannk"
    assert decision.skill_name == "yaannk_tecnico"
    assert decision.via_fallback is False


# --- quedas para o fallback --------------------------------------------------

def test_orch_03_timeout_do_kimi_cai_no_fallback(monkeypatch):
    _use_provider(monkeypatch, _FakeProvider(text=_kimi_json(), delay=0.2))
    _use_timeout(monkeypatch, 0.01)
    decision = asyncio.run(route("quanto gastei esse mês"))
    assert decision.via_fallback is True
    assert decision.skill_name == "vida_casal"  # classify_intent() acerta por keyword


def test_orch_04_json_malformado_nao_propaga(monkeypatch):
    _use_provider(monkeypatch, _FakeProvider(text="isso não é JSON"))
    _use_timeout(monkeypatch, 3.0)
    decision = asyncio.run(route("qualquer coisa"))
    assert decision.via_fallback is True


def test_orch_05_skill_name_invalido_cai_no_fallback(monkeypatch):
    _use_provider(
        monkeypatch, _FakeProvider(text=_kimi_json(skill_name="financeiro"))
    )
    _use_timeout(monkeypatch, 3.0)
    decision = asyncio.run(route("quanto gastei"))
    assert decision.via_fallback is True


def test_orch_provider_indisponivel_cai_no_fallback(monkeypatch):
    _use_provider(monkeypatch, _FakeProvider(raises=LLMError("kimi fora")))
    _use_timeout(monkeypatch, 3.0)
    decision = asyncio.run(route("quanto gastei"))
    assert decision.via_fallback is True


# --- pipeline controla a flag, não o orquestrador ---------------------------

def test_orch_06_flag_desligada_chama_classify_intent_direto(monkeypatch):
    monkeypatch.setattr(
        pipeline, "settings",
        dataclasses.replace(pipeline.settings, orchestrator_enabled=False),
    )

    called = {"route": False}

    async def fake_route(text, history=None):
        called["route"] = True
        raise AssertionError("route() não deveria ser chamado com a flag desligada")

    monkeypatch.setattr(pipeline, "orchestrator_route", fake_route)
    monkeypatch.setattr(pipeline, "classify_intent", lambda text: "vida_casal")

    skill_name = asyncio.run(pipeline._resolve_skill_name("quanto gastei"))
    assert skill_name == "vida_casal"
    assert called["route"] is False


def test_orch_09_via_fallback_nao_aparece_com_flag_desligada(monkeypatch):
    """Com a flag desligada, o `pipeline` nem instancia `OrchestratorDecision`
    — não há `via_fallback` para observar, porque `route()` nunca roda."""
    monkeypatch.setattr(
        pipeline, "settings",
        dataclasses.replace(pipeline.settings, orchestrator_enabled=False),
    )

    async def fail_if_called(text, history=None):
        raise AssertionError("route() não deveria ser instanciado")

    monkeypatch.setattr(pipeline, "orchestrator_route", fail_if_called)
    # Não levanta exceção = a garantia se sustenta.
    asyncio.run(pipeline._resolve_skill_name("o que está pendente"))


def test_orch_flag_ligada_usa_orquestrador(monkeypatch):
    monkeypatch.setattr(
        pipeline, "settings",
        dataclasses.replace(pipeline.settings, orchestrator_enabled=True),
    )

    async def fake_route(text, history=None):
        return OrchestratorDecision(
            agent="yaannk", intent="pergunta técnica", context_hint="",
            confidence=0.8, skill_name="yaannk_tecnico", via_fallback=False,
        )

    monkeypatch.setattr(pipeline, "orchestrator_route", fake_route)
    skill_name = asyncio.run(pipeline._resolve_skill_name("como funciona o pipeline"))
    assert skill_name == "yaannk_tecnico"


# --- pessoas/pendencias preservadas (Q2) ------------------------------------

def test_orch_07_pessoas_preservada_no_fallback():
    decision = _fallback("quem é a Bia")
    assert decision.skill_name == "pessoas"
    assert decision.agent == "default"
    assert decision.via_fallback is True


def test_orch_08_pendencias_preservada_no_fallback():
    decision = _fallback("o que está pendente")
    assert decision.skill_name == "pendencias"
    assert decision.agent == "default"
    assert decision.via_fallback is True


def test_orch_07b_pessoas_preservada_via_kimi(monkeypatch):
    _use_provider(monkeypatch, _FakeProvider(text=_kimi_json(skill_name="pessoas")))
    _use_timeout(monkeypatch, 3.0)
    decision = asyncio.run(route("quem é a Bia"))
    assert decision.skill_name == "pessoas"
    assert decision.via_fallback is False


# --- não conflita com o RoutingDecision do Router ---------------------------

def test_orch_10_nao_conflita_com_routing_decision_do_router():
    from app.services.llm import RoutingDecision

    # símbolos distintos, sem NameError ao importar os dois no mesmo módulo
    assert OrchestratorDecision is not RoutingDecision
    assert orchestrator.OrchestratorDecision is not RoutingDecision


# --- regressão: bugs históricos ---------------------------------------------

def test_orch_reg_01_pergunta_casual_vai_para_default():
    """Bug 4: falso positivo de RAG em conversa casual sem termo técnico nem
    financeiro — não deve rotear nem para yaannk nem para vida."""
    decision = _fallback("O Douglas prefere café ou chá?")
    assert decision.skill_name == "default"
    assert decision.agent == "default"


def test_orch_reg_02_pergunta_tecnica_preserva_skill_e_folder():
    """Bug 2: vazamento de contexto — a skill técnica precisa continuar
    resolvendo para yaannk_tecnico (que ancora o priority_folder do projeto)."""
    from app.services.skills import get_skill

    decision = _fallback("como funciona o pipeline do YaannkAgent")
    assert decision.skill_name == "yaannk_tecnico"
    skill = get_skill(decision.skill_name)
    assert skill["priority_folder"] == "01 - Projetos/Pessoal/YaannkAgent"


# --- confidence ausente / mal formada não derruba o parse -------------------

def test_confidence_ausente_usa_default(monkeypatch):
    payload = json.dumps({"skill_name": "default", "intent": "conversa"})
    _use_provider(monkeypatch, _FakeProvider(text=payload))
    _use_timeout(monkeypatch, 3.0)
    decision = asyncio.run(route("oi"))
    assert decision.via_fallback is False
    assert decision.confidence == 0.5
