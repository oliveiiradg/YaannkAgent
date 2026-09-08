"""Agentic RAG — loop de refinamento de busca (app/services/agentic_rag.py).

Tudo aqui é offline: o juiz (Kimi) e a busca híbrida (MCP + TF-IDF + reranker)
entram como dublês. O que se testa é o contrato do loop — parser da resposta do
juiz, acumulação de documentos e cada condição de parada — porque é isso que
garante a invariante: **o loop nunca devolve menos que a busca one-shot**.
"""

import asyncio
import types

import pytest

from app.services import agentic_rag
from app.services.agentic_rag import accumulate, parse_judge, refine


def _doc(name: str) -> dict:
    return {"filePath": f"{name}.md", "snippet": f"conteúdo de {name}"}


def _fake_settings(max_searches: int = 3):
    return types.SimpleNamespace(
        agentic_rag_enabled=True, agentic_rag_max_searches=max_searches
    )


# --- parser da resposta do juiz -------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "SUFICIENTE",
        "  suficiente  ",
        "`SUFICIENTE`",
        "SUFICIENTE — os trechos cobrem a pergunta",
        "",
        "   ",
        # Fora do contrato: parar é o lado seguro (mantém o one-shot de hoje).
        "Acho que talvez precise de mais contexto sobre o reranker",
        "BUSCAR:",
        "BUSCAR:   ",
    ],
)
def test_parse_judge_para_o_loop(text):
    assert parse_judge(text) is None


@pytest.mark.parametrize(
    "text,esperado",
    [
        ("BUSCAR: reranker top-k documentos", "reranker top-k documentos"),
        ("buscar: reranker top-k", "reranker top-k"),
        ('BUSCAR: "banco de dados do gateway"', "banco de dados do gateway"),
        ("\n\nBUSCAR: tier do router\n", "tier do router"),
    ],
)
def test_parse_judge_extrai_nova_query(text, esperado):
    assert parse_judge(text) == esperado


# --- acumulação ------------------------------------------------------------

def test_accumulate_preserva_ordem_e_deduplica():
    atuais = [_doc("a"), _doc("b")]
    novos = [_doc("b"), _doc("c")]
    merged = accumulate(atuais, novos, limit=6)
    assert [i["filePath"] for i in merged] == ["a.md", "b.md", "c.md"]


def test_accumulate_respeita_o_limite():
    atuais = [_doc("a"), _doc("b")]
    novos = [_doc("c"), _doc("d")]
    merged = accumulate(atuais, novos, limit=3)
    assert [i["filePath"] for i in merged] == ["a.md", "b.md", "c.md"]


def test_accumulate_nunca_encolhe():
    atuais = [_doc("a"), _doc("b")]
    # Mesmo com o limite já estourado, os itens que já estavam ficam.
    merged = accumulate(atuais, [_doc("c")], limit=1)
    assert [i["filePath"] for i in merged] == ["a.md", "b.md"]


# --- o loop ----------------------------------------------------------------

def _run_refine(monkeypatch, *, veredictos, buscas, max_searches=3):
    """Roda `refine` com juiz e busca dublados.

    `veredictos` é a fila de retornos de `_ask_judge` (None = suficiente);
    `buscas` é a fila de resultados de `search_vault_hybrid_ranked`.
    """
    monkeypatch.setattr(agentic_rag, "settings", _fake_settings(max_searches))
    chamadas = {"juiz": 0, "busca": 0}

    async def fake_judge(question, context, tried):
        chamadas["juiz"] += 1
        v = veredictos.pop(0)
        return v, "" if v else "contexto suficiente"

    async def fake_search(query, priority_folder, **kwargs):
        chamadas["busca"] += 1
        return buscas.pop(0)

    monkeypatch.setattr(agentic_rag, "_ask_judge", fake_judge)
    monkeypatch.setattr(agentic_rag, "search_vault_hybrid_ranked", fake_search)

    result = asyncio.run(
        refine("pergunta original", None, initial_items=[_doc("a"), _doc("b")])
    )
    return result, chamadas


def test_para_quando_o_contexto_e_suficiente(monkeypatch):
    result, chamadas = _run_refine(monkeypatch, veredictos=[None], buscas=[])
    assert result.stop_reason == "contexto suficiente"
    assert result.n_searches == 1
    assert chamadas["busca"] == 0
    assert [i["filePath"] for i in result.items] == ["a.md", "b.md"]


def test_refina_e_acumula_documentos_ineditos(monkeypatch):
    result, chamadas = _run_refine(
        monkeypatch,
        veredictos=["nova query", None],
        buscas=[[_doc("c"), _doc("d")]],
    )
    assert result.n_searches == 2
    assert result.queries == ["pergunta original", "nova query"]
    assert result.stop_reason == "contexto suficiente"
    # Ordem de inserção preservada: os docs da 1ª busca ficam na frente.
    assert [i["filePath"] for i in result.items] == ["a.md", "b.md", "c.md", "d.md"]


def test_respeita_o_teto_de_buscas(monkeypatch):
    result, chamadas = _run_refine(
        monkeypatch,
        veredictos=["q2", "q3", "q4"],
        buscas=[[_doc("c")], [_doc("d")], [_doc("e")]],
        max_searches=3,
    )
    assert result.n_searches == 3
    assert result.stop_reason == "teto de buscas"
    assert chamadas["busca"] == 2  # a inicial não é refeita aqui


def test_para_em_query_repetida(monkeypatch):
    result, _ = _run_refine(
        monkeypatch, veredictos=["  Pergunta   Original "], buscas=[]
    )
    assert result.stop_reason == "query repetida"
    assert result.n_searches == 1


def test_para_quando_nada_inedito_aparece(monkeypatch):
    result, _ = _run_refine(
        monkeypatch, veredictos=["outra query"], buscas=[[_doc("a"), _doc("b")]]
    )
    assert result.stop_reason == "nenhum documento inédito"
    assert [i["filePath"] for i in result.items] == ["a.md", "b.md"]


def test_falha_da_busca_nao_propaga(monkeypatch):
    monkeypatch.setattr(agentic_rag, "settings", _fake_settings(3))

    async def fake_judge(question, context, tried):
        return "outra query", ""

    async def fake_search(query, priority_folder, **kwargs):
        raise RuntimeError("MCP fora do ar")

    monkeypatch.setattr(agentic_rag, "_ask_judge", fake_judge)
    monkeypatch.setattr(agentic_rag, "search_vault_hybrid_ranked", fake_search)

    result = asyncio.run(refine("p", None, initial_items=[_doc("a")]))
    assert result.stop_reason == "falha na busca"
    assert [i["filePath"] for i in result.items] == ["a.md"]


def test_para_no_teto_de_documentos(monkeypatch):
    monkeypatch.setattr(agentic_rag, "settings", _fake_settings(9))
    iniciais = [_doc(c) for c in "abcdef"]  # já em _MAX_DOCS

    async def nunca(*args, **kwargs):
        raise AssertionError("não deveria consultar o juiz com o pool cheio")

    monkeypatch.setattr(agentic_rag, "_ask_judge", nunca)
    result = asyncio.run(refine("p", None, initial_items=iniciais))
    assert result.stop_reason == "teto de documentos"
    assert len(result.items) == agentic_rag._MAX_DOCS


def test_juiz_indisponivel_encerra_sem_excecao(monkeypatch):
    """`_ask_judge` devolve None quando o provider falha — o loop para e
    entrega o que tinha (invariante: nunca pior que o one-shot)."""
    from app.services.llm.base import LLMError

    monkeypatch.setattr(agentic_rag, "settings", _fake_settings(3))

    class FakeProvider:
        async def generate(self, messages, **kwargs):
            raise LLMError("kimi fora")

    monkeypatch.setattr(agentic_rag, "get_provider", lambda name: FakeProvider())
    monkeypatch.setattr(agentic_rag, "record_error", lambda *a, **k: None)

    result = asyncio.run(refine("p", None, initial_items=[_doc("a")]))
    assert result.stop_reason == "juiz indisponível"
    assert result.n_searches == 1
    assert [i["filePath"] for i in result.items] == ["a.md"]


# --- refatoração de vault_search_semantic ---------------------------------

def test_hybrid_e_a_composicao_de_ranked_com_build(monkeypatch):
    """`search_vault_hybrid()` (usada pelo ramo decomposto e pelos scripts de
    calibração) tem que continuar sendo exatamente ranked + build."""
    from app.services import vault_search_semantic as vss

    itens = [_doc("a"), _doc("b")]

    async def fake_ranked(query, priority_folder=None, **kwargs):
        return itens

    monkeypatch.setattr(vss, "search_vault_hybrid_ranked", fake_ranked)
    assert asyncio.run(vss.search_vault_hybrid("q")) == vss.build_context_blocks(itens)


def test_hybrid_vazio_devolve_string_vazia(monkeypatch):
    from app.services import vault_search_semantic as vss

    async def fake_ranked(query, priority_folder=None, **kwargs):
        return []

    monkeypatch.setattr(vss, "search_vault_hybrid_ranked", fake_ranked)
    assert asyncio.run(vss.search_vault_hybrid("q")) == ""
