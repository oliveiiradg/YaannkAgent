"""Agregação determinística de gastos do Agent Vida (Fase B, Sessão 21).

Cobre o que o benchmark valida por `reply_equals` nos casos `cas-*`, mas sem
LLM e sem vault real: monta notas `Gastos - YYYY-MM.md` num tmp_path e checa
a string exata. Existe porque a Fase B entregou essa formatação ao Kimi, que
além de variar a cada rodada errou soma (cas-03: R$ 2.265,00 num total de
R$ 1.265,00).
"""

import asyncio
import datetime

import pytest

from app.agents import agent_vida
from app.agents.agent_vida import (
    AgentVida,
    _data_iso_registro,
    _parse_gastos_md,
    _periodo_da_pergunta,
)

_HEADER = (
    "| Data | Pessoa | Categoria | Descrição | Valor |\n"
    "|---|---|---|---|---|\n"
)

_SETEMBRO = _HEADER + (
    "| 02-09-26 | Alice | Moradia | aluguel | 800.00 |\n"
    "| 03-09-26 | Alice | Mercado | feira | 150.00 |\n"
    "| 04-09-26 | Bob | Mercado | mercadinho | 80.00 |\n"
    "| 05-09-26 | Bob | Transporte | uber | 33.50 |\n"
)
_AGOSTO = _HEADER + (
    "| 10-08-26 | Alice | Saúde | consulta | 200.00 |\n"
    "| 11-08-26 | Bob | Contas | internet | 90.00 |\n"
)


@pytest.fixture
def vault_gastos(tmp_path, monkeypatch):
    """Vault só com as notas de gasto — aponta `GASTOS_ROOT_OVERRIDE`, não
    `vault_search.settings.vault_path`: repontar a raiz global esvaziaria a
    perna TF-IDF do RRF (foi o que derrubou tec-16/tec-17/adv-08 na Sessão 20).
    """
    financas = tmp_path / "03 - Vida" / "Finanças"
    financas.mkdir(parents=True)
    (financas / "Gastos - 2026-09.md").write_text(_SETEMBRO, encoding="utf-8")
    (financas / "Gastos - 2026-08.md").write_text(_AGOSTO, encoding="utf-8")
    monkeypatch.setattr(agent_vida, "GASTOS_ROOT_OVERRIDE", str(tmp_path))
    # Data fixa: os casos falam "esse mês"/"mês passado".
    class _Data(datetime.date):
        @classmethod
        def today(cls):
            return cls(2026, 9, 15)
    monkeypatch.setattr(agent_vida.datetime, "date", _Data)
    return tmp_path


def _responder(message: str, autor: str | None = "Alice") -> str | None:
    r = asyncio.run(AgentVida()._try_expenses(message, autor))
    return r.text if r else None


# --- parsing ---------------------------------------------------------------

def test_parse_gastos_md_le_todas_as_linhas():
    registros = _parse_gastos_md(_SETEMBRO)
    assert len(registros) == 4
    assert registros[0] == {
        "data": "02-09-26", "pessoa": "Alice", "categoria": "Moradia",
        "descricao": "aluguel", "valor": 800.0,
    }


def test_parse_gastos_md_ignora_linha_sem_valor_numerico():
    nota = _SETEMBRO + "| 06-09-26 | Bob | Lazer | cinema | a combinar |\n"
    assert len(_parse_gastos_md(nota)) == 4


@pytest.mark.parametrize(
    "entrada,esperado",
    [
        ("05-09-26", "2026-09-05"),
        ("05-09-2026", "2026-09-05"),
        ("2026-09-05", "2026-09-05"),
        ("32-09-26", None),
        ("sem data", None),
    ],
)
def test_data_iso_registro(entrada, esperado):
    assert _data_iso_registro(entrada) == esperado


def test_periodo_ultimos_meses():
    meses, ini, fim, rotulo = _periodo_da_pergunta(
        datetime.date(2026, 9, 15), "quanto gastamos nos últimos 3 meses?"
    )
    assert meses == [(2026, 7), (2026, 8), (2026, 9)]
    assert (ini, fim) == ("2026-07-01", "2026-09-16")
    assert rotulo == "nos últimos 3 meses"


# --- formatação (mesmos templates do benchmark) ----------------------------

def test_total_do_casal_com_breakdown(vault_gastos):
    assert _responder("quanto gastamos esse mês?") == (
        "Neste mês, vocês gastaram R$ 1.063,50 (4 lançamentos).\n"
        "- Moradia: R$ 800,00\n"
        "- Mercado: R$ 230,00\n"
        "- Transporte: R$ 33,50"
    )


def test_filtro_por_categoria(vault_gastos):
    assert _responder("quanto gastamos em mercado esse mês?") == (
        "Neste mês, vocês gastaram R$ 230,00 em Mercado (2 lançamentos)."
    )


def test_categoria_com_um_lancamento_no_singular(vault_gastos):
    assert _responder("quanto gastamos com moradia esse mês?") == (
        "Neste mês, vocês gastaram R$ 800,00 em Moradia (1 lançamento)."
    )


def test_pessoa_citada_no_texto(vault_gastos):
    assert _responder("quanto a Alice gastou esse mês?") == (
        "Neste mês, Alice gastou R$ 950,00 (2 lançamentos).\n"
        "- Moradia: R$ 800,00\n"
        "- Mercado: R$ 150,00"
    )


def test_quanto_eu_gastei_usa_o_autor(vault_gastos):
    assert _responder("quanto eu gastei esse mês?", autor="Bob") == (
        "Neste mês, você gastou R$ 113,50 (2 lançamentos).\n"
        "- Mercado: R$ 80,00\n"
        "- Transporte: R$ 33,50"
    )


def test_quebra_por_pessoa(vault_gastos):
    assert _responder("quanto cada um gastou esse mês?") == (
        "Neste mês, por pessoa:\n"
        "- Alice: R$ 950,00 (2 lanç.)\n"
        "- Bob: R$ 113,50 (2 lanç.)\n"
        "Total: R$ 1.063,50"
    )


def test_mes_passado_le_a_nota_do_mes_certo(vault_gastos):
    assert _responder("total gasto no mês passado") == (
        "Em agosto/2026, vocês gastaram R$ 290,00 (2 lançamentos).\n"
        "- Saúde: R$ 200,00\n"
        "- Contas: R$ 90,00"
    )


def test_periodo_sem_nota_nao_soma_de_outro_mes(vault_gastos):
    """Julho não tem nota: a resposta não pode vazar o total de outro mês —
    a regressão que motivou tirar o Kimi do caminho normal."""
    resposta = _responder("quanto gastamos em julho?")
    assert resposta is not None
    assert "R$ 1.063,50" not in resposta and "R$ 290,00" not in resposta
