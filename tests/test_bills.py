"""Parser determinístico de contas fixas (Grupo do casal) e seu fast-path."""

import dataclasses
import datetime

from app.services import vault_search
from app.services.bills import (
    answer_bills_query,
    get_contas_do_mes,
    parse_contas_fixas,
    _CONTAS_DIR,
)
from app.agents.agent_vida import _format_bills_template
from app.services.finance_patterns import BILLS_QUERY_RE, FINANCIAL_QUERY_STRICT_RE


def _usar_vault(monkeypatch, path):
    """`Settings` é `frozen=True` — troca o objeto `settings` visto por
    `vault_search` (só ele resolve `safe_vault_path`) por uma cópia com
    `vault_path` apontando pro vault de teste."""
    monkeypatch.setattr(
        vault_search, "settings",
        dataclasses.replace(vault_search.settings, vault_path=str(path)),
    )

_HEADER = "| Descrição | Valor | Vencimento | Status |\n|---|---|---|---|\n"


def _nota(corpo: str) -> str:
    return (
        "---\ntipo: nota\n---\n\n# Contas — Teste\n\n## Contas Fixas\n\n"
        + _HEADER
        + corpo
    )


def test_tabela_valida_normal():
    nota = _nota("| Aluguel | R$ 800,00 | dia 10 | ✅ |\n")
    rows = parse_contas_fixas(nota)
    assert rows == [
        {"descricao": "Aluguel", "valor": 800.0, "dia_vencimento": 10, "status": "pago"}
    ]


def test_multiplas_contas():
    nota = _nota(
        "| Aluguel | R$ 800,00 | dia 10 | ✅ |\n"
        "| Wifi | R$ 90,00 | dia 05 | ⏳ |\n"
        "| Tim | R$ 49,99 | dia 15 | ⏳ |\n"
    )
    rows = parse_contas_fixas(nota)
    assert [r["descricao"] for r in rows] == ["Aluguel", "Wifi", "Tim"]


def test_valor_com_cifrao():
    nota = _nota("| Tim | R$ 49,99 | dia 15 | ✅ |\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["valor"] == 49.99


def test_valor_com_milhar():
    nota = _nota("| Rodrigo (Moto) | R$ 1.090,00 | dia 05 | ✅ |\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["valor"] == 1090.0


def test_vencimento_dia_n():
    nota = _nota("| MEI | R$ 86,05 | dia 20 | ⏳ |\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["dia_vencimento"] == 20


def test_vencimento_formato_dd_mm():
    nota = _nota("| Claro | R$ 40,00 | 02/07 | ✅ Pago |\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["dia_vencimento"] == 2


def test_status_pago_bare_check():
    nota = _nota("| Aluguel | R$ 800,00 | dia 10 | ✅ |\n")
    assert parse_contas_fixas(nota)[0]["status"] == "pago"


def test_status_pago_com_palavra():
    nota = _nota("| Claro | R$ 40,00 | 02/07 | ✅ Pago |\n")
    assert parse_contas_fixas(nota)[0]["status"] == "pago"


def test_status_pago_so_palavra_sem_emoji():
    nota = _nota("| Claro | R$ 40,00 | 02/06 | Pago |\n")
    assert parse_contas_fixas(nota)[0]["status"] == "pago"


def test_status_pendente():
    nota = _nota("| Tim | R$ 49,99 | dia 15 | ⏳ |\n")
    assert parse_contas_fixas(nota)[0]["status"] == "pendente"


def test_status_confirmar():
    nota = _nota("| Inter | R$ 495,00 | 22/07 | ⚠️ Confirmar |\n")
    assert parse_contas_fixas(nota)[0]["status"] == "confirmar"


def test_status_vazio():
    nota = _nota("| Aluguel | R$ 800,00 | dia 10 |  |\n")
    assert parse_contas_fixas(nota)[0]["status"] == "desconhecido"


def test_linha_invalida_sem_valor_e_ignorada():
    nota = _nota(
        "| Aluguel | R$ 800,00 | dia 10 | ✅ |\n"
        "| Sem valor nenhum | | dia 10 | ✅ |\n"
    )
    rows = parse_contas_fixas(nota)
    assert [r["descricao"] for r in rows] == ["Aluguel"]


def test_tabela_sem_contas():
    nota = _nota("")
    assert parse_contas_fixas(nota) == []


def test_sem_tabela_reconhecivel():
    nota = "---\ntipo: nota\n---\n\n# Nada aqui\n\nSó texto solto, sem tabela.\n"
    assert parse_contas_fixas(nota) == []


def test_whitespace_extra_nao_quebra():
    nota = _nota("|  Aluguel   |   R$ 800,00  |  dia 10  |   ✅   |\n")
    rows = parse_contas_fixas(nota)
    assert rows == [
        {"descricao": "Aluguel", "valor": 800.0, "dia_vencimento": 10, "status": "pago"}
    ]


def test_nao_confunde_com_tabela_de_gastos_variaveis():
    """A tabela "Gastos Variáveis" também tem Descrição/Valor — não pode ser
    lida no lugar da "Contas Fixas" (falta Vencimento/Status)."""
    nota = (
        "---\ntipo: nota\n---\n\n# Contas — Teste\n\n## Contas Fixas\n\n"
        + _HEADER
        + "| Aluguel | R$ 800,00 | dia 10 | ✅ |\n\n"
        "## Gastos Variáveis\n\n"
        "| Data | Descrição | Valor | Categoria |\n|---|---|---|---|\n"
        "| 05/08 | Mercado | R$ 120,00 | Mercado |\n"
    )
    rows = parse_contas_fixas(nota)
    assert [r["descricao"] for r in rows] == ["Aluguel"]


def test_processamento_repetido_e_idempotente(tmp_path, monkeypatch):
    """Sem persistência SQL: parsear a mesma nota duas vezes devolve o mesmo
    resultado (não há estado acumulado entre chamadas)."""
    vault = tmp_path / "vault"
    # Caminho vem de `bills._CONTAS_DIR` — o teste apontava pro layout
    # antigo (`02 - Áreas/…`) e nenhuma nota era encontrada.
    (vault / _CONTAS_DIR).mkdir(parents=True)
    nota_path = vault / _CONTAS_DIR / "Contas - 2026-08.md"
    nota_path.write_text(_nota("| Aluguel | R$ 800,00 | dia 10 | ✅ |\n"), encoding="utf-8")

    _usar_vault(monkeypatch, vault)
    primeira = get_contas_do_mes(2026, 8)
    segunda = get_contas_do_mes(2026, 8)
    assert primeira == segunda


def test_mes_sem_nota_devolve_none(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    _usar_vault(monkeypatch, vault)
    assert get_contas_do_mes(2026, 9) is None


# --- fast-path ---------------------------------------------------------------


def _preparar_vault_com_agosto(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    # Caminho vem de `bills._CONTAS_DIR` — o teste apontava pro layout
    # antigo (`02 - Áreas/…`) e nenhuma nota era encontrada.
    (vault / _CONTAS_DIR).mkdir(parents=True)
    nota_path = vault / _CONTAS_DIR / "Contas - 2026-08.md"
    nota_path.write_text(
        _nota(
            "| Aluguel | R$ 800,00 | dia 10 | ✅ |\n"
            "| Wifi | R$ 90,00 | dia 05 | ⏳ |\n"
        ),
        encoding="utf-8",
    )
    _usar_vault(monkeypatch, vault)


def test_fastpath_contas_pendentes(tmp_path, monkeypatch):
    _preparar_vault_com_agosto(tmp_path, monkeypatch)
    hoje = datetime.date(2026, 8, 20)
    resposta = answer_bills_query("quais contas estão pendentes?", today=hoje)
    assert resposta is not None
    assert "Wifi" in _format_bills_template(resposta)
    assert "Aluguel" not in _format_bills_template(resposta)  # já paga, não deve aparecer como pendente


def test_fastpath_total_conta_fixa(tmp_path, monkeypatch):
    _preparar_vault_com_agosto(tmp_path, monkeypatch)
    hoje = datetime.date(2026, 8, 20)
    resposta = answer_bills_query("quanto tenho de conta fixa?", today=hoje)
    assert resposta is not None
    assert "890" in _format_bills_template(resposta) or "R$ 890,00" in _format_bills_template(resposta)


def test_fastpath_conta_especifica(tmp_path, monkeypatch):
    _preparar_vault_com_agosto(tmp_path, monkeypatch)
    hoje = datetime.date(2026, 8, 20)
    resposta = answer_bills_query("a conta de Aluguel já foi paga?", today=hoje)
    assert resposta is not None
    assert "paga" in _format_bills_template(resposta).lower()


def test_fastpath_nao_ativa_para_gasto_variavel():
    """"Quanto gastamos no mercado" é do domínio de `expenses.py`
    (FINANCIAL_QUERY_STRICT_RE), não de contas fixas."""
    assert BILLS_QUERY_RE.search("quanto gastamos no mercado esse mês") is None
    assert FINANCIAL_QUERY_STRICT_RE.search("quanto gastamos no mercado esse mês")


def test_fastpath_nao_ativa_para_perguntas_nao_financeiras():
    """Casos adversariais do benchmark: "falta" sozinho não deve acionar o
    fast-path de contas fixas."""
    assert answer_bills_query("quanto tempo falta pro nosso aniversário?") is None
    assert answer_bills_query("o que falta fazer no YaannkAgent?") is None


def test_fastpath_sem_nota_do_mes_cai_para_none(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    _usar_vault(monkeypatch, vault)
    hoje = datetime.date(2026, 9, 4)
    assert answer_bills_query("o que falta pagar esse mês?", today=hoje) is None


def test_fastpath_conta_nao_encontrada_por_nome_cai_para_none(tmp_path, monkeypatch):
    _preparar_vault_com_agosto(tmp_path, monkeypatch)
    hoje = datetime.date(2026, 8, 20)
    # A tabela de teste só tem "Aluguel" e "Wifi" — "internet" não bate por
    # substring (sem mapeamento de sinônimo, ver docstring de `_match_conta`).
    assert answer_bills_query("a conta de internet já foi paga?", today=hoje) is None
