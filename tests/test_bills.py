"""Parser determinístico de contas fixas (Grupo do casal) e seu fast-path.

Formato de checkbox (D-10, Sessão 23) — ver docstring de `app.services.bills`.
"""

import asyncio
import dataclasses
import datetime

import pytest

from app.services import vault_search
from app.services.bills import (
    add_bill,
    answer_bills_query,
    create_next_month_note,
    get_contas_do_mes,
    mark_bill_paid,
    parse_contas_fixas,
    remove_bill,
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


def _nota(corpo: str) -> str:
    return (
        "---\ntipo: financas-mensal\nmes: 2026-09\ntags: [finanças, mensal]\n---\n\n"
        "# 📋 Contas — Setembro/2026\n\n" + corpo
    )


# --- parse_contas_fixas ------------------------------------------------------


def test_checkbox_pendente():
    nota = _nota("- [ ] Aluguel — R$ 800,00 — dia 10\n")
    rows = parse_contas_fixas(nota)
    assert rows == [
        {"descricao": "Aluguel", "valor": 800.0, "dia_vencimento": 10, "status": "pendente"}
    ]


def test_checkbox_pago():
    nota = _nota("- [x] Aluguel — R$ 800,00 — dia 10\n")
    assert parse_contas_fixas(nota)[0]["status"] == "pago"


def test_checkbox_pago_maiusculo():
    nota = _nota("- [X] Aluguel — R$ 800,00 — dia 10\n")
    assert parse_contas_fixas(nota)[0]["status"] == "pago"


def test_multiplas_contas():
    nota = _nota(
        "- [x] Aluguel — R$ 800,00 — dia 10\n"
        "- [ ] Wifi — R$ 90,00 — dia 5\n"
        "- [ ] Tim — R$ 49,99 — dia 15\n"
    )
    rows = parse_contas_fixas(nota)
    assert [r["descricao"] for r in rows] == ["Aluguel", "Wifi", "Tim"]


def test_valor_com_cifrao_e_milhar():
    nota = _nota("- [ ] Rodrigo (Moto) — R$ 1.090,00 — dia 5\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["valor"] == 1090.0


def test_vencimento_dia_n():
    nota = _nota("- [ ] MEI — R$ 86,05 — dia 20\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["dia_vencimento"] == 20


def test_sem_dia_vencimento():
    """Parcela já quitada às vezes não tem vencimento (ex.: "Wonder Foz
    5/6 (quitado)") — dia é opcional na linha."""
    nota = _nota("- [x] Wonder Foz 5/6 (quitado) — R$ 57,00\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["dia_vencimento"] is None
    assert rows[0]["valor"] == 57.0


def test_nome_com_parenteses_preservado():
    nota = _nota("- [ ] TIM (Douglas + Bia) — R$ 116,99 — dia 15\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["descricao"] == "TIM (Douglas + Bia)"


def test_linha_invalida_sem_valor_e_ignorada():
    nota = _nota(
        "- [ ] Aluguel — R$ 800,00 — dia 10\n"
        "- [ ] Sem valor nenhum\n"
    )
    rows = parse_contas_fixas(nota)
    assert [r["descricao"] for r in rows] == ["Aluguel"]


def test_tabela_sem_contas():
    assert parse_contas_fixas(_nota("")) == []


def test_texto_solto_nao_confunde():
    nota = "---\ntipo: nota\n---\n\n# Nada aqui\n\nSó texto solto, sem checkbox.\n"
    assert parse_contas_fixas(nota) == []


def test_nao_confunde_com_outra_lista_de_checkbox():
    """Uma lista de compras (`- [ ] item`, sem "R$") não deve ser lida como
    conta fixa — falta o valor monetário no formato esperado."""
    nota = _nota("- [ ] Aluguel — R$ 800,00 — dia 10\n") + "\n- [ ] Comprar pão\n"
    rows = parse_contas_fixas(nota)
    assert [r["descricao"] for r in rows] == ["Aluguel"]


def test_whitespace_extra_nao_quebra():
    nota = _nota("-   [ ]   Aluguel   —   R$ 800,00   —   dia 10  \n")
    rows = parse_contas_fixas(nota)
    assert rows == [
        {"descricao": "Aluguel", "valor": 800.0, "dia_vencimento": 10, "status": "pendente"}
    ]


def test_separador_hifen_tolerado():
    """Travessão (`—`) é o que o código sempre escreve, mas hífen simples
    (edição manual no Obsidian) também é aceito."""
    nota = _nota("- [ ] Aluguel - R$ 800,00 - dia 10\n")
    rows = parse_contas_fixas(nota)
    assert rows[0]["descricao"] == "Aluguel"
    assert rows[0]["valor"] == 800.0
    assert rows[0]["dia_vencimento"] == 10


def test_sufixo_data_conclusao_do_plugin_tasks_nao_derruba_a_linha():
    """Achado real (Sessão 23, nota de 2026-09): o plugin Tasks do Obsidian
    anexa `✅ AAAA-MM-DD` na linha quando o checkbox é marcado pela UI — com
    âncora de fim de linha no regex, isso derrubava 10 de 22 contas da nota
    inteira, silenciosamente."""
    nota = _nota("- [x] Faculdade — R$ 530,14 — dia 15 ✅ 2026-09-14\n")
    rows = parse_contas_fixas(nota)
    assert rows == [
        {"descricao": "Faculdade", "valor": 530.14, "dia_vencimento": 15, "status": "pago"}
    ]


def test_processamento_repetido_e_idempotente(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / _CONTAS_DIR).mkdir(parents=True)
    nota_path = vault / _CONTAS_DIR / "Contas - 2026-08.md"
    nota_path.write_text(_nota("- [x] Aluguel — R$ 800,00 — dia 10\n"), encoding="utf-8")

    _usar_vault(monkeypatch, vault)
    primeira = get_contas_do_mes(2026, 8)
    segunda = get_contas_do_mes(2026, 8)
    assert primeira == segunda


def test_mes_sem_nota_devolve_none(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    _usar_vault(monkeypatch, vault)
    assert get_contas_do_mes(2026, 9) is None


# --- fast-path (leitura) ------------------------------------------------------


def _preparar_vault_com_agosto(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / _CONTAS_DIR).mkdir(parents=True)
    nota_path = vault / _CONTAS_DIR / "Contas - 2026-08.md"
    nota_path.write_text(
        _nota(
            "- [x] Aluguel — R$ 800,00 — dia 10\n"
            "- [ ] Wifi — R$ 90,00 — dia 5\n"
        ),
        encoding="utf-8",
    )
    _usar_vault(monkeypatch, vault)
    return vault


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
    assert BILLS_QUERY_RE.search("quanto gastamos no mercado esse mês") is None
    assert FINANCIAL_QUERY_STRICT_RE.search("quanto gastamos no mercado esse mês")


def test_fastpath_nao_ativa_para_perguntas_nao_financeiras():
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
    assert answer_bills_query("a conta de internet já foi paga?", today=hoje) is None


# --- write-path (mark_bill_paid / remove_bill / add_bill) --------------------


class _FakeMCP:
    """Substitui `bills._mcp_call` — grava em memória em vez de chamar o MCP
    real, pra testar o write-path sem depender do Obsidian estar rodando."""

    def __init__(self):
        self.escritas: dict[str, str] = {}

    async def __call__(self, name, arguments):
        assert name == "create_vault_file"
        self.escritas[arguments["path"]] = arguments["content"]
        return {}


@pytest.fixture
def vault_setembro(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / _CONTAS_DIR).mkdir(parents=True)
    nota_path = vault / _CONTAS_DIR / "Contas - 2026-09.md"
    nota_path.write_text(
        _nota(
            "- [x] Aluguel — R$ 1.650,00 — dia 5\n"
            "- [ ] Faculdade — R$ 530,14 — dia 15\n"
            "- [ ] TIM (Douglas + Bia) — R$ 116,99 — dia 15\n"
            "- [x] Wonder Foz 5/6 (quitado) — R$ 57,00\n"
        ),
        encoding="utf-8",
    )
    _usar_vault(monkeypatch, vault)
    return nota_path


def test_mark_bill_paid_marca_checkbox(vault_setembro, monkeypatch):
    import app.services.bills as bills_mod
    fake = _FakeMCP()
    monkeypatch.setattr(bills_mod, "_mcp_call", fake)

    resultado = asyncio.run(mark_bill_paid("Faculdade", "2026-09"))
    assert resultado["success"] is True
    assert "Faculdade" in resultado["message"]

    novo_content = next(iter(fake.escritas.values()))
    contas = parse_contas_fixas(novo_content)
    faculdade = next(c for c in contas if c["descricao"] == "Faculdade")
    assert faculdade["status"] == "pago"
    # As outras linhas continuam intactas, inclusive valor e dia.
    assert faculdade["valor"] == 530.14
    assert faculdade["dia_vencimento"] == 15
    tim = next(c for c in contas if "TIM" in c["descricao"])
    assert tim["status"] == "pendente"


def test_mark_bill_paid_ja_paga(vault_setembro, monkeypatch):
    import app.services.bills as bills_mod
    monkeypatch.setattr(bills_mod, "_mcp_call", _FakeMCP())

    resultado = asyncio.run(mark_bill_paid("Aluguel", "2026-09"))
    assert resultado["success"] is False
    assert "já está marcada como paga" in resultado["message"]


def test_mark_bill_paid_nao_encontrada(vault_setembro, monkeypatch):
    import app.services.bills as bills_mod
    monkeypatch.setattr(bills_mod, "_mcp_call", _FakeMCP())

    resultado = asyncio.run(mark_bill_paid("Netflix", "2026-09"))
    assert resultado["success"] is False
    assert "não encontrada" in resultado["message"]


def test_remove_bill(vault_setembro, monkeypatch):
    import app.services.bills as bills_mod
    fake = _FakeMCP()
    monkeypatch.setattr(bills_mod, "_mcp_call", fake)

    resultado = asyncio.run(remove_bill("TIM", "2026-09"))
    assert resultado["success"] is True

    novo_content = next(iter(fake.escritas.values()))
    contas = parse_contas_fixas(novo_content)
    assert all("TIM" not in c["descricao"] for c in contas)
    # As outras 3 contas continuam.
    assert len(contas) == 3


def test_add_bill(vault_setembro, monkeypatch):
    import app.services.bills as bills_mod
    fake = _FakeMCP()
    monkeypatch.setattr(bills_mod, "_mcp_call", fake)

    resultado = asyncio.run(add_bill("Academia", 80.0, 10, "2026-09"))
    assert resultado["success"] is True

    novo_content = next(iter(fake.escritas.values()))
    contas = parse_contas_fixas(novo_content)
    academia = next(c for c in contas if c["descricao"] == "Academia")
    assert academia["valor"] == 80.0
    assert academia["dia_vencimento"] == 10
    assert academia["status"] == "pendente"
    assert len(contas) == 5


def test_create_next_month_note_tudo_vira_pendente(vault_setembro, monkeypatch):
    import app.services.bills as bills_mod
    fake = _FakeMCP()
    monkeypatch.setattr(bills_mod, "_mcp_call", fake)

    resultado = asyncio.run(create_next_month_note(2026, 9))
    assert resultado["success"] is True
    assert resultado["month"] == "2026-10"

    novo_content = next(iter(fake.escritas.values()))
    contas = parse_contas_fixas(novo_content)
    assert len(contas) == 4
    assert all(c["status"] == "pendente" for c in contas)
    faculdade = next(c for c in contas if c["descricao"] == "Faculdade")
    assert faculdade["valor"] == 530.14


def test_create_next_month_note_recusa_sobrescrever(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / _CONTAS_DIR).mkdir(parents=True)
    (vault / _CONTAS_DIR / "Contas - 2026-09.md").write_text(
        _nota("- [ ] Aluguel — R$ 800,00 — dia 10\n"), encoding="utf-8",
    )
    (vault / _CONTAS_DIR / "Contas - 2026-10.md").write_text(
        _nota("- [ ] Já existe — R$ 1,00 — dia 1\n"), encoding="utf-8",
    )
    _usar_vault(monkeypatch, vault)

    import app.services.bills as bills_mod
    monkeypatch.setattr(bills_mod, "_mcp_call", _FakeMCP())

    resultado = asyncio.run(create_next_month_note(2026, 9))
    assert resultado["success"] is False
    assert "já existe" in resultado["message"]
