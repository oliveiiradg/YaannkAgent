"""Agenda da Bia (D-13): formato por dia, leitura dos dois formatos e migração.

Vault temporário e MCP mockado — nada toca o vault de produção.
"""

import asyncio
import datetime

import pytest

from app.services import agenda_bia

_TABELA = """---
tipo: agenda-bia
mes: 2026-09
---

# Agenda — Setembro/2026

| Cliente | Data | Dia | Hora | Status |
|---|---|---|---|---|
| Ana | 15/09 | terça | 18:00 | cancelado |
| Carla | 16/09 | quarta | 09:30 | confirmado |
| Ana | 16/09 | quarta | 08:00 | confirmado |
| Duda | 17/09 | quinta | 11:30 | confirmado |
| Eva | 17/09 | quinta | 18:00 | confirmado |
| Beto | 18/09 | sexta | 18:30 | confirmado |
"""

_CALENDARIO = """---
tipo: agenda-bia
mes: 2026-09
---

# Agenda — Setembro/2026

## Terça, 15/09
- ~~18:00 — Ana~~ (cancelado)

## Quarta, 16/09
- 08:00 — Ana
- 09:30 — Carla

## Quinta, 17/09
- 11:30 — Duda
- 18:00 — Eva

## Sexta, 18/09
- 18:30 — Beto
"""

_REL = "02 - Áreas/Bia/Agenda - 2026-09.md"


@pytest.fixture
def vault(monkeypatch, tmp_path):
    def _path(relpath, **kw):
        return tmp_path / relpath

    async def _mcp(tool, args):
        assert tool == "create_vault_file"
        destino = tmp_path / args["path"]
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(args["content"], encoding="utf-8")

    monkeypatch.setattr(agenda_bia, "safe_vault_path", _path)
    monkeypatch.setattr(agenda_bia, "_mcp_call", _mcp)

    def _escreve(content):
        nota = tmp_path / _REL
        nota.parent.mkdir(parents=True, exist_ok=True)
        nota.write_text(content, encoding="utf-8")
        return nota

    return _escreve


def _chaves(entradas):
    return sorted((e["data"], e["hora"], e["cliente"], e["status"]) for e in entradas)


def test_migracao_gera_calendario_por_dia_e_hora():
    assert agenda_bia.migrar_nota(_TABELA, 2026) == _CALENDARIO


def test_migracao_e_idempotente():
    assert agenda_bia.migrar_nota(_CALENDARIO, 2026) == _CALENDARIO


def test_parser_le_os_dois_formatos_igual():
    tabela = agenda_bia.parse_agenda(_TABELA, 2026)
    calendario = agenda_bia.parse_agenda(_CALENDARIO, 2026)
    assert _chaves(tabela) == _chaves(calendario)
    assert len(calendario) == 6
    assert {e["dia_semana"] for e in calendario if e["data"] == "16/09"} == {"quarta"}


def test_parser_aceita_edicao_a_mao_no_obsidian():
    nota = "## quarta 16/09\n- 9:05 - Fulana de Tal\n* ~~10:00 – Beltrana~~\n"
    assert _chaves(agenda_bia.parse_agenda(nota, 2026)) == [
        ("16/09", "09:05", "Fulana de Tal", "confirmado"),
        ("16/09", "10:00", "Beltrana", "cancelado"),
    ]


def test_agenda_do_dia_ignora_cancelados(vault):
    vault(_CALENDARIO)
    dia = agenda_bia.get_agenda_dia(datetime.date(2026, 9, 15))
    assert dia == []
    dia = agenda_bia.get_agenda_dia(datetime.date(2026, 9, 17))
    assert [(e["hora"], e["cliente"]) for e in dia] == [("11:30", "Duda"), ("18:00", "Eva")]


def test_agendar_em_nota_antiga_migra_e_insere_na_ordem(vault):
    nota = vault(_TABELA)
    r = asyncio.run(agenda_bia.add_agendamento("Fabi", datetime.date(2026, 9, 16), "09:00"))
    assert r["success"] is True
    assert "## Quarta, 16/09\n- 08:00 — Ana\n- 09:00 — Fabi\n- 09:30 — Carla\n" in nota.read_text()
    assert "|" not in nota.read_text()


def test_agendar_dia_novo_cria_titulo_na_ordem_de_data(vault):
    nota = vault(_CALENDARIO)
    asyncio.run(agenda_bia.add_agendamento("Gabi", datetime.date(2026, 9, 17), "08:00"))
    asyncio.run(agenda_bia.add_agendamento("Helo", datetime.date(2026, 9, 1), "10:00"))
    asyncio.run(agenda_bia.add_agendamento("Iris", datetime.date(2026, 9, 30), "10:00"))
    texto = nota.read_text()
    assert "## Quinta, 17/09\n- 08:00 — Gabi\n- 11:30 — Duda" in texto
    assert texto.index("## Terça, 01/09") < texto.index("## Terça, 15/09")
    assert texto.endswith("## Quarta, 30/09\n- 10:00 — Iris\n")


def test_agendar_em_mes_sem_nota_cria_nota_no_formato_novo(vault, tmp_path):
    r = asyncio.run(agenda_bia.add_agendamento("Ana", datetime.date(2026, 10, 2), "14:00"))
    assert r["success"] is True
    texto = (tmp_path / "02 - Áreas/Bia/Agenda - 2026-10.md").read_text()
    assert texto.endswith("# Agenda — Outubro/2026\n\n## Sexta, 02/10\n- 14:00 — Ana\n")


def test_conflito_no_mesmo_horario_nao_escreve(vault):
    nota = vault(_CALENDARIO)
    r = asyncio.run(agenda_bia.add_agendamento("Zé", datetime.date(2026, 9, 16), "09:30"))
    assert r["conflict"] is True
    assert nota.read_text() == _CALENDARIO


def test_horario_de_cancelado_fica_livre(vault):
    vault(_CALENDARIO)
    r = asyncio.run(agenda_bia.add_agendamento("Zé", datetime.date(2026, 9, 15), "18:00"))
    assert r["success"] is True


def test_cancelar_risca_a_linha(vault):
    nota = vault(_TABELA)
    r = asyncio.run(agenda_bia.cancel_agendamento("carla", datetime.date(2026, 9, 16)))
    assert r == {"success": True, "message": "Agendamento de Carla cancelado"}
    assert "- ~~09:30 — Carla~~ (cancelado)" in nota.read_text()
    assert agenda_bia.get_agenda_dia(datetime.date(2026, 9, 16))[0]["cliente"] == "Ana"


def test_cancelar_inexistente_nao_escreve(vault):
    nota = vault(_TABELA)
    r = asyncio.run(agenda_bia.cancel_agendamento("Ninguém", datetime.date(2026, 9, 16)))
    assert r["success"] is False
    assert nota.read_text() == _TABELA
