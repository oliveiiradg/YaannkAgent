"""Endpoints proativos consumidos pelos subworkflows do n8n."""

import asyncio

from app.routes import proactive


def test_mensagem_da_agenda_diz_se_tem_atendimento(monkeypatch):
    monkeypatch.setattr(
        proactive, "get_bia_agenda_today",
        lambda: [{"cliente": "Fulana", "hora": "15:00", "status": "confirmado"}],
    )

    resposta = asyncio.run(proactive.bia_agenda_today_message())

    assert resposta["has_appointments"] is True
    assert "Fulana" in resposta["message"]


def test_mensagem_da_agenda_sem_atendimento(monkeypatch):
    monkeypatch.setattr(proactive, "get_bia_agenda_today", lambda: [])

    resposta = asyncio.run(proactive.bia_agenda_today_message())

    assert resposta["has_appointments"] is False
    assert resposta["message"]
