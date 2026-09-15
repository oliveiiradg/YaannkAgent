"""Endpoints proativos consumidos pelos subworkflows do n8n."""

import asyncio
import dataclasses

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import proactive

_TOKEN = "token-de-teste-0123456789abcdef"


def _client(monkeypatch, token=_TOKEN):
    monkeypatch.setattr(
        proactive, "settings", dataclasses.replace(proactive.settings, proactive_token=token)
    )
    monkeypatch.setattr(proactive, "get_bia_agenda_today", lambda: [])
    app = FastAPI()
    app.include_router(proactive.router)
    return TestClient(app)


def test_rota_proativa_sem_token_ou_com_token_errado_e_recusada(monkeypatch):
    client = _client(monkeypatch)

    assert client.get("/proactive/bia-agenda-today-message").status_code == 401
    assert client.get(
        "/proactive/bia-agenda-today-message", headers={"x-proactive-token": "errado"}
    ).status_code == 401
    assert client.post("/proactive/create-next-month").status_code == 401


def test_rota_proativa_com_token_certo_responde(monkeypatch):
    client = _client(monkeypatch)

    resposta = client.get(
        "/proactive/bia-agenda-today-message", headers={"x-proactive-token": _TOKEN}
    )

    assert resposta.status_code == 200
    assert resposta.json()["has_appointments"] is False


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
