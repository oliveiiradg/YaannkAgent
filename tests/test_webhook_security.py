"""Testes de segurança do webhook: autenticação, spoofing de remetente,
validação de payload, tamanho e proteção anti-replay."""

import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.routes import webhook as webhook_module

ALLOWED = "5521999999999"
OUTSIDER = "5511000000000"


@pytest.fixture
def client(monkeypatch):
    # Não deixa nenhum teste tocar Evolution/pipeline reais.
    async def _fake_send(number, text):
        return None

    async def _fake_answer(*a, **kw):
        from app.services.pipeline import PipelineResult

        return PipelineResult(
            reply="ok", source="llm", succeeded=True, request_id="x",
            skill_name="default", intent="conversation",
        )

    monkeypatch.setattr(webhook_module, "send_message", _fake_send)
    monkeypatch.setattr(webhook_module.pipeline, "answer", _fake_answer)
    monkeypatch.setattr(webhook_module, "add_message", lambda *a, **kw: None)

    from app.main import app

    return TestClient(app)


def _payload(text=" olá", sender=ALLOWED, ts=None, msg_id="m1"):
    return {
        "event": "messages.upsert",
        "instance": "test",
        "sender": "5521888888888@s.whatsapp.net",
        "data": {
            "key": {"remoteJid": f"{sender}@s.whatsapp.net", "fromMe": False, "id": msg_id},
            "pushName": "Tester",
            "message": {"conversation": text},
            "messageTimestamp": ts if ts is not None else int(time.time()),
        },
    }


def _hdr(secret=settings.webhook_secret):
    return {"X-Webhook-Secret": secret} if secret else {}


def test_missing_secret_is_rejected(client):
    r = client.post("/webhook", json=_payload())
    assert r.status_code == 401
    assert "secret" not in r.text.lower()  # erro genérico, não vaza motivo


def test_wrong_secret_is_rejected(client):
    r = client.post("/webhook", json=_payload(), headers={"X-Webhook-Secret": "nope"})
    assert r.status_code == 401


def test_valid_secret_passes_auth(client):
    r = client.post("/webhook", json=_payload(msg_id="ok-1"), headers=_hdr())
    assert r.status_code == 200


def test_sender_spoofing_outside_allowlist_is_ignored(client):
    r = client.post(
        "/webhook", json=_payload(sender=OUTSIDER, msg_id="spoof-1"), headers=_hdr()
    )
    assert r.status_code == 200
    assert r.json()["reason"] == "sender not allowed"


def test_oversized_payload_is_rejected(client):
    big = _payload(text="x" * 20000, msg_id="big-1")
    r = client.post("/webhook", json=big, headers=_hdr())
    assert r.status_code == 413


def test_message_too_long_is_ignored(client):
    long_text = "a" * (settings.message_max_chars + 50)
    # cabe no limite de corpo, estoura o limite de mensagem
    r = client.post("/webhook", json=_payload(text=long_text, msg_id="long-1"), headers=_hdr())
    assert r.status_code == 200
    assert r.json()["reason"] == "message too long"


def test_replay_old_event_is_rejected(client):
    old = int(time.time()) - (settings.replay_max_skew_seconds + 120)
    r = client.post("/webhook", json=_payload(ts=old, msg_id="replay-1"), headers=_hdr())
    assert r.status_code == 408


def test_malformed_payload_is_422_not_500(client):
    r = client.post("/webhook", json={"event": "messages.upsert"}, headers=_hdr())
    assert r.status_code == 422


def test_health_reveals_nothing(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_docs_disabled_in_non_dev(client):
    # APP_ENV=test → docs desligada por padrão
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
