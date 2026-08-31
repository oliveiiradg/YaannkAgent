"""Configuração compartilhada dos testes.

Define um ambiente mínimo e SEGURO antes de qualquer import de `app.*`
(app.config lê o ambiente no import). Nada aqui toca em serviços reais.
"""

import os

_TEST_ENV = {
    "APP_ENV": "test",
    "EVOLUTION_API_URL": "http://localhost:8080",
    "EVOLUTION_API_KEY": "test-evo-key",
    "EVOLUTION_INSTANCE_NAME": "test",
    "WEBHOOK_HOST": "127.0.0.1",
    "WEBHOOK_PORT": "5000",
    "WEBHOOK_SECRET": "test-webhook-secret-0123456789",
    "WEBHOOK_MAX_BODY_BYTES": "4096",
    "MESSAGE_MAX_CHARS": "500",
    "REPLAY_MAX_SKEW_SECONDS": "300",
    "ALLOWED_NUMBERS": "5521999999999",
    "VAULT_PATH": "/tmp/yaannk-test-vault",
    "OBSIDIAN_MCP_TOKEN": "",
    "LLM_TELEMETRY": "false",
}

for _k, _v in _TEST_ENV.items():
    os.environ.setdefault(_k, _v)

os.makedirs(_TEST_ENV["VAULT_PATH"], exist_ok=True)
