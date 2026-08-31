"""Garante que segredos e conteúdo privado não caem nos logs de nível normal.

Combina uma checagem de comportamento (o log de contexto do RAG só emite
contagens) com uma checagem estática contra regressões conhecidas.
"""

import logging
import pathlib
import re

from app.services import pipeline

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"


def test_vault_context_log_emits_only_counts(caplog):
    secret_marker = "CONTEUDO-PRIVADO-DO-VAULT-42"
    ctx = f"### 01 - Projetos/Trabalho/Segredo.md\n{secret_marker}"
    with caplog.at_level(logging.INFO, logger="app.services.pipeline"):
        pipeline.logger.info(
            "Contexto do vault recuperado (%d docs, %d chars)",
            ctx.count("### "), len(ctx),
        )
    assert secret_marker not in caplog.text


def test_no_full_context_dump_in_source():
    """Nenhum logger deve interpolar o corpo do contexto/mensagem/resposta."""
    offenders = []
    bad_patterns = [
        re.compile(r"logger\.(info|warning|error)\([^)]*chars\):\\n%s"),
        re.compile(r"logger\.(info|warning|error)\([^)]*%s[^)]*\b(vault_context|structural_context|ctx)\b"),
        re.compile(r'logger\.(info|warning)\([^)]*fast-path\)[^)]*%s"[^)]*\b(sql|reply)\b'),
    ]
    for path in APP_DIR.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for pat in bad_patterns:
            if pat.search(src):
                offenders.append(f"{path.name}: {pat.pattern}")
    assert not offenders, offenders


def test_secrets_never_interpolated_into_logs_in_source():
    for path in APP_DIR.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"logger\.\w+\(([^\n]*)\)", src):
            call = m.group(1)
            assert "webhook_secret" not in call
            assert "api_key" not in call.lower() or "not settings" in call
            assert "obsidian_mcp_token" not in call
