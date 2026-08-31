"""Configuração segura (produção falha sem secret) e sandbox de filesystem."""

import pathlib
import subprocess
import sys
import textwrap

import pytest

from app.services.vault_search import safe_vault_path


def _run_config(env_extra: dict) -> subprocess.CompletedProcess:
    code = textwrap.dedent(
        """
        from app.config import _load_settings
        _load_settings()
        print("OK")
        """
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(pathlib.Path(__file__).resolve().parent.parent),
        "EVOLUTION_API_URL": "http://localhost:8080",
        "EVOLUTION_API_KEY": "k",
        "EVOLUTION_INSTANCE_NAME": "i",
        "VAULT_PATH": "/tmp/yaannk-test-vault",
        # Sobrescreve explicitamente (vazio) para o load_dotenv() de app.config
        # não puxar valores do .env real do projeto.
        "APP_ENV": "development",
        "WEBHOOK_SECRET": "",
        "ALLOWED_NUMBERS": "",
        "ALLOWED_LIDS": "",
        **env_extra,
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=pathlib.Path(__file__).resolve().parent,
    )


def test_production_without_secret_fails_to_start():
    r = _run_config({"APP_ENV": "production", "ALLOWED_NUMBERS": "5521999999999"})
    assert r.returncode != 0
    assert "WEBHOOK_SECRET" in r.stderr


def test_production_without_allowlist_fails_to_start():
    r = _run_config(
        {"APP_ENV": "production", "WEBHOOK_SECRET": "x" * 32}
    )
    assert r.returncode != 0
    assert "ALLOWED_NUMBERS" in r.stderr


def test_production_with_weak_secret_fails():
    r = _run_config(
        {"APP_ENV": "production", "WEBHOOK_SECRET": "short", "ALLOWED_NUMBERS": "5521999999999"}
    )
    assert r.returncode != 0


def test_production_valid_config_starts():
    r = _run_config(
        {"APP_ENV": "production", "WEBHOOK_SECRET": "x" * 32, "ALLOWED_NUMBERS": "5521999999999"}
    )
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_dev_without_secret_is_allowed():
    r = _run_config({"APP_ENV": "development"})
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize(
    "evil",
    [
        "../../etc/passwd",
        "../.env",
        "../../../../../../etc/shadow",
        "/etc/passwd",
        "notas/../../../secret.txt",
        "foo\x00.md",
    ],
)
def test_path_traversal_is_blocked(evil):
    assert safe_vault_path(evil) is None


def test_normal_relative_path_inside_vault_is_allowed():
    from pathlib import Path

    from app.config import settings

    root = Path(settings.vault_path)
    (root / "sub").mkdir(parents=True, exist_ok=True)
    f = root / "sub" / "nota.md"
    f.write_text("oi")
    assert safe_vault_path("sub/nota.md") == f.resolve()
