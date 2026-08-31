"""Carrega e valida `benchmark/cases.jsonl` — Fase 8.

Um caso por linha. Cada caso declara só as chaves de `expect` que quer verificar
(ver `grader.py` para a semântica de cada uma).
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path

from app.services.expenses import _MES_NOME

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"

CATEGORIES = {"tecnico", "casal", "parsing", "adversarial"}

# Chaves aceitas em `expect`. O grader ignora silenciosamente nenhuma — chave
# desconhecida é erro de validação (typo em caso novo).
EXPECT_KEYS = {
    # determinístico (sem LLM)
    "skill", "skill_not", "intent", "tier", "fast_path",
    "save_tipo", "valor", "categoria", "data", "descricao_contains",
    # RAG (precisa retrieve())
    "rag_contains", "rag_not_contains",
    # resposta (precisa answer() + --llm)
    "answer_contains", "answer_any", "answer_not_contains",
    "reply_equals", "expect_nao_encontrado",
    # meta
    "autor",
}


@dataclass(frozen=True)
class Case:
    id: str
    cat: str
    input: str
    expect: dict
    xfail: str | None = None

    @property
    def autor(self) -> str | None:
        return self.expect.get("autor")


def _interp_map(today: datetime.date) -> dict[str, str]:
    """Placeholders de mês para casos que dependem da data em que a suíte roda —
    a fixture é relativa a hoje, então `reply_equals` não pode ter mês fixo.

    - `{mes_atual}` / `{mes_atual_ano}`    → "agosto" / "agosto/2026"
    - `{mes_passado}` / `{mes_passado_ano}` → "julho" / "julho/2026"
    """
    pm, py = (12, today.year - 1) if today.month == 1 else (today.month - 1, today.year)
    return {
        "{mes_atual}": _MES_NOME[today.month],
        "{mes_atual_ano}": f"{_MES_NOME[today.month]}/{today.year}",
        "{mes_passado}": _MES_NOME[pm],
        "{mes_passado_ano}": f"{_MES_NOME[pm]}/{py}",
    }


def _interp(value, mapping: dict[str, str]):
    if isinstance(value, str):
        for k, v in mapping.items():
            value = value.replace(k, v)
        return value
    if isinstance(value, dict):
        return {k: _interp(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp(v, mapping) for v in value]
    return value


def load_cases(
    path: Path | str = CASES_PATH, today: datetime.date | None = None
) -> list[Case]:
    path = Path(path)
    if not path.exists():
        return []
    mapping = _interp_map(today or datetime.date.today())
    cases: list[Case] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("//") or raw.startswith("#"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno}: JSON inválido — {exc}") from exc
        cases.append(
            Case(
                id=obj["id"],
                cat=obj["cat"],
                input=_interp(obj["input"], mapping),
                expect=_interp(obj.get("expect", {}), mapping),
                xfail=obj.get("xfail"),
            )
        )
    return cases


def validate(cases: list[Case]) -> list[str]:
    """Retorna lista de problemas (vazia = tudo certo)."""
    problems: list[str] = []
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            problems.append(f"id duplicado: {c.id}")
        seen.add(c.id)
        if c.cat not in CATEGORIES:
            problems.append(f"{c.id}: categoria inválida {c.cat!r}")
        if not c.input.strip():
            problems.append(f"{c.id}: input vazio")
        if not c.expect:
            problems.append(f"{c.id}: sem chaves em 'expect'")
        for k in c.expect:
            if k not in EXPECT_KEYS:
                problems.append(f"{c.id}: chave de expect desconhecida {k!r}")
    return problems
