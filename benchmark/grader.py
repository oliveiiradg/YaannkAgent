"""Avaliação dos casos do benchmark — Fase 8.

Três níveis, do mais barato ao mais caro:

- `grade_deterministic(case)` — só regex/dict/SQL (classify_intent, route,
  helpers de parsing, fast-path). Sem LLM, sem rede além do SQLite de fixture.
- `grade_retrieval(case, retrieval)` — precisa de `pipeline.retrieve()` (Bloco
  1/2 + RAG; Ollama + MCP, sem Kimi).
- `grade_answer(case, result)` — precisa de `pipeline.answer()` (Bloco 3, Kimi).

`judge()` é um hook para LLM-as-judge (`--judge`); não está ligado ainda.
"""

import datetime
import re
import unicodedata
from dataclasses import dataclass

from app.services.expenses import query_expenses
from app.services.finance_patterns import FINANCIAL_QUERY_STRICT_RE
from app.services.intent_classifier import classify_intent
from app.services.llm import route
from app.services.query_decomposer import decompose_query
from app.services.vault_writer import (
    _REL_DATE_RE,
    _categoria,
    _data_iso_gasto,
    _data_relativa,
    _limpa_descricao,
    _valor_float,
    detect_save_intent,
)
from benchmark.fixture import DEFAULT_AUTOR, FIXTURE_CHAT
from benchmark.loader import Case

DET_KEYS = {
    "skill", "skill_not", "intent", "tier", "fast_path",
    "save_tipo", "valor", "categoria", "data", "descricao_contains",
}
RETRIEVAL_KEYS = {"rag_contains", "rag_not_contains"}
ANSWER_KEYS = {
    "answer_contains", "answer_any", "answer_not_contains",
    "reply_equals", "expect_nao_encontrado",
}

# Checks sobre o texto livre do LLM: variam entre rodadas, dependem de infra
# (Ollama para o Bloco 1) e do humor do modelo. Entram na tabela como métrica
# observada, mas NÃO contam para o exit code (decisão Fase 8). `reply_equals`
# fica FORA daqui — é resposta determinística do fast-path SQL, gate duro.
ADVISORY_KEYS = {
    "answer_contains", "answer_any", "answer_not_contains", "expect_nao_encontrado",
}


@dataclass
class Check:
    key: str
    ok: bool
    expected: object
    got: object

    def line(self) -> str:
        flag = "ok  " if self.ok else "FAIL"
        return f"[{flag}] {self.key}: esperado={self.expected!r} obtido={self.got!r}"


@dataclass
class CaseResult:
    case: Case
    checks: list[Check]
    skipped: list[str]

    @property
    def hard_checks(self) -> list[Check]:
        return [c for c in self.checks if c.key not in ADVISORY_KEYS]

    @property
    def advisory_checks(self) -> list[Check]:
        return [c for c in self.checks if c.key in ADVISORY_KEYS]

    @property
    def ran(self) -> bool:
        return bool(self.hard_checks)

    @property
    def passed(self) -> bool:
        """Só os checks de gate duro — advisory (answer_*) não conta."""
        return self.ran and all(c.ok for c in self.hard_checks)

    @property
    def failed(self) -> bool:
        return self.ran and not self.passed

    @property
    def is_xfail(self) -> bool:
        return self.case.xfail is not None


def _as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def _norm(s: str) -> str:
    """Normaliza para o match de `answer_*`: minúsculas, sem acento, e
    hífen/underscore/barra/espaços viram um espaço só. Assim `"top-3"` casa com
    `"top 3"`, `"NÃO ENCONTRADO"` com `"nao encontrado"`, etc. Não faz sinônimo —
    só tira ruído de formatação (RC5, Fase 9)."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[\s\-_/]+", " ", s).strip()


def _contains_all(haystack: str, needles) -> bool:
    h = _norm(haystack)
    return all(_norm(n) in h for n in _as_list(needles))


def _contains_any(haystack: str, needles) -> bool:
    h = _norm(haystack)
    return any(_norm(n) in h for n in _as_list(needles))


def _expected_date(value: str) -> str:
    """Resolve o `expect.data` de um caso. Aceita ISO literal (`2026-08-12`) ou
    um token relativo (`hoje`, `ontem`, `anteontem`, `semana passada`,
    `mês passado`) — assim o caso não precisa hardcodar a data do dia em que a
    suíte roda. A resolução usa o mesmo helper do código de produção."""
    if _REL_DATE_RE.fullmatch(value.strip()):
        return _data_relativa(value, datetime.date.today()).isoformat()
    return value


# --------------------------------------------------------------------------- #
# nível 1 — determinístico
# --------------------------------------------------------------------------- #
def grade_deterministic(case: Case) -> list[Check]:
    e = case.expect
    text = case.input
    autor = case.autor or DEFAULT_AUTOR
    out: list[Check] = []

    if "skill" in e:
        got = classify_intent(text)
        out.append(Check("skill", got == e["skill"], e["skill"], got))

    if "skill_not" in e:
        got = classify_intent(text)
        out.append(Check("skill_not", got != e["skill_not"], f"!= {e['skill_not']}", got))

    if "intent" in e or "tier" in e:
        d = route(text, classify_intent(text))
        if "intent" in e:
            out.append(Check("intent", d.intent == e["intent"], e["intent"], d.intent))
        if "tier" in e:
            out.append(Check("tier", d.tier == e["tier"], e["tier"], d.tier))

    if "fast_path" in e:
        # Espelha os gates de `pipeline.answer`: o branch de registro
        # (`detect_save_intent`) responde antes; e uma pergunta multi-item é
        # decomposta em vez de fast-path.
        fired = (
            detect_save_intent(text) is None
            and len(decompose_query(text)) == 1
            and bool(FINANCIAL_QUERY_STRICT_RE.search(text))
            and query_expenses(FIXTURE_CHAT, text, autor=autor) is not None
        )
        out.append(Check("fast_path", fired == bool(e["fast_path"]), e["fast_path"], fired))

    if "save_tipo" in e:
        got = detect_save_intent(text)
        out.append(Check("save_tipo", got == e["save_tipo"], e["save_tipo"], got))

    if "valor" in e:
        got = _valor_float(text)
        ok = got is not None and abs(got - float(e["valor"])) < 0.005
        out.append(Check("valor", ok, e["valor"], got))

    if "categoria" in e:
        got = _categoria(text)
        out.append(Check("categoria", got == e["categoria"], e["categoria"], got))

    if "data" in e:
        got = _data_iso_gasto(text)
        want = _expected_date(e["data"])
        out.append(Check("data", got == want, want, got))

    if "descricao_contains" in e:
        got = _limpa_descricao(text)
        ok = _contains_all(got, e["descricao_contains"])
        out.append(Check("descricao_contains", ok, e["descricao_contains"], got))

    return out


# --------------------------------------------------------------------------- #
# nível 2 — retrieval (pipeline.retrieve)
# --------------------------------------------------------------------------- #
def grade_retrieval(case: Case, retrieval) -> list[Check]:
    e = case.expect
    files = retrieval.rag_files
    out: list[Check] = []

    if "rag_contains" in e:
        ok = all(any(sub.lower() in f.lower() for f in files) for sub in _as_list(e["rag_contains"]))
        out.append(Check("rag_contains", ok, e["rag_contains"], files))

    if "rag_not_contains" in e:
        ok = not any(
            sub.lower() in f.lower() for f in files for sub in _as_list(e["rag_not_contains"])
        )
        out.append(Check("rag_not_contains", ok, e["rag_not_contains"], files))

    return out


# --------------------------------------------------------------------------- #
# nível 3 — resposta (pipeline.answer)
# --------------------------------------------------------------------------- #
def grade_answer(case: Case, result) -> list[Check]:
    e = case.expect
    reply = result.reply or ""
    out: list[Check] = []

    if "reply_equals" in e:
        ok = reply.strip() == str(e["reply_equals"]).strip()
        out.append(Check("reply_equals", ok, e["reply_equals"], reply))

    if "answer_contains" in e:
        out.append(Check(
            "answer_contains", _contains_all(reply, e["answer_contains"]),
            e["answer_contains"], reply,
        ))

    if "answer_any" in e:
        out.append(Check(
            "answer_any", _contains_any(reply, e["answer_any"]),
            e["answer_any"], reply,
        ))

    if "answer_not_contains" in e:
        out.append(Check(
            "answer_not_contains", not _contains_any(reply, e["answer_not_contains"]),
            e["answer_not_contains"], reply,
        ))

    if "expect_nao_encontrado" in e:
        # aceita "NÃO ENCONTRADO" literal ou "não encontr(ei/ou)" / "não consta"
        # / "não está document(ado)" — o prompt pede o literal, mas o guard não
        # deve falhar por variação de fraseado do modelo.
        n = _norm(reply)
        got = any(p in n for p in (
            "nao encontrado", "nao encontr", "nao consta", "nao esta document",
            "nao ha informacao", "nao foi document",
        ))
        out.append(Check(
            "expect_nao_encontrado", got == bool(e["expect_nao_encontrado"]),
            e["expect_nao_encontrado"], got,
        ))

    return out


# --------------------------------------------------------------------------- #
# hook — LLM-as-judge (--judge). Não ligado ainda.
# --------------------------------------------------------------------------- #
def judge(case: Case, result, *, enabled: bool = False) -> Check | None:
    if not enabled:
        return None
    return Check("judge", ok=True, expected="score>=3", got="not-wired")
