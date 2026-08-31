"""Runner do benchmark de 60 casos — Fase 8.

    venv/bin/python scripts/benchmark.py                  # só camada determinística (grátis, rápido)
    venv/bin/python scripts/benchmark.py --llm            # + retrieve()/answer() reais (Ollama + MCP + Kimi)
    venv/bin/python scripts/benchmark.py --llm --judge    # + LLM-as-judge (hook, ainda não ligado)
    venv/bin/python scripts/benchmark.py --cat parsing --verbose
    venv/bin/python scripts/benchmark.py --llm --save baseline.json
    venv/bin/python scripts/benchmark.py --llm --diff  baseline.json

Exit 0 se a taxa determinística >= limiar (--threshold, default 0.95), sem
falhas não-xfail e sem regressão vs --diff. Caso contrário exit 1.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.fixture import DEFAULT_AUTOR, FIXTURE_CHAT, use_fixture_db  # noqa: E402
from benchmark.grader import (  # noqa: E402
    ADVISORY_KEYS,
    ANSWER_KEYS,
    DET_KEYS,
    RETRIEVAL_KEYS,
    CaseResult,
    grade_answer,
    grade_deterministic,
    grade_retrieval,
    judge,
)
from benchmark.loader import CATEGORIES, load_cases, validate  # noqa: E402


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark do pipeline Yaannk (Fase 8)")
    p.add_argument("--llm", action="store_true", help="roda o pipeline real (Ollama + MCP + Kimi)")
    p.add_argument("--judge", action="store_true", help="ativa o LLM-as-judge (hook)")
    p.add_argument("--cat", choices=sorted(CATEGORIES), help="roda só uma categoria")
    p.add_argument("--id", help="roda só um caso (por id)")
    p.add_argument("--cases", type=Path, help="caminho alternativo do cases.jsonl")
    p.add_argument("--save", type=Path, help="grava o resultado (JSON) para regressão")
    p.add_argument("--diff", type=Path, help="compara com um baseline salvo")
    p.add_argument("--threshold", type=float, default=0.95, help="taxa mínima determinística")
    p.add_argument("--verbose", action="store_true", help="mostra cada check")
    return p.parse_args(argv)


async def _run_case(case, args) -> CaseResult:
    keys = set(case.expect) - {"autor"}
    checks = []
    skipped = []

    if keys & DET_KEYS:
        checks += grade_deterministic(case)

    need_answer = bool(keys & ANSWER_KEYS)
    need_retrieval = bool(keys & RETRIEVAL_KEYS)

    if need_answer or need_retrieval:
        if not args.llm:
            skipped += sorted((keys & ANSWER_KEYS) | (keys & RETRIEVAL_KEYS))
        else:
            from app.services import pipeline

            if need_answer:
                result = await pipeline.answer(
                    case.input, conv_key=FIXTURE_CHAT,
                    autor=case.autor or DEFAULT_AUTOR, history=[],
                    request_id=f"bench-{case.id}",
                )
                checks += grade_answer(case, result)
                if need_retrieval:
                    checks += grade_retrieval(case, result)
                j = judge(case, result, enabled=args.judge)
                if j:
                    checks.append(j)
            else:
                ret = await pipeline.retrieve(case.input, conv_key=FIXTURE_CHAT)
                checks += grade_retrieval(case, ret)

    return CaseResult(case=case, checks=checks, skipped=skipped)


def _tabulate(results: list[CaseResult]) -> dict:
    per_cat: dict[str, dict] = {
        c: {"n": 0, "pass": 0, "fail": 0, "skiponly": 0, "xfail": 0, "xpass": 0}
        for c in sorted(CATEGORIES)
    }
    for r in results:
        b = per_cat[r.case.cat]
        b["n"] += 1
        if r.is_xfail:
            if r.passed:
                b["xpass"] += 1
            else:
                b["xfail"] += 1
        elif r.passed:
            b["pass"] += 1
        elif r.failed:
            b["fail"] += 1
        else:  # nenhum check rodou (tudo skip)
            b["skiponly"] += 1
    return per_cat


def _print_report(results, per_cat, args, elapsed):
    print(f"\nBENCHMARK — {len(results)} casos  ({elapsed:.1f}s)"
          f"   {'pipeline real' if args.llm else 'só determinístico'}")
    print("=" * 72)
    if not results:
        print("(nenhum caso em cases.jsonl — encanamento OK)")
        return

    hdr = f"{'cat':<12}{'casos':>6}{'pass':>6}{'fail':>6}{'skip':>6}{'xfail':>7}{'xpass':>7}"
    print(hdr)
    print("-" * 72)
    tot = {"n": 0, "pass": 0, "fail": 0, "skiponly": 0, "xfail": 0, "xpass": 0}
    for cat, b in per_cat.items():
        if b["n"] == 0:
            continue
        print(f"{cat:<12}{b['n']:>6}{b['pass']:>6}{b['fail']:>6}"
              f"{b['skiponly']:>6}{b['xfail']:>7}{b['xpass']:>7}")
        for k in tot:
            tot[k] += b[k]
    print("-" * 72)
    print(f"{'TOTAL':<12}{tot['n']:>6}{tot['pass']:>6}{tot['fail']:>6}"
          f"{tot['skiponly']:>6}{tot['xfail']:>7}{tot['xpass']:>7}")

    fails = [r for r in results if r.failed and not r.is_xfail]
    if fails:
        print("\nFALHAS (gate duro)")
        print("-" * 72)
        for r in fails:
            print(f"  {r.case.id}  ({r.case.cat})  {r.case.input!r}")
            for c in r.hard_checks:
                if not c.ok:
                    print(f"      {c.line()}")

    # advisory (answer_*) — métrica observada, não conta pro exit code
    adv = [c for r in results for c in r.advisory_checks]
    if adv:
        adv_ok = sum(1 for c in adv if c.ok)
        print(f"\nADVISORY — answer_* (não conta pro exit): {adv_ok}/{len(adv)} ok")
        adv_fail = [
            (r, c) for r in results for c in r.advisory_checks if not c.ok
        ]
        if adv_fail:
            print("-" * 72)
            for r, c in adv_fail:
                got = c.got if isinstance(c.got, str) else repr(c.got)
                print(f"  {r.case.id}  {c.key}  esperado={c.expected!r}")
                print(f"      obtido: {got[:200]}")

    xpass = [r for r in results if r.is_xfail and r.passed]
    if xpass:
        print("\nXPASS (xfail que passou — reavaliar o caso)")
        for r in xpass:
            print(f"  {r.case.id}: {r.case.xfail}")

    if args.verbose:
        print("\nDETALHE")
        print("-" * 72)
        for r in results:
            print(f"  {r.case.id} ({r.case.cat})")
            for c in r.checks:
                print(f"      {c.line()}")
            for s in r.skipped:
                print(f"      [skip] {s} (rode com --llm)")


def _cost_of_run(prefix="bench-") -> float:
    from app.services.db import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls WHERE request_id LIKE ?",
            (prefix + "%",),
        ).fetchone()
    finally:
        conn.close()
    return row[0] or 0.0


def _dump(results) -> dict:
    """Baseline = só checks de gate duro (advisory answer_* varia entre rodadas)."""
    return {
        r.case.id: {c.key: c.ok for c in r.hard_checks}
        for r in results
        if r.hard_checks
    }


def _diff(current: dict, baseline_path: Path) -> list[str]:
    try:
        base = json.loads(baseline_path.read_text())
    except OSError as exc:
        return [f"não consegui ler o baseline: {exc}"]
    regressions = []
    for cid, checks in base.items():
        cur = current.get(cid, {})
        for key, was_ok in checks.items():
            if was_ok and cur.get(key) is False:
                regressions.append(f"{cid}.{key}: ok → FAIL")
    return regressions


async def _main_async(args) -> int:
    use_fixture_db()

    cases = load_cases(args.cases) if args.cases else load_cases()
    problems = validate(cases)
    if problems:
        print("cases.jsonl inválido:")
        for p in problems:
            print(f"  - {p}")
        return 2

    if args.cat:
        cases = [c for c in cases if c.cat == args.cat]
    if args.id:
        cases = [c for c in cases if c.id == args.id]

    t0 = time.monotonic()
    results = [await _run_case(c, args) for c in cases]
    elapsed = time.monotonic() - t0

    per_cat = _tabulate(results)
    _print_report(results, per_cat, args, elapsed)

    if args.llm and results:
        print(f"\ncusto estimado da rodada: ${_cost_of_run():.4f}")

    current = _dump(results)
    if args.save:
        args.save.write_text(json.dumps(current, indent=2, ensure_ascii=False))
        print(f"\nresultado salvo em {args.save}")

    regressions = _diff(current, args.diff) if args.diff else []
    if regressions:
        print("\nREGRESSÕES vs baseline")
        for r in regressions:
            print(f"  - {r}")

    # --- exit code ---
    det_checks = [
        c for r in results if not r.is_xfail
        for c in r.checks if c.key in DET_KEYS
    ]
    det_rate = (
        sum(1 for c in det_checks if c.ok) / len(det_checks) if det_checks else 1.0
    )
    hard_fails = [r for r in results if r.failed and not r.is_xfail]

    if det_checks:
        print(f"\ntaxa determinística: {det_rate:.1%} "
              f"({sum(1 for c in det_checks if c.ok)}/{len(det_checks)})  "
              f"limiar: {args.threshold:.0%}")

    ok = det_rate >= args.threshold and not hard_fails and not regressions
    return 0 if ok else 1


def main(argv=None) -> int:
    return asyncio.run(_main_async(_parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
