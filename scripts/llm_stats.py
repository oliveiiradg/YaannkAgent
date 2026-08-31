"""Resumo da telemetria de LLM (tabela `llm_calls`) — provider a provider.

Uso:
  venv/bin/python scripts/llm_stats.py            # últimos 7 dias
  venv/bin/python scripts/llm_stats.py 30         # últimos 30 dias
  venv/bin/python scripts/llm_stats.py 7 tail     # + as 20 chamadas mais recentes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.db import get_connection

days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
show_tail = len(sys.argv) > 2 and sys.argv[2] == "tail"

conn = get_connection()
since = f"-{days} days"

rows = conn.execute(
    """
    SELECT provider, model,
           COUNT(*)                             AS calls,
           SUM(success)                         AS ok,
           SUM(COALESCE(input_tokens, 0))       AS in_tok,
           SUM(COALESCE(output_tokens, 0))      AS out_tok,
           ROUND(AVG(latency_ms))               AS avg_ms,
           ROUND(SUM(COALESCE(cost_usd, 0)), 4) AS cost_usd
    FROM llm_calls
    WHERE ts >= datetime('now', ?)
    GROUP BY provider, model
    ORDER BY calls DESC
    """,
    (since,),
).fetchall()

print(f"\nTelemetria LLM — últimos {days} dias\n" + "=" * 78)
if not rows:
    print("(sem registros — tabela vazia ou LLM_TELEMETRY desligado)")
else:
    hdr = f"{'provider':<12}{'model':<18}{'calls':>6}{'ok':>5}{'in tok':>10}{'out tok':>10}{'avg ms':>8}{'USD':>10}"
    print(hdr)
    print("-" * 78)
    tot_cost = 0.0
    for provider, model, calls, ok, in_tok, out_tok, avg_ms, cost in rows:
        tot_cost += cost or 0.0
        print(
            f"{provider:<12}{(model or '-'):<18}{calls:>6}{ok:>5}"
            f"{in_tok:>10}{out_tok:>10}{(avg_ms or 0):>8.0f}{(cost or 0):>10.4f}"
        )
    print("-" * 78)
    print(f"{'custo total estimado (USD)':<59}{tot_cost:>19.4f}")

by_block = conn.execute(
    """
    SELECT block, provider, COUNT(*) AS calls, ROUND(AVG(latency_ms)) AS avg_ms
    FROM llm_calls
    WHERE ts >= datetime('now', ?)
    GROUP BY block, provider
    ORDER BY block
    """,
    (since,),
).fetchall()
if by_block:
    print("\npor bloco do pipeline")
    print("-" * 40)
    for block, provider, calls, avg_ms in by_block:
        print(f"  {(block or '-'):<9} {provider:<12} {calls:>4} calls  {(avg_ms or 0):>6.0f} ms")

by_intent = conn.execute(
    """
    SELECT intent, tier, provider, COUNT(*) AS calls,
           ROUND(SUM(COALESCE(cost_usd, 0)), 4) AS cost
    FROM llm_calls
    WHERE ts >= datetime('now', ?) AND block = 'bloco3'
    GROUP BY intent, tier, provider
    ORDER BY calls DESC
    """,
    (since,),
).fetchall()
if by_intent:
    print("\nBloco 3 por intent / tier (decisão do Router)")
    print("-" * 52)
    for intent, tier, provider, calls, cost in by_intent:
        print(
            f"  {(intent or '-'):<16} tier {tier if tier is not None else '-'}  "
            f"{provider:<12} {calls:>4} calls  ${cost or 0:.4f}"
        )

fb = conn.execute(
    """
    SELECT provider, fallback_reason, COUNT(*) AS n
    FROM llm_calls
    WHERE ts >= datetime('now', ?) AND fallback = 1
    GROUP BY provider, fallback_reason
    ORDER BY n DESC
    """,
    (since,),
).fetchall()
if fb:
    print("\nescalonamentos (Fase 6)")
    print("-" * 52)
    for provider, reason, n in fb:
        print(f"  {provider:<12} {n:>3}x  {reason or '-'}")

if show_tail:
    tail = conn.execute(
        """
        SELECT ts, request_id, block, provider, model, intent,
               input_tokens, output_tokens, latency_ms, cost_usd, success, error
        FROM llm_calls ORDER BY id DESC LIMIT 20
        """
    ).fetchall()
    print("\n20 chamadas mais recentes")
    print("-" * 78)
    for r in tail:
        ts, rid, block, prov, model, intent, it, ot, ms, cost, ok, err = r
        flag = "" if ok else f"  ERRO: {(err or '')[:60]}"
        print(
            f"  {ts}  {rid or '-':<12} {block or '-':<7} {prov:<11} {model or '-':<16}"
            f" {intent or '-':<14} {it or 0:>6}/{ot or 0:<6} {ms or 0:>6}ms ${cost or 0:.5f}{flag}"
        )

conn.close()
