"""Estimativa de custo por chamada de LLM — USD.

Ollama (local) custa 0. Para Kimi (via OpenRouter), o preço por 1M de tokens
vem da config (`KIMI_PRICE_IN_PER_MTOK` / `KIMI_PRICE_OUT_PER_MTOK`) e vale
para qualquer modelo `*kimi-k2*`. Defaults: 0.95 in / 4.00 out (USD/Mtok,
tabela do OpenRouter para kimi-k2.6).
"""

from app.config import settings


def estimate_cost_usd(
    provider: str, model: str, input_tokens: int | None, output_tokens: int | None
) -> float:
    """Custo estimado da chamada. `provider` é o nome do adapter
    (`ollama` / `kimi` / `kimi-heavy`)."""
    if provider == "ollama":
        return 0.0

    if provider.startswith("kimi") or "kimi-k2" in (model or ""):
        in_tok = input_tokens or 0
        out_tok = output_tokens or 0
        return round(
            in_tok / 1_000_000 * settings.kimi_price_in_per_mtok
            + out_tok / 1_000_000 * settings.kimi_price_out_per_mtok,
            6,
        )

    # Provider sem tabela de preço (ex: Anthropic, se adotado) — 0 até ter dados.
    return 0.0
