import asyncio
import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)

_MODEL_FILE = "model_int8.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_MAX_LENGTH = 512

_session = None
_tokenizer = None
_load_failed = False


class RerankerError(Exception):
    """Erro no reranker — sinaliza que quem chama deve usar o resultado do RRF direto."""


def _ensure_loaded() -> None:
    """Carrega o modelo ONNX e o tokenizer uma única vez (custa ~3s)."""
    global _session, _tokenizer, _load_failed

    if _session is not None and _tokenizer is not None:
        return
    if _load_failed:
        raise RerankerError("carregamento do modelo já falhou antes nesta sessão")

    model_path = os.path.join(settings.reranker_model_dir, _MODEL_FILE)
    tokenizer_path = os.path.join(settings.reranker_model_dir, _TOKENIZER_FILE)

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        _load_failed = True
        raise RerankerError(
            f"arquivos do modelo não encontrados em {settings.reranker_model_dir}"
        )

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_padding()
        tokenizer.enable_truncation(max_length=_MAX_LENGTH)
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        _load_failed = True
        raise RerankerError(f"falha ao carregar modelo/tokenizer: {exc!r}") from exc

    _session = session
    _tokenizer = tokenizer
    logger.info("Reranker carregado (%s)", settings.reranker_model_dir)


def _rerank_sync(
    query: str, candidates: list[dict], score_chars: int | None = None
) -> list[float]:
    """Roda a inferência do cross-encoder — CPU-bound, chamada via to_thread.

    `score_chars` corta o snippet SÓ para o scoring (o dict do candidato não é
    tocado — o snippet completo segue para o Bloco 3). O tokenizer já trunca em
    `_MAX_LENGTH` tokens, então isto é só um pré-corte barato.
    """
    import numpy as np

    _ensure_loaded()

    pairs = [
        (query, c["snippet"][:score_chars] if score_chars else c["snippet"])
        for c in candidates
    ]
    encodings = _tokenizer.encode_batch(pairs)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    logits = _session.run(
        ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
    )[0]
    return [float(score) for score in logits[:, 0]]


FOLDER_BONUS = 2.0


async def rerank(
    query: str, candidates: list[dict], top_k: int = 3,
    priority_folder: str | None = None, score_chars: int | None = None,
) -> list[dict]:
    """Reordena os candidatos do RRF via cross-encoder, retorna os top_k.

    candidates: lista de {"filePath": str, "snippet": str, ...} vinda do RRF,
    já ordenada por rrf_score — essa ordem é o fallback se o reranker falhar.
    priority_folder: pasta identificada pelo Bloco 2 — candidatos dela recebem
    um bônus no score, já que o cross-encoder não tem esse sinal por conta
    própria (só julga o texto do par pergunta/snippet).
    score_chars: corta o snippet só para o scoring (não muta o candidato).
    """
    if not candidates:
        return []

    try:
        scores = await asyncio.to_thread(_rerank_sync, query, candidates, score_chars)
    except RerankerError as exc:
        logger.warning("Reranker falhou (%s) — usando ordem do RRF direto", exc)
        return candidates[:top_k]

    candidates = [dict(c, rerank_score=s) for c, s in zip(candidates, scores)]
    if priority_folder:
        for c in candidates:
            if c["filePath"].startswith(priority_folder):
                c["rerank_score"] += FOLDER_BONUS

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    top = reranked[:top_k]
    logger.info(
        "Reranker: %d candidatos reordenados, top %d: %s",
        len(candidates), len(top), [(c["filePath"], round(c["rerank_score"], 3)) for c in top],
    )
    return top
