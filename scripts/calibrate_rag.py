"""Mede os scores do RAG (vault_search) para um conjunto de perguntas reais,
sem alterar o threshold — só para calibração manual.

Uso: venv/bin/python3 scripts/calibrate_rag.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.vault_search import (
    _MIN_DISTINCT_KEYWORDS,
    _MIN_SCORE,
    _boost_factor,
    _iter_vault_files,
    _keywords,
    _word_counts,
)

# Perguntas de calibração — ajuste para o conteúdo do seu vault.
QUESTIONS = [
    "O que o Bloco 2 faz no pipeline?",
    "Qual modelo o reranker usa?",
    "Como funciona o RRF na busca?",
    "O que roda sempre no Ollama local?",
    "Quais são os tiers do LLM Router?",
    "O que está pendente no projeto?",
    "Qual foi o último bug corrigido?",
    "Qual a capital da França?",  # controle: conhecimento geral, fora do vault
    "Qual a cor favorita do gato do vizinho?",  # controle: não está no vault
    "Como o gateway roda em produção?",
]

TOP_N = 5


def score_all(query: str, priority_folder: str | None = None):
    keywords = _keywords(query)
    if not keywords:
        return keywords, []

    docs = []
    for path in _iter_vault_files():
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        docs.append((path, content, _word_counts(content)))

    if not docs:
        return keywords, []

    doc_freq = {kw: sum(1 for _, _, c in docs if c.get(kw)) for kw in keywords}
    n_docs = len(docs)
    idf = {kw: math.log((n_docs + 1) / (doc_freq[kw] + 1)) + 1 for kw in keywords}
    min_distinct = min(_MIN_DISTINCT_KEYWORDS, len(keywords))
    root = Path(settings.vault_path)

    results = []
    for path, content, counts in docs:
        matched = [kw for kw in keywords if counts.get(kw)]
        if len(matched) < min_distinct:
            continue
        tf_idf = sum(counts[kw] * idf[kw] for kw in matched)
        content_len = sum(counts.values()) or 1
        raw_score = tf_idf / math.sqrt(content_len)
        boosted = raw_score * _boost_factor(path, root, priority_folder)
        results.append((raw_score, boosted, len(matched), path.relative_to(root)))

    results.sort(key=lambda r: r[0], reverse=True)
    return keywords, results


def main():
    print(f"_MIN_DISTINCT_KEYWORDS = {_MIN_DISTINCT_KEYWORDS}")
    print(f"_MIN_SCORE = {_MIN_SCORE}")
    print("=" * 100)

    for question in QUESTIONS:
        keywords, results = score_all(question)
        print(f"\nPERGUNTA: {question}")
        print(f"  keywords extraidas: {keywords}")

        if not results:
            print("  -> NENHUM candidato (0 arquivos com >= min_distinct keywords)")
            continue

        passed = [r for r in results if r[0] >= _MIN_SCORE]
        print(f"  -> {len(passed)}/{len(results)} candidatos passam no threshold (_MIN_SCORE={_MIN_SCORE})")

        for raw_score, boosted, n_matched, rel_path in results[:TOP_N]:
            gate = "PASSA" if raw_score >= _MIN_SCORE else "abaixo"
            print(
                f"    [{gate:6s}] raw={raw_score:6.3f}  boosted={boosted:6.3f}  "
                f"keywords_batidas={n_matched}  arquivo={rel_path}"
            )


if __name__ == "__main__":
    main()
