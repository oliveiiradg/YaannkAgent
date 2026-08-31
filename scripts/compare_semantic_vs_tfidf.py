"""Compara busca semantica (MCP) vs TF-IDF para a bateria de teste do RAG.

Uso: venv/bin/python3 scripts/compare_semantic_vs_tfidf.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.vault_search import identify_relevant_folder, search_vault
from app.services.vault_structure import get_structural_context
from app.services.vault_search_semantic import search_vault_semantic, _McpSearchError

# Ajuste para o conteúdo do seu vault.
QUESTIONS = [
    "O que o Bloco 2 faz no pipeline?",
    "Quais são as pendências do projeto?",
    "Qual modelo o reranker usa?",
    "Como funciona o LLM Router?",
    "Qual foi o último erro corrigido no gateway?",
]


def _first_file(context: str) -> str:
    for line in context.splitlines():
        if line.startswith("### "):
            return line[4:]
    return "(vazio)"


async def main():
    structural_context = await get_structural_context()

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n{'=' * 100}\n{i}. {q}")

        priority_folder = await identify_relevant_folder(q, structural_context)
        print(f"   pasta prioritaria (Bloco 2 classificador): {priority_folder or 'nenhuma'}")

        t0 = time.monotonic()
        try:
            semantic_result = await search_vault_semantic(q, priority_folder)
            semantic_ms = (time.monotonic() - t0) * 1000
            print(f"   SEMANTICO ({semantic_ms:.0f}ms): top arquivo = {_first_file(semantic_result)}")
            print(f"     ({len(semantic_result)} chars retornados)")
        except _McpSearchError as exc:
            semantic_ms = (time.monotonic() - t0) * 1000
            print(f"   SEMANTICO ({semantic_ms:.0f}ms): FALHOU -> {exc}")

        t0 = time.monotonic()
        tfidf_result = search_vault(q, priority_folder)
        tfidf_ms = (time.monotonic() - t0) * 1000
        print(f"   TF-IDF    ({tfidf_ms:.0f}ms): top arquivo = {_first_file(tfidf_result)}")
        print(f"     ({len(tfidf_result)} chars retornados)")


if __name__ == "__main__":
    asyncio.run(main())
