"""
Avaliação do sistema RAG.

Métricas:
1. Recall@k da recuperação: o chunk certo está nos top-k recuperados?
2. Geração: salva respostas para avaliação manual (subjetiva 1-5).

Uso:
    python -m evals.run_evals
"""

import json
from pathlib import Path

from astronomia_rag.retriever import search
from astronomia_rag.chain import build_chain
from astronomia_rag.memory import ChatHistory


EVAL_DIR = Path(__file__).parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"


def evaluate_retrieval(golden_set: list, top_k: int = 3) -> dict:
    """
    Recall@k: para cada pergunta, verifica se o arquivo esperado
    apareceu nos top-k chunks recuperados.
    """
    hits = 0
    misses = []

    for item in golden_set:
        query = item["question"]
        expected_source = item["expected_source"]

        docs = search(query, top_k=top_k)
        retrieved_sources = [
            d.metadata.get("source", "").split("/")[-1] for d in docs
        ]

        if expected_source in retrieved_sources:
            hits += 1
        else:
            misses.append({
                "question": query,
                "expected": expected_source,
                "got": retrieved_sources,
            })

    total = len(golden_set)
    return {
        "recall_at_k": hits / total,
        "hits": hits,
        "total": total,
        "misses": misses,
    }


def evaluate_generation(golden_set: list) -> list:
    """
    Gera resposta para cada pergunta. Avaliação subjetiva depois.
    Cada pergunta roda em isolamento (sem histórico cruzado).
    """
    chain = build_chain()
    results = []

    for i, item in enumerate(golden_set, 1):
        print(f"  [{i}/{len(golden_set)}] {item['question'][:60]}...")
        history = ChatHistory()
        response = chain.invoke({
            "input": item["question"],
            "chat_history": history.get_messages(),
        })

        results.append({
            "question": item["question"],
            "expected_answer": item.get("expected_answer", ""),
            "actual_answer": response["answer"],
            "sources_retrieved": [
                d.metadata.get("source", "").split("/")[-1]
                for d in response.get("context", [])
            ],
            "subjective_score": None,  # pra você preencher (1-5)
        })

    return results


def main():
    print(f"Carregando golden set de {GOLDEN_SET_PATH}...")
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        golden_set = json.load(f)
    print(f"   {len(golden_set)} perguntas carregadas.\n")

    print("Avaliando recuperação (Recall@3)...")
    retrieval = evaluate_retrieval(golden_set, top_k=3)
    print(f"   Recall@3: {retrieval['recall_at_k']:.2%} "
          f"({retrieval['hits']}/{retrieval['total']})")
    if retrieval["misses"]:
        print(f"  {len(retrieval['misses'])} misses (ver eval_results.json)")
    print()

    print("Gerando respostas (vai demorar — 1 LLM call por pergunta)...")
    generation_results = evaluate_generation(golden_set)

    output = {
        "retrieval": retrieval,
        "generation": generation_results,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {RESULTS_PATH}")
    print("Abra o JSON e preencha `subjective_score` (1-5) pra cada resposta.")


if __name__ == "__main__":
    main()