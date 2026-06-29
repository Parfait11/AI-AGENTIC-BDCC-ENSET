"""Execute les 20 questions d'evaluation (10 simples + 10 complexes), mesure
le temps de reponse et les sources recuperees, et ecrit evaluation/results/results.csv.

Lancer avec : uv run python -m evaluation.run_evaluation
"""
import csv
import time
from pathlib import Path

from langchain.messages import HumanMessage

from evaluation.questions import COMPLEX_QUESTIONS, SIMPLE_QUESTIONS
from src.graph import graph

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_question(question_id: str, question_type: str, question: str) -> dict:
    config = {"configurable": {"thread_id": f"eval-{question_id}"}}
    start = time.perf_counter()
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
    elapsed = time.perf_counter() - start
    answer = result["messages"][-1].content
    sources = " | ".join(result.get("retrieved_docs", []))[:500]
    return {
        "id": question_id,
        "type": question_type,
        "question": question,
        "answer": answer,
        "time_s": round(elapsed, 2),
        "sources": sources,
        "llm_calls": result.get("llm_calls", 0),
        "rewrites": result.get("rewrite_count", 0),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, q in enumerate(SIMPLE_QUESTIONS, start=1):
        print(f"Simple {i}/10: {q}")
        rows.append(run_question(f"S{i}", "simple", q))
    for i, q in enumerate(COMPLEX_QUESTIONS, start=1):
        print(f"Complexe {i}/10: {q}")
        rows.append(run_question(f"C{i}", "complexe", q))

    csv_path = RESULTS_DIR / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultats ecrits dans {csv_path}")


if __name__ == "__main__":
    main()
