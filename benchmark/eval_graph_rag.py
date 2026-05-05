"""
Benchmark - Graph RAG Evaluation
Runs testset.json through Graph RAG chain → evaluates with RAGAS → saves results CSV
Format mirrors eval_flat_rag.py for easy comparison
"""

import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from graph_rag.chain import build_chain, ask  # build_chain now returns (chain, graph, entities)

load_dotenv()

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_PATH = Path(__file__).parent / "graph_rag_results.csv"


def run_pipeline(testset: list[dict], chain, graph, entities: str = "") -> list[dict]:
    """Run each question through Graph RAG chain, collect answers + contexts."""
    rows = []
    for item in testset:
        print(f"  [{item['id']}] {item['question'][:70]}...")
        out = ask(item["question"], chain, graph, entities)
        print(f"       Cypher: {out['cypher'][:80]}...")
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "answer": out["answer"],
            "contexts": out["contexts"],          # list[str] of graph rows
            "ground_truth": item["ground_truth"],
            "category": item.get("category", ""),
            "cypher": out["cypher"],
        })
    return rows


def evaluate_with_ragas(rows: list[dict]) -> pd.DataFrame:
    """Convert rows to RAGAS Dataset format and evaluate."""
    # RAGAS needs at least one non-empty context per row
    # If graph returned no rows, fall back to a placeholder
    ragas_data = {
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] if r["contexts"] else ["no graph data returned"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    }
    dataset = Dataset.from_dict(ragas_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores_df = result.to_pandas().reset_index(drop=True)

    meta_df = pd.DataFrame([
        {"id": r["id"], "question": r["question"], "category": r["category"], "cypher": r["cypher"]}
        for r in rows
    ]).reset_index(drop=True)

    final_df = pd.concat([meta_df, scores_df.drop(columns=[c for c in ["question"] if c in scores_df.columns], errors="ignore")], axis=1)
    return final_df


if __name__ == "__main__":
    print("=== Graph RAG Benchmark ===\n")

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    print(f"Loaded {len(testset)} test case(s) from {TESTSET_PATH.name}\n")

    print("[1] Running Graph RAG chain...")
    chain, graph, entities = build_chain()
    rows = run_pipeline(testset, chain, graph, entities)

    print("\n[2] Evaluating with RAGAS...")
    df = evaluate_with_ragas(rows)

    print("\n=== Results ===")
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(df[["id", "category"] + metric_cols].to_string(index=False))

    print(f"\n--- Mean Scores ---")
    print(df[metric_cols].mean().to_string())

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved to: {RESULTS_PATH}")
