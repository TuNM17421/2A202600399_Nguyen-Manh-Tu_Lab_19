"""
Benchmark - Flat RAG Evaluation
Runs testset.json through Flat RAG chain → evaluates with RAGAS → saves results CSV
"""

import json
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from flat_rag.chain import build_chain, ask

load_dotenv()

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_PATH = Path(__file__).parent / "flat_rag_results.csv"


def run_pipeline(testset: list[dict], chain) -> list[dict]:
    """Run each question through Flat RAG chain, collect answers + contexts."""
    rows = []
    for item in testset:
        print(f"  [{item['id']}] {item['question'][:70]}...")
        out = ask(item["question"], chain)
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "answer": out["answer"],
            "contexts": out["contexts"],          # list[str]
            "ground_truth": item["ground_truth"],
            "category": item.get("category", ""),
            "sources": out["sources"],
        })
    return rows


def evaluate_with_ragas(rows: list[dict]) -> pd.DataFrame:
    """Convert rows to RAGAS Dataset format and evaluate."""
    ragas_data = {
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    }
    dataset = Dataset.from_dict(ragas_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores_df = result.to_pandas().reset_index(drop=True)

    meta_df = pd.DataFrame([
        {"id": r["id"], "question": r["question"], "category": r["category"], "sources": str(r["sources"])}
        for r in rows
    ]).reset_index(drop=True)

    final_df = pd.concat([meta_df, scores_df.drop(columns=[c for c in ["question"] if c in scores_df.columns], errors="ignore")], axis=1)
    return final_df


if __name__ == "__main__":
    print("=== Flat RAG Benchmark ===\n")

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    print(f"Loaded {len(testset)} test case(s) from {TESTSET_PATH.name}\n")

    print("[1] Running Flat RAG chain...")
    chain = build_chain()
    rows = run_pipeline(testset, chain)

    print("\n[2] Evaluating with RAGAS...")
    df = evaluate_with_ragas(rows)

    print("\n=== Results ===")
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(df[["id", "category"] + metric_cols].to_string(index=False))

    print(f"\n--- Mean Scores ---")
    print(df[metric_cols].mean().to_string())

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved to: {RESULTS_PATH}")
