"""
Benchmark - Compare Flat RAG vs Graph RAG results
Reads flat_rag_results.csv and graph_rag_results.csv → prints comparison table + saves charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
FLAT_CSV = RESULTS_DIR / "flat_rag_results.csv"
GRAPH_CSV = RESULTS_DIR / "graph_rag_results.csv"
CHART_PATH = RESULTS_DIR / "comparison_chart.png"

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_results():
    flat_df = pd.read_csv(FLAT_CSV)
    graph_df = pd.read_csv(GRAPH_CSV)
    flat_df["pipeline"] = "Flat RAG"
    graph_df["pipeline"] = "Graph RAG"
    return flat_df, graph_df


def print_comparison(flat_df: pd.DataFrame, graph_df: pd.DataFrame):
    print("=" * 60)
    print("        Flat RAG vs Graph RAG — Mean Scores")
    print("=" * 60)

    summary = pd.DataFrame({
        "Flat RAG": flat_df[METRIC_COLS].mean(),
        "Graph RAG": graph_df[METRIC_COLS].mean(),
    })
    summary["Delta"] = summary["Graph RAG"] - summary["Flat RAG"]
    summary["Winner"] = summary["Delta"].apply(
        lambda d: "Graph RAG ✓" if d > 0 else ("Flat RAG ✓" if d < 0 else "Tie")
    )
    print(summary.round(4).to_string())

    print("\n--- Per Question Comparison ---")
    merged = flat_df[["id", "category", "question"] + METRIC_COLS].merge(
        graph_df[["id"] + METRIC_COLS],
        on="id",
        suffixes=("_flat", "_graph"),
    )
    print(merged.to_string(index=False))
    return summary


def plot_comparison(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Flat RAG vs Graph RAG — Benchmark Comparison", fontsize=14)

    # Bar chart — mean scores
    summary[["Flat RAG", "Graph RAG"]].plot(
        kind="bar", ax=axes[0], color=["steelblue", "coral"], rot=30
    )
    axes[0].set_title("Mean Scores by Metric")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="lower right")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    # Delta bar chart
    summary["Delta"].plot(
        kind="bar", ax=axes[1],
        color=["coral" if d < 0 else "steelblue" for d in summary["Delta"]],
        rot=30,
    )
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title("Delta (Graph RAG − Flat RAG)")
    axes[1].set_ylabel("Score Difference")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    print(f"\nChart saved to: {CHART_PATH}")
    plt.show()


if __name__ == "__main__":
    print("=== Benchmark Comparison ===\n")
    flat_df, graph_df = load_results()
    summary = print_comparison(flat_df, graph_df)
    plot_comparison(summary)
