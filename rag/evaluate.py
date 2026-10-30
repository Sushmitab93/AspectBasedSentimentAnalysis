"""Compare retrieval with and without the ABSA model's label filter.

Run from the repo root:  python rag/evaluate.py
"""
import pandas as pd

from rag_core import retrieve, ASPECTS, SENTIMENTS

K = 12
truth = pd.read_csv("results/predictions.csv").set_index("sentence_id")

rows = []
for aspect in ASPECTS:
    for sentiment in SENTIMENTS:
        question = f"What do customers say that is {sentiment} about the {aspect}?"

        for mode, filters in [
            ("no filter", {}),
            ("model filter", {"aspect": aspect, "sentiment": sentiment}),
        ]:
            docs = retrieve(question, k=K, **filters)
            ids = [d.metadata["sentence_id"] for d in docs]
            found = truth.loc[ids]

            rows.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "mode": mode,
                "aspect_correct": (found["true_aspect"] == aspect).mean(),
                "sentiment_correct": (found["true_sentiment"] == sentiment).mean(),
            })

results = pd.DataFrame(rows)
results.to_csv("results/rag_eval.csv", index=False)

print("Average share of retrieved sentences with the correct label:\n")
print(results.groupby("mode")[["aspect_correct", "sentiment_correct"]].mean().round(3))