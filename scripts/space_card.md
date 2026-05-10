---
title: SciFact Relevance Classifier
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Predict whether a paper is relevant evidence for a claim
models:
- andreiaalexa/scifact-relevance-classifier
- intfloat/e5-small-v2
datasets:
- andreiaalexa/scifact-relevance-pairs
---

# Scientific Evidence Search

A live retrieve-and-rerank demo with **stance prediction**: paste a scientific claim or question, and the system finds the most semantically related abstracts from the [SciFact corpus](https://huggingface.co/datasets/BeIR/scifact) (~5,000 biomedical papers), then a 3-class classifier predicts whether each paper *supports*, *refutes*, or takes *no clear stance* on the claim.

## How it works (two-stage IR pipeline)

1. **Retrieval** — encode the user query with [`intfloat/e5-small-v2`](https://huggingface.co/intfloat/e5-small-v2) (asymmetric `query:` prefix), compute cosine similarity against the pre-encoded corpus, keep top 50 candidates.
2. **Stance re-ranking** — for each candidate, build the InferSent pair features `[q, d, |q − d|, q * d, cos(q, d)]` (1537 dims), then predict the stance class (SUPPORTS / REFUTES / NEI) with a `LogisticRegression` classifier trained on `allenai/scifact` claim-evidence stance annotations.

The top 10 results are returned ranked by `P(stance) = 1 − P(NEI)`, so the most evidence-bearing papers (whether supporting OR refuting) appear first. A summary above the table aggregates the stance distribution across the full top-50 retrieval window.

## Two classifiers ship in this Space's backing model repo

| Task | Classes | Best model | macro-F1 |
|---|---|---|---:|
| Binary relevance | relevant / not_relevant | HistGradientBoosting | **0.872** |
| 3-class stance | SUPPORTS / REFUTES / NEI | LogisticRegression | **0.533** |

The Space currently uses the **stance** classifier because it directly answers the user-facing question "*does the literature support or refute this claim?*". The binary relevance classifier is also published in the [model repo](https://huggingface.co/andreiaalexa/scifact-relevance-classifier) for comparison and downstream re-use.

## Resources

- **Model + corpus embeddings**: <https://huggingface.co/andreiaalexa/scifact-relevance-classifier>
- **Training dataset**: <https://huggingface.co/datasets/andreiaalexa/scifact-relevance-pairs>
- **Source code**: <https://github.com/alexandreia/scifact-relevance-classifier>

## Disclaimer

Educational demo for the *Information Retrieval 5LN712* course at Uppsala University. The retrieval corpus is biomedical (SciFact); queries about other scientific domains may return irrelevant or misleading results. Predictions are a *first-pass* evidence sort, not a final verdict — read the abstracts and weigh the underlying study designs before drawing conclusions. Not a clinical decision-making tool.
