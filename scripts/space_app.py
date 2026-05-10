"""Gradio Space — retrieve-and-rerank scientific evidence search with STANCE.

Two-stage IR pipeline:
  1. RETRIEVAL   — encode user query with e5-small-v2, cosine vs pre-encoded
                   SciFact corpus (~5k abstracts), keep top-K candidates.
  2. STANCE RE-RANK — for each candidate, predict whether the paper SUPPORTS
                      or REFUTES the claim (or takes no position: NEI).
                      Rank by P(takes a stance) = 1 - P(NEI).

For each result we show predicted stance + all 3 class probabilities, so the
user can see whether the evidence supports or refutes, not just whether it
is topically relevant.
"""

from __future__ import annotations

import json
import math
import pickle

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

MODEL_REPO = "andreiaalexa/scifact-relevance-classifier"
ENCODER_ID = "intfloat/e5-small-v2"

TOP_K_RETRIEVE = 50      # how many candidates the retriever feeds the re-ranker
TOP_N_DISPLAY = 10       # how many results the user sees after re-ranking

STANCE_LABELS = {0: "NEI", 1: "REFUTES", 2: "SUPPORTS"}
STANCE_COLORS = {
    "SUPPORTS": "#2e7d32",   # green
    "REFUTES":  "#c62828",   # red
    "NEI":      "#757575",   # gray
}


def colored_label(label: str) -> str:
    color = STANCE_COLORS.get(label, "#000000")
    return f"<span style='color:{color}; font-weight:600;'>{label}</span>"


# --- Module-level loads (cold start, ~30 s) ---
print(f"[boot] downloading stance classifier from {MODEL_REPO} ...")
clf_path = hf_hub_download(MODEL_REPO, "classifier_stance.pkl")
with open(clf_path, "rb") as f:
    classifier = pickle.load(f)

print("[boot] downloading stance metadata ...")
meta_path = hf_hub_download(MODEL_REPO, "metadata_stance.json")
with open(meta_path) as f:
    stance_meta = json.load(f)
ID2LABEL = {int(k): v for k, v in stance_meta["id2label"].items()}
print(f"[boot]   stance label map: {ID2LABEL}")

print("[boot] downloading corpus embeddings ...")
emb_path = hf_hub_download(MODEL_REPO, "corpus_embeddings.npy")
corpus_embeddings = np.load(emb_path)
print(f"[boot]   shape = {corpus_embeddings.shape}, dtype = {corpus_embeddings.dtype}")

print("[boot] downloading corpus metadata ...")
meta_csv = hf_hub_download(MODEL_REPO, "corpus_meta.csv")
corpus_meta = pd.read_csv(meta_csv)
assert len(corpus_meta) == corpus_embeddings.shape[0], "meta and embeddings out of sync"

print(f"[boot] loading encoder: {ENCODER_ID}")
encoder = SentenceTransformer(ENCODER_ID)
print("[boot] ready.\n")


def pair_features_against(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    q = np.broadcast_to(query_emb, doc_embs.shape)
    cos = np.sum(q * doc_embs, axis=1, keepdims=True)
    return np.hstack([q, doc_embs, np.abs(q - doc_embs), q * doc_embs, cos])


# Coverage thresholds — tuned for the SciFact corpus (~5k biomedical papers).
# When average top-50 cosine is below COVERAGE_LOW, the corpus probably lacks
# evidence about the claim at all. We display a prominent warning rather than
# pretending the stance counts are meaningful.
COVERAGE_LOW = 0.55
COVERAGE_OK = 0.65

# Confidence threshold for counting a SUPPORTS or REFUTES prediction in the
# evidence summary. 0.6 is stricter than the classifier's argmax (which can be
# below 0.5 in 3-class settings) and avoids counting low-confidence guesses.
CONFIDENCE_THRESHOLD = 0.6
MIN_STANCE_DOCS_FOR_TEST = 5


def exact_two_sided_binomial_p(successes: int, trials: int) -> float | None:
    """Exact two-sided binomial test against p=0.5.

    Here a "success" means SUPPORTS and a "failure" means REFUTES. This tests
    whether the confidently classified stance-bearing documents are balanced
    between support and refutation. It is not a p-value from the original
    biomedical studies.
    """
    if trials <= 0:
        return None

    observed = math.comb(trials, successes) * (0.5 ** trials)
    p_value = 0.0
    for k in range(trials + 1):
        prob = math.comb(trials, k) * (0.5 ** trials)
        if prob <= observed + 1e-15:
            p_value += prob
    return min(1.0, p_value)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson 95% interval for the SUPPORTS share among SUPPORTS+REFUTES docs."""
    if trials <= 0:
        return None

    p_hat = successes / trials
    denom = 1.0 + (z * z) / trials
    centre = p_hat + (z * z) / (2 * trials)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4 * trials)) / trials)
    return (centre - margin) / denom, (centre + margin) / denom


def directional_evidence_summary(s: dict[str, int], max_cos: float, avg_cos: float) -> str:
    supports = s["SUPPORTS"]
    refutes = s["REFUTES"]
    stance_docs = supports + refutes

    if max_cos < COVERAGE_LOW:
        return (
            "### Directional evidence test\n\n"
            "No statistical evidence test is reported because retrieval coverage is too low: "
            "the corpus does not appear to contain directly relevant papers for this claim.\n\n"
        )

    if stance_docs < MIN_STANCE_DOCS_FOR_TEST:
        return (
            "### Directional evidence test\n\n"
            f"Only **{stance_docs}** confidently stance-bearing papers were found "
            f"({supports} SUPPORTS, {refutes} REFUTES). At least "
            f"{MIN_STANCE_DOCS_FOR_TEST} are required before reporting a p-value.\n\n"
        )

    p_value = exact_two_sided_binomial_p(supports, stance_docs)
    interval = wilson_interval(supports, stance_docs)
    support_share = supports / stance_docs
    ci_low, ci_high = interval

    if p_value is not None and p_value < 0.05:
        direction = "supporting" if supports > refutes else "refuting"
        verdict = f"statistically significant skew toward **{direction}** the claim"
    else:
        verdict = "no statistically significant directional skew"

    return (
        "### Directional evidence test\n\n"
        "| Measure | Value |\n"
        "|---|---:|\n"
        f"| Confident SUPPORTS | {supports} |\n"
        f"| Confident REFUTES | {refutes} |\n"
        f"| SUPPORTS share | {support_share:.1%} |\n"
        f"| 95% Wilson CI | {ci_low:.1%} to {ci_high:.1%} |\n"
        f"| Exact binomial p-value | {p_value:.4f} |\n\n"
        f"Interpretation: **{verdict}** among the confidently classified, retrieved "
        f"stance-bearing documents. Retrieval coverage: avg cosine = {avg_cos:.2f}, "
        f"max cosine = {max_cos:.2f}.\n\n"
        "> This is a document-level evidence aggregation test over model-classified "
        "retrieval results, not a p-value extracted from the original studies.\n\n"
    )


def search(query: str):
    query = query.strip()
    if not query:
        empty_df = pd.DataFrame({"info": ["Please enter a scientific claim or question."]})
        return "", empty_df

    # 1) RETRIEVAL
    q_emb = encoder.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    cos_scores = corpus_embeddings @ q_emb
    candidate_idx = np.argpartition(-cos_scores, TOP_K_RETRIEVE)[:TOP_K_RETRIEVE]
    candidate_idx = candidate_idx[np.argsort(-cos_scores[candidate_idx])]

    avg_cos = float(cos_scores[candidate_idx].mean())
    max_cos = float(cos_scores[candidate_idx].max())

    # 2) STANCE RE-RANK
    candidate_embs = corpus_embeddings[candidate_idx]
    X = pair_features_against(q_emb, candidate_embs)
    proba = classifier.predict_proba(X)             # (K, 3) — [P(NEI), P(REFUTES), P(SUPPORTS)]
    pred_ids = classifier.predict(X)                # (K,)

    # Rank by P(stance) = 1 - P(NEI). Higher first.
    p_stance = 1.0 - proba[:, 0]
    rerank_order = np.argsort(-p_stance)
    top_local_idx = rerank_order[:TOP_N_DISPLAY]

    # 3) Build summary — but only count CONFIDENT predictions.
    # A prediction is "confident" if the predicted class probability is >= 0.6.
    # Anything below that we treat as low-confidence (uncertain), separate column.
    summary_counts = {"SUPPORTS": 0, "REFUTES": 0, "NEI": 0, "uncertain": 0}
    for i, pid in enumerate(pred_ids):
        label = ID2LABEL[int(pid)]
        confidence = float(proba[i, int(pid)])
        if confidence >= CONFIDENCE_THRESHOLD:
            summary_counts[label] += 1
        else:
            summary_counts["uncertain"] += 1

    s = summary_counts

    # 4) Coverage warning — fires when the corpus has no semantically close docs.
    coverage_warning = ""
    if max_cos < COVERAGE_LOW:
        coverage_warning = (
            "> **WARNING — the SciFact corpus appears to have no papers directly relevant to this claim.** "
            f"Top retrieval cosine is only {max_cos:.2f} (typical relevant hits are >0.7). "
            "Stance predictions below are made on weakly related papers and **should not be trusted as evidence about your claim**. "
            "Real fact-checking on this topic would require a much larger corpus (e.g. PubMed, Semantic Scholar Open Research Corpus).\n\n"
        )
    elif avg_cos < COVERAGE_OK:
        coverage_warning = (
            f"> **Note — corpus coverage is limited for this claim** (avg top-50 cosine = {avg_cos:.2f}). "
            "Some retrieved papers may be only loosely related. Read the abstracts and verify before relying on the summary.\n\n"
        )

    summary = coverage_warning
    summary += (
        f"### Evidence summary across top {TOP_K_RETRIEVE} retrieved papers\n\n"
        f"- {colored_label('SUPPORTS')} (confident, P>={CONFIDENCE_THRESHOLD}): **{s['SUPPORTS']}** papers\n"
        f"- {colored_label('REFUTES')} (confident, P>={CONFIDENCE_THRESHOLD}): **{s['REFUTES']}** papers\n"
        f"- {colored_label('NEI')} (confident, P>={CONFIDENCE_THRESHOLD}): **{s['NEI']}** papers\n"
        f"- *uncertain* (no class above {CONFIDENCE_THRESHOLD}): **{s['uncertain']}** papers\n\n"
        f"_Retrieval coverage: avg cosine = {avg_cos:.2f}, max cosine = {max_cos:.2f}._\n\n"
    )
    summary += directional_evidence_summary(s, max_cos, avg_cos)

    # 5) Honest trend messaging — driven by STANCE DENSITY, not coverage cosine.
    # Coverage tells you "are these papers semantically near my query?" — high
    # cosine can still mean "topically adjacent but not on-claim".
    # Stance density tells you "how many papers actually take a position?" —
    # this is what matters for an evidence-summary verdict.
    confident_stance_count = s["SUPPORTS"] + s["REFUTES"]
    stance_density = confident_stance_count / max(TOP_K_RETRIEVE, 1)
    nei_dominant = s["NEI"] >= 0.7 * TOP_K_RETRIEVE  # >=70% NEI

    if max_cos < COVERAGE_LOW:
        summary += (
            "*The corpus does not appear to cover this claim "
            "(retrieval cosine too low). **No reliable trend can be inferred.***\n"
        )
    elif nei_dominant or stance_density < 0.20:
        summary += (
            "*Most retrieved papers are classified as **NEI** (no clear stance). "
            "This typically means the corpus contains papers that are topically "
            "near your claim but do not specifically support or refute it. "
            "**No reliable trend can be inferred** from the few confident stance "
            "predictions alone — they may be classifier noise.*\n"
        )
    elif confident_stance_count < 5:
        summary += (
            "*Too few confident stance predictions "
            f"({confident_stance_count} out of {TOP_K_RETRIEVE} retrieved) "
            "to establish a trend.*\n"
        )
    elif s["SUPPORTS"] >= 3 * s["REFUTES"]:
        summary += "*Trend: among confidently-classified papers, evidence strongly skews toward **supporting** the claim.*\n"
    elif s["REFUTES"] >= 3 * s["SUPPORTS"]:
        summary += "*Trend: among confidently-classified papers, evidence strongly skews toward **refuting** the claim.*\n"
    elif s["SUPPORTS"] >= 2 * s["REFUTES"] and s["SUPPORTS"] >= 5:
        summary += "*Trend: among confidently-classified papers, evidence skews toward **supporting** the claim.*\n"
    elif s["REFUTES"] >= 2 * s["SUPPORTS"] and s["REFUTES"] >= 5:
        summary += "*Trend: among confidently-classified papers, evidence skews toward **refuting** the claim.*\n"
    else:
        summary += "*Trend: among confidently-classified papers, evidence is **mixed** between support and refutation.*\n"

    # 4) Build display DataFrame (top-N only, sorted by P(stance))
    rows = []
    for rank, local_i in enumerate(top_local_idx, start=1):
        corpus_i = candidate_idx[local_i]
        meta_row = corpus_meta.iloc[corpus_i]
        title = (meta_row["title"] or "").strip()
        abstract = (meta_row["abstract"] or "").strip()
        excerpt = abstract if len(abstract) <= 200 else abstract[:197].rstrip() + "..."
        doc_id = str(meta_row["doc_id"])
        s2_url = f"https://www.semanticscholar.org/paper/CorpusID:{doc_id}"

        pred_id = int(pred_ids[local_i])
        pred_label = ID2LABEL[pred_id]
        pred_confidence = float(proba[local_i, pred_id])
        contributes = (
            pred_label in ("SUPPORTS", "REFUTES")
            and pred_confidence >= CONFIDENCE_THRESHOLD
            and max_cos >= COVERAGE_LOW
        )
        rows.append({
            "rank": rank,
            "stance": colored_label(pred_label),
            "P(SUPPORTS)": round(float(proba[local_i, 2]), 3),
            "P(REFUTES)":  round(float(proba[local_i, 1]), 3),
            "P(NEI)":      round(float(proba[local_i, 0]), 3),
            "retrieval_cos": round(float(cos_scores[corpus_i]), 3),
            "used_in_test": "yes" if contributes else "no",
            "title (click for source)": f"[{title}]({s2_url})",
            "abstract_excerpt": excerpt,
        })
    return summary, pd.DataFrame(rows)


examples = [
    ["Vitamin D supplementation reduces respiratory infections."],
    ["MMR vaccines cause autism in children."],
    ["Coffee consumption causes humans to photosynthesize in direct sunlight."],
    ["Regular physical activity reduces the risk of cardiovascular disease."],
    ["Statins lower LDL cholesterol levels."],
    ["Hydroxychloroquine is an effective treatment for COVID-19."],
]


with gr.Blocks(title="SciFact Evidence Search") as demo:
    gr.Markdown("# Scientific Evidence Search")
    gr.Markdown(
        "Enter a scientific claim or question. The system retrieves the most "
        "semantically similar abstracts from the SciFact corpus (~5,000 papers), "
        "then a 3-class **stance classifier** predicts whether each paper "
        "*supports*, *refutes*, or *takes no clear position* on the claim."
    )

    with gr.Row():
        query = gr.Textbox(
            label="Scientific claim or question",
            placeholder="e.g. Vitamin D supplementation reduces respiratory infections.",
            lines=2,
            scale=4,
        )
        button = gr.Button("Search evidence", variant="primary", scale=1)

    summary_md = gr.Markdown()

    output = gr.Dataframe(
        label="Top results re-ranked by P(stance) — most evidence-bearing first",
        headers=["rank", "stance", "P(SUPPORTS)", "P(REFUTES)", "P(NEI)",
                 "retrieval_cos", "used_in_test", "title (click for source)", "abstract_excerpt"],
        datatype=["number", "markdown", "number", "number", "number",
                  "number", "str", "markdown", "str"],
        wrap=True,
        interactive=False,
    )

    with gr.Accordion("About these scores (click to expand)", open=False):
        gr.Markdown(
            """
**`stance`** — the classifier's predicted relationship between the paper and your query:

- <span style='color:#2e7d32; font-weight:600;'>SUPPORTS</span> — the paper provides evidence *for* the claim.
- <span style='color:#c62828; font-weight:600;'>REFUTES</span> — the paper provides evidence *against* the claim.
- <span style='color:#757575; font-weight:600;'>NEI</span> — *not enough info*. The paper is on-topic but doesn't take a clear position.

**`P(SUPPORTS)` / `P(REFUTES)` / `P(NEI)`** — the classifier's class probabilities (sum to 1.0). Higher = more confident in that class. Useful when the prediction is borderline.

**`retrieval_cos`** — cosine similarity between your query embedding and the paper embedding (both via `intfloat/e5-small-v2`). This is what the *retrieval stage* used to pick the top 50 candidates. The classifier then re-ranked them by P(stance).

**`used_in_test`** — whether this paper was counted in the directional evidence test. A paper is counted only when it is confidently classified as SUPPORTS or REFUTES and retrieval coverage is adequate.

**Directional evidence test** — an exact binomial test over confident SUPPORTS vs REFUTES documents. It asks whether the retrieved stance-bearing papers are balanced, or whether they significantly skew toward one direction. This is **not** the same thing as the p-values or effect sizes reported inside the original papers.

**Important caveats:**

1. The model was trained on **biomedical** SciFact data (3,545 (claim, document) pairs, of which only 194 are REFUTES). Predictions on non-biomedical claims may be unreliable.
2. **REFUTES is the minority class** (7% of training data) — the classifier's REFUTES recall is the weakest. Treat low-confidence REFUTES predictions sceptically; verify by reading the abstract.
3. The model is small (`HistGradientBoostingClassifier` on top of frozen 384-dim e5-small-v2 embeddings). It cannot understand fine-grained negation or quantitative reversals like a fine-tuned RoBERTa would. Consider it a **first-pass** evidence sorter, not a final verdict.
4. **`P(SUPPORTS)` and `P(REFUTES)` are NOT confidence in whether the CLAIM is TRUE.** They are confidence in whether *this particular paper* takes that stance. The directional p-value is an aggregate over retrieved documents, not a clinical/statistical conclusion about the real world.

**Citing a result** — click the title to open the paper on Semantic Scholar (authors, journal, year, DOI, BibTeX export).
            """
        )

    gr.Examples(examples=examples, inputs=[query])
    button.click(search, [query], [summary_md, output])
    query.submit(search, [query], [summary_md, output])

    gr.Markdown(
        "---\n"
        "_Educational demo only — not a clinical decision-making tool._\n\n"
        "[Model](https://huggingface.co/andreiaalexa/scifact-relevance-classifier) · "
        "[Dataset](https://huggingface.co/datasets/andreiaalexa/scifact-relevance-pairs) · "
        "[Source](https://github.com/alexandreia/scifact-relevance-classifier)"
    )


if __name__ == "__main__":
    demo.launch()
