# Scientific Evidence Relevance Classifier

## Project Goal

Train a lightweight classifier on top of embeddings to decide whether a
scientific paper is relevant evidence for a scientific claim.

## Problem

Scientific search is not only keyword matching. A claim and an abstract can use
different wording while describing the same phenomenon, and a keyword-similar
abstract can still be irrelevant. The classifier learns from embedding features
instead of raw words.

## Dataset

The custom dataset is derived from BEIR SciFact:

- claim: query text
- document: title, abstract, or title + abstract
- label: `relevant` if BEIR marks the paper as evidence for the claim,
  otherwise `not_relevant`

Negatives are sampled in two ways:

- random negatives, to teach broad non-relevance
- TF-IDF hard negatives, to teach the model not to trust lexical overlap alone

## Embedding Features

For each claim-document pair:

```text
q = embedding(claim)
d = embedding(document)

features = [q, d, abs(q - d), q * d, cosine(q, d)]
```

This is a common pair-classification pattern: the classifier sees both semantic
representations, their distance, their dimension-wise interaction, and a direct
similarity score.

## Experiments

The same classifier setup is trained for three document-field variants:

1. title only
2. abstract only
3. title + abstract

Primary metrics:

- accuracy
- macro-F1
- per-class precision/recall

Five classifier families compared with `intfloat/e5-small-v2` and the
InferSent pair-feature recipe `[q, d, |q-d|, q*d, cos(q,d)]` (1537 dims).
Negatives: two random + two TF-IDF hard per positive.

**Macro-F1 by (variant x classifier):**

| Variant | LogReg | LinSVM | RF | HGB | MLP |
|---|---:|---:|---:|---:|---:|
| title | 0.778 | 0.762 | 0.738 | 0.846 | **0.852** |
| abstract | 0.799 | 0.779 | 0.753 | **0.863** | 0.859 |
| title + abstract | 0.804 | 0.785 | 0.744 | **0.872** | 0.844 |

**Accuracy by (variant x classifier):**

| Variant | LogReg | LinSVM | RF | HGB | MLP |
|---|---:|---:|---:|---:|---:|
| title | 0.790 | 0.774 | 0.792 | 0.860 | **0.865** |
| abstract | 0.817 | 0.798 | 0.805 | **0.874** | 0.871 |
| title + abstract | 0.818 | 0.800 | 0.797 | **0.883** | 0.857 |

**Best overall:** `HistGradientBoosting` on `title + abstract` →
macro-F1 = 0.872, accuracy = 0.883. This is the model saved in
`artefacts_scifact/classifier.pkl` and served by the Gradio demo.

**Interpretation:**

1. *Title + abstract beats abstract beats title.* More text → more signal,
   as expected. The lift from abstract to title+abstract is small (~0.01 F1),
   suggesting the abstract carries most of the evidential content.
2. *Non-linear models clearly beat linear models.* HGB and MLP push past
   0.84 macro-F1; LogReg and LinearSVM cap out around 0.80. Evidence that
   the embedding feature space contains useful non-linear interactions.
3. *Random Forest is a surprising loser.* High accuracy (0.79-0.81) but low
   macro-F1 (0.74-0.75) means RF is biased toward the majority class
   (`not_relevant`) despite `class_weight="balanced"`. RFs underperform on
   dense, mid-dimensional features where optimal split thresholds are smooth.
4. *HGB >= MLP on macro-F1 in 2 of 3 variants.* Consistent with Shwartz-Ziv
   & Armon (2022): gradient-boosted trees often beat neural nets on
   small-to-mid tabular datasets (here ~2.5k rows).

## Commands

```bash
python scifact_dataset.py
python train_scifact_classifier.py
python app_scifact.py
```

## References

- SciFact: Wadden et al., "Fact or Fiction: Verifying Scientific Claims",
  ACL Anthology: https://aclanthology.org/2020.emnlp-main.609/
- BEIR SciFact on Hugging Face:
  https://huggingface.co/datasets/BeIR/scifact
- BEIR qrels on Hugging Face:
  https://huggingface.co/datasets/BeIR/scifact-qrels
- BEIR: Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
  of Information Retrieval Models"
- E5: Wang et al., "Text Embeddings by Weakly-Supervised Contrastive
  Pre-training": https://www.microsoft.com/en-us/research/publication/text-embeddings-by-weakly-supervised-contrastive-pre-training/
