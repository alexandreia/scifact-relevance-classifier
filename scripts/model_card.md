---
license: mit
language:
- en
tags:
- text-classification
- information-retrieval
- claim-verification
- sentence-transformers
- scifact
- evidence-relevance
library_name: sklearn
pipeline_tag: text-classification
datasets:
- andreiaalexa/scifact-relevance-pairs
base_model: intfloat/e5-small-v2
model-index:
- name: scifact-relevance-classifier
  results:
  - task:
      type: text-classification
      name: Scientific Evidence Relevance Classification
    dataset:
      type: andreiaalexa/scifact-relevance-pairs
      name: SciFact Evidence Relevance Pairs (title+abstract config)
    metrics:
    - type: accuracy
      value: 0.883
    - type: f1
      value: 0.872
      name: macro-F1
---

# Scientific Evidence Relevance Classifier

Lightweight scikit-learn `HistGradientBoostingClassifier` trained on top of `intfloat/e5-small-v2` sentence embeddings to decide whether a scientific paper is **relevant evidence** for a scientific claim.

This model accompanies the [`scifact-relevance-classifier`](https://github.com/alexandreia/scifact-relevance-classifier) project, the Lab 3 / Assignment 1 deliverable for *Information Retrieval 5LN712* (Master's in Language Technology, Uppsala University, 2026).

## Headline numbers

| Metric | Value |
|---|---:|
| Macro-F1 (test) | **0.872** |
| Accuracy (test) | **0.883** |
| Test set size | 939 pairs |
| Class balance (test) | 64% `not_relevant`, 36% `relevant` |

## TL;DR

| | |
|---|---|
| **Task** | Binary classification: `relevant` / `not_relevant` |
| **Embedding** | `intfloat/e5-small-v2` (frozen, asymmetric query/passage prefixes) |
| **Pair features** | `[q, d, abs(q−d), q*d, cos(q,d)]` (1537 dims; InferSent recipe) |
| **Classifier** | `sklearn.ensemble.HistGradientBoostingClassifier` (max_iter=400, lr=0.05, class_weight=balanced) |
| **Best variant** | `title_abstract` (concatenated title + abstract) |
| **Training data** | [`andreiaalexa/scifact-relevance-pairs`](https://huggingface.co/datasets/andreiaalexa/scifact-relevance-pairs), config `title_abstract`, split `train` |
| **Code** | <https://github.com/alexandreia/scifact-relevance-classifier> |

## Usage

```python
import pickle
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

REPO = "andreiaalexa/scifact-relevance-classifier"

# 1) Download the trained classifier
clf_path = hf_hub_download(REPO, "classifier.pkl")
with open(clf_path, "rb") as f:
    clf = pickle.load(f)

# 2) Load the (public) embedding model used at training time
encoder = SentenceTransformer("intfloat/e5-small-v2")

# 3) Build pair features (E5 asymmetric prefixes, then InferSent recipe)
def pair_features(claims, documents):
    q = encoder.encode([f"query: {c}" for c in claims], normalize_embeddings=True)
    d = encoder.encode([f"passage: {p}" for p in documents], normalize_embeddings=True)
    cos = np.sum(q * d, axis=1, keepdims=True)
    return np.hstack([q, d, np.abs(q - d), q * d, cos])

# 4) Predict
claim = "Vitamin D supplementation reduces respiratory infections."
title = "Vitamin D supplementation to prevent acute respiratory tract infections."
abstract = "Randomized trials have evaluated whether vitamin D supplementation prevents acute respiratory tract infections in diverse populations."
document = f"{title}. {abstract}"

X = pair_features([claim], [document])
pred_id = int(clf.predict(X)[0])
proba = clf.predict_proba(X)[0]

label = {0: "not_relevant", 1: "relevant"}[pred_id]
print(f"prediction = {label} | P(relevant) = {proba[1]:.3f}")
```

The helper `scifact_features.pair_features(...)` is also bundled in this repo — you can download and import it directly instead of re-implementing the function above.

## Classifier comparison (full ablation)

Five classifier families compared. Best-in-row in **bold**.

**Macro-F1:**

| Variant | LogReg | LinSVM | RF | HGB | MLP |
|---|---:|---:|---:|---:|---:|
| title | 0.778 | 0.762 | 0.738 | 0.846 | **0.852** |
| abstract | 0.799 | 0.779 | 0.753 | **0.863** | 0.859 |
| title + abstract | 0.804 | 0.785 | 0.744 | **0.872** | 0.844 |

**Accuracy:**

| Variant | LogReg | LinSVM | RF | HGB | MLP |
|---|---:|---:|---:|---:|---:|
| title | 0.790 | 0.774 | 0.792 | 0.860 | **0.865** |
| abstract | 0.817 | 0.798 | 0.805 | **0.874** | 0.871 |
| title + abstract | 0.818 | 0.800 | 0.797 | **0.883** | 0.857 |

Full per-class precision/recall is in `experiment_results.json` in this repo.

**Key finding:** non-linear models (HGB, MLP) clearly beat linear ones (LogReg, LinearSVM), indicating that useful pairwise interactions between embedding dimensions exist beyond what a linear weighting can capture. Random Forest underperformed on macro-F1 despite balanced class weighting.

## Files in this repo

| File | Purpose |
|---|---|
| `classifier.pkl` | Trained sklearn pipeline: StandardScaler not needed (HGB handles raw features); the pickle is just `HistGradientBoostingClassifier`. |
| `metadata.json` | Best (variant, classifier) chosen during training; label maps; embedding model id. |
| `experiment_results.json` | Per-classifier, per-variant metrics including per-class precision/recall. |
| `scifact_features.py` | The exact `pair_features()` function used at training time. Import this for inference. |
| `README.md` | This card. |

## Intended use

- Re-ranking candidate scientific evidence retrieved by an upstream IR system (e.g., BM25 or DPR top-k → this classifier scores each candidate).
- Filtering noisy retrieval results.
- Educational use illustrating embedding-based pairwise classification.

## Out of scope / limitations

- **Not a clinical decision tool.** The training data is BEIR SciFact (biomedical claims with expert-annotated evidence). Predictions on non-biomedical scientific content may be unreliable.
- **English only.** `e5-small-v2` is multilingual-capable but the training data is English; cross-lingual performance is untested.
- **Frozen-encoder ceiling.** Because we don't fine-tune e5-small-v2, performance is bounded by what its general-purpose embeddings already capture about scientific text.
- **Hard-negative mining is TF-IDF based**, which means the model learns to distinguish lexically-similar but irrelevant documents — but it may be overconfident on completely off-topic documents.

## Citations

If you use this model, please cite both the underlying dataset (SciFact / BEIR) and the embedding model:

```bibtex
@inproceedings{wadden-etal-2020-fact,
  title = "Fact or Fiction: Verifying Scientific Claims",
  author = "Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and van Zuylen, Madeleine and Cohan, Arman and Hajishirzi, Hannaneh",
  booktitle = "EMNLP", year = "2020",
  url = "https://aclanthology.org/2020.emnlp-main.609/"
}

@article{wang-etal-2022-text-embeddings,
  title = "Text Embeddings by Weakly-Supervised Contrastive Pre-training",
  author = "Wang, Liang and Yang, Nan and Huang, Xiaolong and Jiao, Binxing and Yang, Linjun and Jiang, Daxin and Majumder, Rangan and Wei, Furu",
  journal = "arXiv:2212.03533", year = "2022",
  url = "https://arxiv.org/abs/2212.03533"
}

@inproceedings{conneau-etal-2017-supervised,
  title = "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data",
  author = "Conneau, Alexis and Kiela, Douwe and Schwenk, Holger and Barrault, Lo{\"i}c and Bordes, Antoine",
  booktitle = "EMNLP", year = "2017",
  url = "https://aclanthology.org/D17-1070/"
}
```

## License

MIT.
