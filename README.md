# scifact-relevance-classifier

> Embeddings-based binary classifier that decides whether a scientific paper is **relevant evidence** for a scientific claim. Built on BEIR SciFact, `intfloat/e5-small-v2` embeddings, and scikit-learn.

[![Hugging Face Dataset](https://img.shields.io/badge/HF-dataset-yellow)](https://huggingface.co/datasets/USERNAME/scifact-relevance-pairs)
[![Hugging Face Model](https://img.shields.io/badge/HF-model-yellow)](https://huggingface.co/USERNAME/scifact-relevance-classifier)
[![Hugging Face Space](https://img.shields.io/badge/HF-space-blue)](https://huggingface.co/spaces/USERNAME/scifact-relevance-classifier)

> Course project for **Information Retrieval 5LN712**, Master's in Language Technology, Uppsala University. The badges above will resolve once the Hugging Face artifacts are published.

## Why this project

Scientific search is not keyword matching. A claim and an abstract can describe the same phenomenon in different words; a lexically similar abstract can still be irrelevant evidence. The classifier here learns from *embedding features* (semantic geometry) rather than raw word overlap, and is explicitly trained against TF-IDF hard negatives so it cannot rely on shallow lexical similarity.

## Headline results

Best model: `HistGradientBoosting` on the `title + abstract` field variant.

| Variant | LogReg | LinSVM | RF | HGB | MLP |
|---|---:|---:|---:|---:|---:|
| title | 0.778 | 0.762 | 0.738 | 0.846 | **0.852** |
| abstract | 0.799 | 0.779 | 0.753 | **0.863** | 0.859 |
| title + abstract | 0.804 | 0.785 | 0.744 | **0.872** | 0.844 |

*Macro-F1 on a 939-pair held-out test set. Test set is 64% `not_relevant` / 36% `relevant`, so macro-F1 is the primary metric. Full per-class precision/recall is in `artefacts_scifact/experiment_results.json`.*

**Key finding:** non-linear classifiers (HGB, MLP) clearly beat linear ones (LogReg, LinearSVM) on the InferSent-style pair-feature space, indicating that useful pairwise interactions exist between embedding dimensions that linear models cannot exploit.

## Quick start

```bash
git clone https://github.com/USERNAME/scifact-relevance-classifier.git
cd scifact-relevance-classifier

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Build the custom claim-document dataset from BEIR SciFact (~1 min)
python scifact_dataset.py

# 2. Train and compare 5 classifiers across 3 field variants (~5-10 min on CPU)
python train_scifact_classifier.py

# 3. (optional) Run the concat-only baseline ablation
python baseline_scifact.py

# 4. Launch the local Gradio demo
python app_scifact.py
```

The trained classifier (`artefacts_scifact/classifier.pkl`) and the local copy of the embedding model (`artefacts_scifact/embedding_model/`) are produced by step 2; both are gitignored. The demo loads them automatically.

## How it works

The pipeline has four conceptual stages, each represented by a single Python module.

```
BEIR SciFact (HF Hub)
      │
      ▼
scifact_dataset.py    Convert retrieval qrels into (claim, doc, label) pairs.
                      Negatives = random + TF-IDF hard negatives.
      │
      ▼
scifact_features.py   Encode each pair into a 1537-dim feature vector:
                          [q, d, |q-d|, q*d, cos(q,d)]
                      using intfloat/e5-small-v2 with E5 asymmetric prefixes.
      │
      ▼
train_scifact_classifier.py
                      Train 5 sklearn classifiers (LogReg, LinearSVC, RF,
                      HistGradientBoosting, MLP) per field variant. Save best.
      │
      ▼
app_scifact.py        Gradio demo: paste a claim + a candidate paper,
                      get a `relevant` / `not_relevant` prediction.
```

`baseline_scifact.py` runs an honest ablation: it strips the InferSent recipe down to plain `[q, d]` concatenation (768 dims), so the report can quantify how much the `|q-d|`, `q*d`, and cosine features actually contribute.

For a deeper walkthrough of every design decision, see [SCIENTIFIC_RELEVANCE_PROJECT.md](SCIENTIFIC_RELEVANCE_PROJECT.md).

## Project structure

```
.
├── README.md                          # this file
├── SCIENTIFIC_RELEVANCE_PROJECT.md    # design doc with full results table
├── requirements.txt
├── LICENSE
├── scifact_config.py                  # central config (model, paths, seed)
├── scifact_dataset.py                 # build custom CSV dataset from BEIR
├── scifact_features.py                # InferSent-style pair features
├── train_scifact_classifier.py        # train + evaluate 5 classifiers
├── baseline_scifact.py                # concat-only ablation
├── app_scifact.py                     # Gradio demo
├── artefacts_scifact/
│   ├── experiment_results.json        # full per-class metrics (committed)
│   ├── metadata.json                  # best (variant, classifier) (committed)
│   ├── classifier.pkl                 # trained sklearn pipeline (gitignored)
│   └── embedding_model/               # local copy of e5-small-v2 (gitignored)
├── data_scifact/
│   └── scifact_pairs_*.csv            # generated CSVs (gitignored)
├── scripts/                           # HF Hub publish helpers
├── report/                            # 2-page academic report
└── .gitignore
```

## Hugging Face artifacts

| Artifact | Link |
|---|---|
| Dataset (custom claim-document pairs) | _to be published_ |
| Model (trained classifier + embedding model card) | _to be published_ |
| Demo Space (Gradio app) | _to be published_ |

## Report

The 2-page academic report lives in `report/report.pdf` (and the LaTeX/Markdown source alongside).

## Methodology citations

- **SciFact** — Wadden, D. et al. (2020). *Fact or Fiction: Verifying Scientific Claims*. EMNLP. <https://aclanthology.org/2020.emnlp-main.609/>
- **BEIR benchmark** — Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. NeurIPS Datasets and Benchmarks Track. <https://arxiv.org/abs/2104.08663>
- **E5 embeddings** — Wang, L. et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. <https://arxiv.org/abs/2212.03533>
- **InferSent pair-feature recipe** — Conneau, A. et al. (2017). *Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*. EMNLP. <https://aclanthology.org/D17-1070/>
- **Sentence-BERT** — Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP. <https://aclanthology.org/D19-1410/>
- **Hard negative mining (DPR)** — Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP. <https://aclanthology.org/2020.emnlp-main.550/>
- **Trees vs. neural on tabular** — Shwartz-Ziv, R. & Armon, A. (2022). *Tabular Data: Deep Learning is Not All You Need*. <https://arxiv.org/abs/2106.03253>

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built as the Lab 3 / Assignment 1 deliverable for *Information Retrieval 5LN712* at Uppsala University, taught by Birger Moëll. AI coding tools (Claude) were used during development; see the report's reflection section.
