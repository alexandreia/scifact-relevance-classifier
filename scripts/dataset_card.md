---
license: mit
language:
- en
task_categories:
- text-classification
size_categories:
- 1K<n<10K
tags:
- scifact
- information-retrieval
- claim-verification
- relevance-classification
- scientific-text
configs:
- config_name: title
  data_files:
  - split: train
    path: title/train-*
  - split: test
    path: title/test-*
- config_name: abstract
  data_files:
  - split: train
    path: abstract/train-*
  - split: test
    path: abstract/test-*
- config_name: title_abstract
  data_files:
  - split: train
    path: title_abstract/train-*
  - split: test
    path: title_abstract/test-*
---

# SciFact Evidence Relevance Pairs

Custom (claim, document, label) pairs derived from [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) for binary **evidence relevance classification**: given a scientific claim and a candidate paper field, decide whether the paper is relevant evidence for the claim.

This dataset accompanies the [`scifact-relevance-classifier`](https://github.com/alexandreia/scifact-relevance-classifier) project, built as the Lab 3 / Assignment 1 deliverable for *Information Retrieval 5LN712* (Master's in Language Technology, Uppsala University, 2026).

## Quick start

```python
from datasets import load_dataset

# Three configs (one per document field variant)
ds = load_dataset("andreiaalexa/scifact-relevance-pairs", "title_abstract")
ds["train"][0]
# {
#   "claim": "...", "title": "...", "abstract": "...",
#   "document_text": "...", "label": "relevant", "label_id": 1, ...
# }
```

## Configurations

| Config | Document field used | Train | Test | Total |
|---|---|---:|---:|---:|
| `title`           | only the paper title           | 2,537 | 939 | 3,476 |
| `abstract`        | only the abstract              | 2,537 | 939 | 3,476 |
| `title_abstract`  | `"{title}. {abstract}"`        | 2,537 | 939 | 3,476 |

Class balance is approximately **64% `not_relevant` / 36% `relevant`** in both splits.

## How it was built

BEIR SciFact (Wadden et al., 2020) is originally a **retrieval** benchmark — each claim has one or more abstracts marked as evidence. This dataset converts that into a **classification** task by emitting one row per (claim, candidate-document) pair:

- **Positives**: every (claim, doc) pair that appears in the BEIR qrels for that claim.
- **Negatives**: a mix of two kinds, sampled per claim with `random_state=42`:
  - **Random negatives** (~2 per positive) — sampled uniformly from the corpus, excluding the claim's known positives. Teaches broad non-relevance.
  - **TF-IDF hard negatives** (~2 per positive) — top-similarity documents to the claim by TF-IDF cosine, excluding known positives. Teaches the model not to rely on lexical overlap alone. Same family of technique as the BM25 hard negatives in [DPR (Karpukhin et al., 2020)](https://aclanthology.org/2020.emnlp-main.550/).

The construction script lives at [`scifact_dataset.py`](https://github.com/alexandreia/scifact-relevance-classifier/blob/main/scifact_dataset.py) in the GitHub repo and is fully deterministic given the same seed.

## Columns

| Column | Type | Description |
|---|---|---|
| `split` | string | `train` or `test` (matches BEIR's split). Redundant with the HF split but kept for traceability. |
| `field_variant` | string | Which field is used for `document_text` in this config (`title`, `abstract`, or `title_abstract`). |
| `query_id` | string | BEIR query id. |
| `doc_id` | string | BEIR corpus doc id. |
| `claim` | string | The scientific claim text. |
| `title` | string | Document title (always populated for traceability). |
| `abstract` | string | Document abstract (always populated for traceability). |
| `document_text` | string | The exact text the classifier sees, depending on `field_variant`. |
| `label` | string | `relevant` or `not_relevant`. |
| `label_id` | int | `1` for `relevant`, `0` for `not_relevant`. |

## Intended use

- Training and benchmarking small embedding-based classifiers for scientific evidence retrieval / re-ranking.
- Educational use illustrating: (a) converting a retrieval benchmark into a classification dataset, (b) the role of hard-negative mining, (c) ablations across input field choices.

## Out-of-scope use

- This dataset is **not** suitable for clinical decision-making or any high-stakes scientific judgment. Labels are derived from BEIR qrels, which are themselves expert annotations on a small biomedical claim set; they do not generalise to arbitrary scientific domains.
- Predictions from models trained on this dataset should be treated as a *first-pass relevance signal*, not as ground truth about scientific evidence quality.

## License

MIT, matching the source code repository. The underlying SciFact corpus and qrels are released under [CC BY-NC 2.0](https://allenai.org/data/scifact) by AI2 — please cite Wadden et al. (2020) when using this dataset.

## Citations

If you use this dataset, please cite both the original SciFact paper and (optionally) this project:

```bibtex
@inproceedings{wadden-etal-2020-fact,
    title = "Fact or Fiction: Verifying Scientific Claims",
    author = "Wadden, David  and  Lin, Shanchuan  and  Lo, Kyle  and
              Wang, Lucy Lu  and  van Zuylen, Madeleine  and
              Cohan, Arman  and  Hajishirzi, Hannaneh",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    url = "https://aclanthology.org/2020.emnlp-main.609/",
}

@inproceedings{thakur-etal-2021-beir,
    title = "{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models",
    author = "Thakur, Nandan and Reimers, Nils and R{\"u}ckl{\'e}, Andreas and Srivastava, Abhishek and Gurevych, Iryna",
    booktitle = "NeurIPS 2021 Datasets and Benchmarks Track",
    year = "2021",
    url = "https://arxiv.org/abs/2104.08663",
}
```

## Acknowledgements

Built with [Hugging Face Datasets](https://github.com/huggingface/datasets) and scikit-learn. Course taught by Birger Moëll at Uppsala University.
