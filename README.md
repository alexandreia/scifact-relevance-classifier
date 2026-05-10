# SciFact Evidence Classifier

Student project for **Information Retrieval 5LN712**, Master's Programme in Language Technology, Uppsala University.

This project trains classifiers on top of text embeddings to classify scientific evidence. Given a scientific claim and a candidate paper, the system predicts whether the paper is relevant evidence. The Hugging Face demo also retrieves papers from the SciFact corpus and predicts whether they support, refute, or do not clearly address the claim.

## Links

| Resource | Link |
|---|---|
| GitHub repository | <https://github.com/alexandreia/scifact-relevance-classifier> |
| Hugging Face dataset | <https://huggingface.co/datasets/andreiaalexa/scifact-relevance-pairs> |
| Hugging Face model | <https://huggingface.co/andreiaalexa/scifact-relevance-classifier> |
| Hugging Face demo Space | <https://huggingface.co/spaces/andreiaalexa/scifact-relevance-classifier> |

## Dataset

The custom dataset is built from **SciFact / BEIR SciFact**. The original dataset is a scientific retrieval benchmark. I converted it into classification examples:

```text
claim + document text -> label
```

For the binary relevance task, the labels are:

- `relevant`
- `not_relevant`

Negative examples are created using:

- random negative documents
- TF-IDF hard negatives that are lexically similar but not labelled as evidence

This makes the task harder and helps test whether the embedding classifier learns more than simple word overlap.

## Embeddings and Features

The project uses the embedding model:

```text
intfloat/e5-small-v2
```

## Method

The pipeline has four steps:

1. Convert SciFact into claim-document classification pairs.
2. Encode claims and documents with `intfloat/e5-small-v2`.
3. Build pair features: `[q, d, abs(q - d), q * d, cosine(q, d)]`.
4. Train and compare several classifiers using train/test splits.

## Models

Several classifiers are trained and compared:

- Logistic Regression
- Linear SVM
- Random Forest
- HistGradientBoosting
- MLP neural network

## Results

The best binary relevance model was **HistGradientBoosting** using **title + abstract** as input.

Macro-F1 on the held-out test set:

| Input variant | Logistic Regression | Linear SVM | Random Forest | HistGradientBoosting | MLP |
|---|---:|---:|---:|---:|---:|
| title | 0.778 | 0.762 | 0.738 | 0.846 | **0.852** |
| abstract | 0.799 | 0.779 | 0.753 | **0.863** | 0.859 |
| title + abstract | 0.804 | 0.785 | 0.744 | **0.872** | 0.844 |

Main interpretation:

- `title + abstract` gives the best result because the abstract contains more evidence information.
- Non-linear models perform better than linear models for binary relevance.
- The best result is macro-F1 **0.872**.

## How to Run Locally

Create an environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Build the binary relevance dataset:

```bash
python scifact_dataset.py
```

Train and evaluate the binary relevance classifiers:

```bash
python train_scifact_classifier.py
```

Build the stance dataset:

```bash
python scifact_dataset_stance.py
```

Train and evaluate the stance classifiers:

```bash
python train_stance_classifier.py
```

Run the local Gradio relevance demo:

```bash
python app_scifact.py
```

## Project Structure

```text
.
├── README.md
├── report/
│   └── report.md
├── scifact_dataset.py              # builds the binary relevance dataset
├── scifact_dataset_stance.py       # builds the 3-class stance dataset
├── scifact_features.py             # creates embedding pair features
├── train_scifact_classifier.py     # trains/evaluates binary classifiers
├── train_stance_classifier.py      # trains/evaluates stance classifiers
├── app_scifact.py                  # local Gradio demo
├── scripts/
│   ├── space_app.py                # Hugging Face Space app
│   ├── push_dataset.py             # uploads dataset to Hugging Face
│   ├── push_model.py               # uploads trained model artifacts
│   └── push_space.py               # uploads the demo Space
├── artefacts_scifact/              # saved binary results and retrieval files
├── artefacts_stance/               # saved stance model and results
├── data_scifact_stance/            # generated stance CSV files
└── requirements.txt
```

## License

MIT License. See [LICENSE](LICENSE).
