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

## Project Idea

Scientific evidence search is difficult because two texts can be about the same scientific claim without using the same words. A keyword match is not enough. For example, a paper may discuss vaccination and autism but still conclude that there is no evidence for a causal relationship.

The main challenge is to use semantic embeddings to compare a claim with a scientific title or abstract, then train a classifier that can decide whether the document is useful evidence.

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

For each claim-document pair, I encode the claim as `q` and the document as `d`, then build this feature vector:

```text
[q, d, abs(q - d), q * d, cosine(q, d)]
```

This gives the classifier information about the two texts, their distance, their interaction, and their cosine similarity.

## Models

Several classifiers are trained and compared:

- Logistic Regression
- Linear SVM
- Random Forest
- HistGradientBoosting
- MLP neural network

The reason for training several classifiers is to test which type of decision boundary works best on the embedding features. Linear models are useful baselines, while gradient boosting and MLP can learn non-linear interactions between the claim and document embeddings.

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

The demo Space uses the 3-class stance classifier:

- `SUPPORTS`
- `REFUTES`
- `NEI` (not enough information)

The stance task is harder than binary relevance. The best stance model is Logistic Regression on title + abstract with macro-F1 **0.533**.

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

## Limitations

- The corpus is biomedical and based on SciFact, so results are less reliable outside this domain.
- The stance model is weaker than the binary relevance model because support/refutation is a harder task and the classes are imbalanced.
- The demo is for educational use only and should not be used for medical decision-making.
- The statistical summary in the demo is based on retrieved documents and model predictions, not on manually extracted statistical results from the papers.

## AI Tool Reflection

AI coding tools were used to help write and debug code, structure the project, and improve the documentation. I still had to check the code, understand the dataset construction, verify the model outputs, and make sure the explanation matched what the system actually does. One important lesson was that AI tools can generate useful code quickly, but they can also make a project look more complete than it really is unless all files, links, and results are checked manually.

## References

- Wadden et al. (2020). *Fact or Fiction: Verifying Scientific Claims*. EMNLP. <https://aclanthology.org/2020.emnlp-main.609/>
- Thakur et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. <https://arxiv.org/abs/2104.08663>
- Wang et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. <https://arxiv.org/abs/2212.03533>
- Conneau et al. (2017). *Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*. <https://aclanthology.org/D17-1070/>

## License

MIT License. See [LICENSE](LICENSE).
