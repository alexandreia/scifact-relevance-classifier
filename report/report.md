# SciFact Evidence Classifier: Embedding-Based Classification of Scientific Evidence

**Course:** Information Retrieval 5LN712, Uppsala University  
**Student project repository:** <https://github.com/alexandreia/scifact-relevance-classifier>  
**Hugging Face dataset:** <https://huggingface.co/datasets/andreiaalexa/scifact-relevance-pairs>  
**Hugging Face model:** <https://huggingface.co/andreiaalexa/scifact-relevance-classifier>  
**Hugging Face demo:** <https://huggingface.co/spaces/andreiaalexa/scifact-relevance-classifier>

## 1. Introduction

The goal of this project is to train classifiers on top of embedding models. I have chosen a scientific evidence classification task. The selected domain is medical scientific evidence search. Given a scientific claim and a candidate paper title or abstract, the system should predict whether the paper is relevant evidence for the claim. The Hugging Face demo extends this idea by retrieving related papers from the SciFact corpus and predicting whether each retrieved paper supports, refutes, or gives no clear evidence for the user query.

This problem is suitable for embeddings because scientific claims and papers often express related ideas using different wording. A keyword-based system may retrieve documents that share surface terms but do not provide evidence for the claim. The challenge is therefore to at leats attempt to encode claims and documents in a useful manner and train a classifier that can use the text representation to distinguish evidence from irrelevant or only lexically similar documents. 

## 2. Dataset Creation

The custom dataset was created by processing SciFact / BEIR SciFact, which is originally a scientific retrieval benchmark, into supervised text classification examples. Each example has the form:

```text
claim + document text -> label
```

For the binary relevance task, the labels are `relevant` and `not_relevant`. Relevant examples are derived from SciFact evidence annotations. Not-relevant examples are generated using two methods: random negative sampling and TF-IDF hard negative mining. The hard negatives are documents that are lexically similar to the claim but are not labelled as evidence in the dataset. This makes the dataset more useful for training because the classifier cannot rely only on word overlap.

I also created a 3-class stance dataset for the demo with the labels `SUPPORTS`, `REFUTES`, and `NEI` (not enough information). This version prove much harder to design because the model must distinguish the direction of evidence, not only whether a paper is relevant.

## 3. Method

The embedding model used in the project is `intfloat/e5-small-v2`. For each claim-document pair, the claim is encoded as `q` and the document as `d`. The final feature vector is:

```text
[q, d, abs(q - d), q * d, cosine(q, d)]
```

This feature representation gives the classifier access to the original embedding vectors, their distance, their dimension-wise interaction, and their cosine similarity. I trained and compared five classifiers: Logistic Regression, Linear SVM, Random Forest, HistGradientBoosting, and a small MLP neural network. The purpose of comparing several classifiers was to test what type of decision boundary works best on the embedding features. Linear models act as simple baselines, while gradient boosting and MLP can learn non-linear interactions between the query and document embeddings.

The classifiers were evaluated using training and test splits. Macro-F1 was used as the main metric because the labels are not perfectly balanced (sqewed).

## 4. Results

For the binary relevance task, the best model was HistGradientBoosting using title + abstract as input. The main results are shown below.

| Input variant | Logistic Regression | Linear SVM | Random Forest | HistGradientBoosting | MLP |
|---|---:|---:|---:|---:|---:|
| title | 0.778 | 0.762 | 0.738 | 0.846 | **0.852** |
| abstract | 0.799 | 0.779 | 0.753 | **0.863** | 0.859 |
| title + abstract | 0.804 | 0.785 | 0.744 | **0.872** | 0.844 |

The best macro-F1 score was **0.872**. The results show that title + abstract gives the strongest performance, which is expected because the abstract contains more evidence information than the title alone. The non-linear models, especially HistGradientBoosting and MLP, outperformed the linear baselines. This suggests that useful relevance information is captured through interactions between embedding dimensions rather than by a simple linear boundary.

The stance task was more difficult. The best results for this was with the Logistic Regression model on title + abstract with macro-F1 **0.533**. This lower result is expected because support/refutation classification is more semantically complex and the `REFUTES` class is underrepresented in the dataset.

## 5. Demo and Limitations

The working demo is hosted on Hugging Face Spaces. To test it, enter a statement/query such as "Coffee consumption causes humans to photosynthesize in direct sunlight." The system attempts to retrieve semantically similar SciFact papers, predict their stance, and link to the sources. The demo also includes a document-level directional evidence summary over confident `SUPPORTS` and `REFUTES` predictions.

This is still in the experimentation phase, hence there are important limitations to mention. The corpus is biomedical and relatively small, so the system should not be used outside this domain without caution. The stance model (supports vs not supports) is less reliable than the binary relevance model. The statistical summary in the demo is based on retrieved documents and model predictions; it is not the same as extracting p-values or effect sizes from the original papers. The demo is therefore educational and should not be used for medical decision-making or even fact checking fo rreal life situations.

## 6. Reflection on AI Tools

AI coding tools were useful for developing the project structure, writing code, debugging, and improving documentation. They helped speed up implementation, especially for repetitive parts. However, the tools also required careful checking what the model results were actually produced by the scripts. A key lesson is that AI can make coding faster, but understanding the dataset, model behavior, and limitations is still necessary.

## References

Wadden, D., Lin, S., Lo, K., Wang, L. L., van Zuylen, M., Cohan, A., & Hajishirzi, H. (2020). *Fact or Fiction: Verifying Scientific Claims*. EMNLP. <https://aclanthology.org/2020.emnlp-main.609/>

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*. <https://arxiv.org/abs/2104.08663>

Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training*. <https://arxiv.org/abs/2212.03533>

Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017). *Supervised Learning of Universal Sentence Representations from Natural Language Inference Data*. <https://aclanthology.org/D17-1070/>
