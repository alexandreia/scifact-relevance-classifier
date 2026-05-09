"""Configuration for the SciFact embedding-classifier project."""

from pathlib import Path


PROJECT_NAME = "Scientific Evidence Relevance Classifier"

DATASET_CORPUS = "BeIR/scifact"
DATASET_QRELS = "BeIR/scifact-qrels"

EMBEDDING_MODEL = "intfloat/e5-small-v2"

FIELD_VARIANTS = ("title", "abstract", "title_abstract")
LABEL2ID = {"not_relevant": 0, "relevant": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

OUT_DIR = Path("artefacts_scifact")
DATA_DIR = Path("data_scifact")

RANDOM_SEED = 42
