"""Gradio demo for the SciFact embedding relevance classifier."""

import json
import os
import pickle
from pathlib import Path

import gradio as gr
from sentence_transformers import SentenceTransformer

from scifact_features import pair_features


ARTEFACTS = Path("artefacts_scifact")


def load_artifacts():
    with open(ARTEFACTS / "classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    with open(ARTEFACTS / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    model = SentenceTransformer(str(ARTEFACTS / "embedding_model"))
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    return clf, model, meta, id2label


clf, embedding_model, metadata, id2label = load_artifacts()


def classify(claim, title, abstract):
    claim = claim.strip()
    title = title.strip()
    abstract = abstract.strip()
    if not claim or not (title or abstract):
        return "Enter a claim and at least a title or abstract."

    field_variant = metadata["field_variant"]
    if field_variant == "title":
        document = title
    elif field_variant == "abstract":
        document = abstract
    else:
        document = f"{title}. {abstract}".strip()

    X = pair_features(embedding_model, [claim], [document])
    pred = int(clf.predict(X)[0])
    label = id2label[pred]

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        rows = "\n".join(
            f"| {id2label[i]} | {proba[i] * 100:.1f}% |"
            for i in sorted(id2label)
        )
        return f"### `{label}`\n\n| Class | Probability |\n|---|---:|\n{rows}"

    score = clf.decision_function(X)[0]
    return f"### `{label}`\n\nDecision score: `{score:.3f}`"


examples = [
    [
        "Coffee consumption causes humans to photosynthesize in direct sunlight.",
        "Coffee consumption and health outcomes: an umbrella review.",
        "Coffee consumption has been studied in relation to cardiovascular, metabolic, and mortality outcomes. No evidence suggests photosynthesis in humans.",
    ],
    [
        "Vitamin D supplementation reduces the risk of respiratory infection.",
        "Vitamin D supplementation to prevent acute respiratory tract infections.",
        "Randomized trials have evaluated whether vitamin D supplementation prevents acute respiratory tract infections in diverse populations.",
    ],
]


with gr.Blocks(title="Scientific Evidence Relevance Classifier") as demo:
    gr.Markdown("# Scientific Evidence Relevance Classifier")
    gr.Markdown(
        "Embedding-based classifier trained on SciFact-derived claim-document pairs."
    )
    claim = gr.Textbox(label="Scientific claim", lines=2)
    title = gr.Textbox(label="Candidate paper title", lines=1)
    abstract = gr.Textbox(label="Candidate abstract", lines=6)
    button = gr.Button("Classify relevance", variant="primary")
    output = gr.Markdown()
    gr.Examples(examples=examples, inputs=[claim, title, abstract])
    button.click(classify, [claim, title, abstract], output)


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7960"))
    demo.launch(server_name="127.0.0.1", server_port=port)
