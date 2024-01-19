from pathlib import Path

import pandas as pd
import plotly.express as px
import umap
import umap.plot

from utils import load_pickle


def create_umap(embeddings, labels, emb_type, file_path):
    embedding_umap = umap.UMAP(metric="euclidean").fit_transform(embeddings)

    fig = px.scatter(
        x=embedding_umap[:, 0],
        y=embedding_umap[:, 1],
        color=labels,
        labels={"x": "UMAP Dimension 1", "y": "UMAP Dimension 2"},
        title=f"UMAP Visualization of {emb_type} Embeddings",
    )

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

    fig.write_html(str(file_path.with_suffix(".html")))


def main():
    base_path = Path(__file__).resolve().parents[2]

    df_train = pd.read_csv(base_path / "data/processed/training_data_preprocessed.csv")
    for emb_type in ["gpt", "ft", "w2v", "glove", "bert", "tfidf"]:
        embedding = load_pickle(
            base_path / f"data/processed/embeddings/{emb_type}_train.pkl"
        )
        file_path = base_path / f"references/umap/{emb_type}_umap.png"
        create_umap(embedding, df_train["Process"], emb_type, file_path)


if __name__ == "__main__":
    main()
