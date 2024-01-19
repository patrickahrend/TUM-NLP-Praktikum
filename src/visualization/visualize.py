from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import umap
import umap.plot

from utils import load_pickle


def create_umap(embeddings, labels):
    embedding_umap = umap.UMAP(metric="euclidean").fit(embeddings)

    fig, ax = plt.subplots()  # Create a Figure and Axes
    umap.plot.points(embedding_umap, labels=labels, ax=ax)
    return fig


def save_visualization(plot, base_path, emb_type):
    file_path = base_path / f"references/umap/{emb_type}_umap.png"
    plot.savefig(file_path)


def main():
    base_path = Path(__file__).resolve().parents[2]

    df_train = pd.read_csv(base_path / "data/processed/training_data_preprocessed.csv")
    for emb_type in ["gpt", "ft", "w2v", "glove", "bert", "tfidf"]:
        embedding = load_pickle(
            base_path / f"data/processed/embeddings/{emb_type}_train.pkl"
        )
        fig = create_umap(embedding, df_train["Process"])
        save_visualization(fig, base_path, emb_type)


if __name__ == "__main__":
    main()
