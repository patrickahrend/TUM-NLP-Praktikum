import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import umap

from utils import load_pickle


def create_umap_single(embeddings, labels, emb_type, file_path):
    """
    Create a UMAP visualization for a single type of embeddings.

    Parameters
    ----------
    embeddings : array-like
        The embeddings to visualize.
    labels : array-like
        The labels for the embeddings.
    emb_type : str
        The type of the embeddings.
    file_path : Path
        The path where the visualization will be saved.
    """
    umap_model = umap.UMAP(metric="euclidean")
    embedding_umap = umap_model.fit_transform(embeddings)
    fig = create_plot(embedding_umap, labels, emb_type)
    fig.write_html(str(file_path.with_suffix(".html")))


def create_umap_combined(emb_proc_desc, emb_legal_text, labels, emb_type, file_path):
    """
    Create a UMAP visualization for combined embeddings.

    Parameters
    ----------
    emb_proc_desc : array-like
        The embeddings for the process descriptions.
    emb_legal_text : array-like
        The embeddings for the legal texts.
    labels : array-like
        The labels for the embeddings.
    emb_type : str
        The type of the embeddings.
    file_path : Path
        The path where the visualization will be saved.
    """
    combined_embeddings = np.concatenate([emb_proc_desc, emb_legal_text])
    umap_model = umap.UMAP(metric="euclidean")
    umap_embeddings = umap_model.fit_transform(combined_embeddings)

    types = ["Legal Text"] * len(emb_legal_text) + ["Process Description"] * len(
        emb_proc_desc
    )
    df_umap = pd.DataFrame(
        {
            "UMAP_1": umap_embeddings[:, 0],
            "UMAP_2": umap_embeddings[:, 1],
            "Type": types,
            "Process": np.concatenate([labels, labels]),
        }
    )

    fig = create_plot(df_umap, labels, emb_type, types)
    fig.write_html(str(file_path.with_suffix(".html")))


def create_plot(data, labels, emb_type, types=None):
    """
    Create a plotly express scatter plot for the UMAP visualization.

    Parameters
    ----------
    data : array-like or DataFrame
        The data to plot.
    labels : array-like
        The labels for the data.
    emb_type : str
        The type of the embeddings.
    types : array-like, optional
        The types of the embeddings.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The created figure.
    """
    if types is None:
        fig = px.scatter(
            x=data[:, 0],
            y=data[:, 1],
            color=labels,
            labels={
                "color": "Process",
                "x": "UMAP Dimension 1",
                "y": "UMAP Dimension 2",
            },
            title=f"UMAP Visualization of {emb_type} Embeddings",
        )
    else:
        fig = px.scatter(
            data,
            x="UMAP_1",
            y="UMAP_2",
            color="Process",
            symbol="Type",
            labels={"UMAP_1": "UMAP Dimension 1", "UMAP_2": "UMAP Dimension 2"},
            title=f"UMAP Visualization of {emb_type} Embeddings (Process Description vs Legal Text)",
        )
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig


def main():
    """
    Main function to load embeddings, create UMAP visualizations, and save the visualizations.
    """
    base_path = Path(__file__).resolve().parents[2]
    df_train = pd.read_csv(base_path / "data/processed/training_data_preprocessed.csv")

    embedding_files = os.listdir(base_path / "data/processed/embeddings")

    for emb_type in ["fasttext", "word2vec", "tfidf"]:
        combined_files = [
            f
            for f in embedding_files
            if "train_combined" in f and f.startswith(emb_type)
        ]
        seperate_files = [
            f
            for f in embedding_files
            if "train_separate" in f and f.startswith(emb_type)
        ]
        legal_files = [
            f
            for f in embedding_files
            if "train_legal_text" in f and f.startswith(emb_type)
        ]

        # combined embeddings
        for file_name in combined_files:
            file_path = base_path / f"data/processed/embeddings/{file_name}"
            embedding_combined = load_pickle(file_path)
            # remove unnecessary columns
            labels = embedding_combined["Process"]
            columns_to_drop = [
                "Text",
                "Label",
                "Process",
                "Process_description",
                "Combined_Text",
            ]
            embedding_df = embedding_combined.drop(columns=columns_to_drop)

            file_map = base_path / f"references/umap/{emb_type}_umap_combined_increased"
            create_umap_single(embedding_df, labels, emb_type, file_map)

        embedding_dims = {
            "glove": 300,
            "fasttext": 300,
            "word2vec": 300,
            "tfidf": 1300,
            "gpt": 1536,
            "bert": 768,
        }
        # separate embeddings
        for file_name in seperate_files:
            file_path = base_path / f"data/processed/embeddings/{file_name}"
            embedding_separate = load_pickle(file_path)

            # remove unnecessary columns
            columns_to_drop = [
                "Text",
                "Label",
                "Process",
                "Process_description",
                "Combined_Text",
            ]
            labels = embedding_separate["Process"]

            embedding_df = embedding_separate.drop(columns=columns_to_drop)

            dim = embedding_dims[emb_type]

            # Slice the DataFrame for process description embeddings
            embedding_proc_desc = embedding_df.iloc[:, :dim]

            # Slice the DataFrame for legal text embeddings
            embedding_legal_text = embedding_df.iloc[:, dim : dim * 2]

            file_map = (
                base_path / f"references/umap/{emb_type}_umap_separate_single_increased"
            )
            file_multiple = (
                base_path
                / f"references/umap/{emb_type}_umap_separate_multiple_increased"
            )

            create_umap_single(embedding_df, labels, emb_type, file_map)
            create_umap_combined(
                embedding_proc_desc,
                embedding_legal_text,
                embedding_separate["Process"],
                emb_type,
                file_multiple,
            )

        # just legal text
        for file in legal_files:
            file_path = base_path / f"data/processed/embeddings/{file}"
            embedding_legal_text = load_pickle(file_path)

            labels = embedding_legal_text["Process"]
            columns_to_drop = [
                "Text",
                "Label",
                "Process",
                "Process_description",
                "Combined_Text",
            ]
            embedding_legal_text = embedding_legal_text.drop(columns=columns_to_drop)
            file_legal = (
                base_path / f"references/umap/{emb_type}_umap_legal_text_increased"
            )
            create_umap_single(embedding_legal_text, labels, emb_type, file_legal)


if __name__ == "__main__":
    main()
