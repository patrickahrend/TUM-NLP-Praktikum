from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import umap


def create_umap_single(
    embeddings: pd.DataFrame,
    labels: pd.Series,
    emb_type: str,
    title_extension: str,
    file_path: Path,
) -> None:
    """
    Create a UMAP visualization for a single type of embeddings.

    Parameters
    ----------
    embeddings : pandas DataFrame
        The embeddings to visualize.
    labels : pandas Series
        The labels for the embeddings.
    emb_type : str
        The type of the embeddings.
    title_extension : str
        The extension for the title of the visualization.
    file_path : Path
        The path where the visualization will be saved.
    """
    umap_model = umap.UMAP(metric="euclidean")
    embedding_umap = umap_model.fit_transform(embeddings)
    fig = create_plot(embedding_umap, labels, title_extension, emb_type)
    fig.write_html(str(file_path.with_suffix(".html")))


def create_umap_combined(
    emb_proc_desc: pd.DataFrame,
    emb_legal_text: pd.DataFrame,
    labels: pd.Series,
    emb_type: str,
    file_path: Path,
) -> None:
    """
    Create a UMAP visualization for combined embeddings.

    Parameters
    ----------
    emb_proc_desc : pandas DataFrame
        The embeddings for the process descriptions.
    emb_legal_text : pandas DataFrame
        The embeddings for the legal texts.
    labels : pandas Series
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

    fig = create_plot(df_umap, labels, emb_type, "Multiple UMAPs", types)
    fig.write_html(str(file_path.with_suffix(".html")))


def create_plot(
    data: pd.DataFrame,
    labels: pd.Series,
    emb_type: str,
    title_extension: str,
    types=None,
) -> px.scatter:
    """
    Create a plotly express scatter plot for the UMAP visualization.

    Parameters
    ----------
    data : pandas DataFrame
        The data to plot.
    labels : array-like
        The labels for the data.
    emb_type : str
        The type of the embeddings.
    title_extension : str
        The extension for the title of the visualization.
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
            title=f"UMAP Visualization of {emb_type} Embeddings (Process Description vs Legal Text) {title_extension}",
        )
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig
