import os
from pathlib import Path

import pandas as pd

# Custom imports
from src.visualization.utils import load_pickle
from src.visualization.visualize_umap import create_umap_single, create_umap_combined


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
