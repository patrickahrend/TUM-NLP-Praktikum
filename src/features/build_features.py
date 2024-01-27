import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(embedding_file):
    with open(embedding_file, "rb") as f:
        return pickle.load(f)


def load_data():
    project_dir = Path(__file__).resolve().parents[2]
    df_train = pd.read_csv(
        project_dir / "data/processed/training_data_preprocessed.csv"
    )
    df_test = pd.read_csv(
        project_dir / "data/evaluation/gold_standard_preprocessed.csv"
    )
    return df_train, df_test


def compute_word_frequencies(texts, n_most_common=100):
    vectorized = CountVectorizer(max_features=n_most_common, stop_words="english")
    word_freq = vectorized.fit_transform(texts)
    return word_freq.toarray()


def compute_cosine_similarity(embeddings1, embeddings2):
    similarities = [
        cosine_similarity([emb1], [emb2])[0][0]
        for emb1, emb2 in zip(embeddings1, embeddings2)
    ]
    return np.array(similarities).reshape(-1, 1)


def save_features(features, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(features, f)


def main():
    project_dir = Path(__file__).resolve().parents[2]

    embeddings_types = ["tfidf", "w2v", "bert", "gpt", "glove", "ft"]

    for embedding_type in embeddings_types:
        proc_desc_train_embedding = load_embeddings(
            project_dir
            / f"data/processed/embeddings/{embedding_type}_train_proc_desc.pkl"
        )
        legal_text_train_embedding = load_embeddings(
            project_dir
            / f"data/processed/embeddings/{embedding_type}_train_legal_text.pkl"
        )

        proc_desc_test_embedding = load_embeddings(
            project_dir
            / f"data/processed/embeddings/{embedding_type}_test_proc_desc.pkl"
        )
        legal_text_test_embedding = load_embeddings(
            project_dir
            / f"data/processed/embeddings/{embedding_type}_test_legal_text.pkl"
        )

        cos_sim_train = compute_cosine_similarity(
            proc_desc_train_embedding, legal_text_train_embedding
        )
        cos_sim_test = compute_cosine_similarity(
            proc_desc_test_embedding, legal_text_test_embedding
        )

        save_features(
            cos_sim_train,
            project_dir / f"data/processed/features/cos_sim_{embedding_type}_train.pkl",
        )
        save_features(
            cos_sim_test,
            project_dir / f"data/processed/features/cos_sim_{embedding_type}_test.pkl",
        )

        df_train, df_test = load_data()

        train_proc_desc_freqs = compute_word_frequencies(
            df_train["Process_description"]
        )
        test_proc_desc_freqs = compute_word_frequencies(df_test["Process_description"])
        train_legal_text_freqs = compute_word_frequencies(df_train["Text"])
        test_legal_text_freqs = compute_word_frequencies(df_test["Text"])

        save_features(
            train_legal_text_freqs,
            project_dir / "data/processed/features/train_legal_text_freqs.pkl",
        )
        save_features(
            test_legal_text_freqs,
            project_dir / "data/processed/features/test_legal_text_freqs.pkl",
        )
        save_features(
            train_proc_desc_freqs,
            project_dir / "data/processed/features/train_proc_desc_freqs.pkl",
        )
        save_features(
            test_proc_desc_freqs,
            project_dir / "data/processed/features/test_proc_desc_freqs.pkl",
        )


if __name__ == "__main__":
    main()