import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingProcessor:
    def __init__(self, max_features=1000):
        self.tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=max_features
        )

    def compute_tfidf_embedding(self, train_statements, test_statements):
        tfidf_train = self.tfidf.fit_transform(train_statements).toarray()
        tfidf_test = self.tfidf.transform(test_statements).toarray()
        tfidf_train_tensor = torch.tensor(tfidf_train, dtype=torch.float)
        tfidf_test_tensor = torch.tensor(tfidf_test, dtype=torch.float)

        return tfidf_train, tfidf_test, tfidf_train_tensor, tfidf_test_tensor

    def save_embeddings(self, obj, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)


def main():
    project_dir = Path(__file__).resolve().parents[2]
    embedding_processor = EmbeddingProcessor()

    # load data and combine text and process description
    df_train = pd.read_csv(
        project_dir / "data/processed/final_labels_with_description_preprocessed.csv"
    )
    df_test = pd.read_csv(
        project_dir / "data/evaluation/gold_standard_preprocessed.csv"
    )
    df_train["Combined_Text"] = df_train["Process_description"] + " " + df_train["Text"]
    df_test["Combined_Text"] = df_test["Process_description"] + " " + df_test["Text"]

    # TF-IDF
    (
        tfidf_train,
        tfidf_test,
        tfidf_train_tensor,
        tfidf_test_tensor,
    ) = embedding_processor.compute_tfidf_embedding(
        df_train["Combined_Text"], df_test["Combined_Text"]
    )
    embedding_processor.save_embeddings(
        tfidf_train, project_dir / "data/processed/embeddings/tfidf_train.pkl"
    )
    embedding_processor.save_embeddings(
        tfidf_test, project_dir / "data/processed/embeddings/tfidf_test.pkl"
    )

    torch.save(
        tfidf_train_tensor,
        project_dir / "data/processed/embeddings/tfidf_train_tensor.pt",
    )
    torch.save(
        tfidf_test_tensor,
        project_dir / "data/processed/embeddings/tfidf_test_tensor.pt",
    )


if __name__ == "__main__":
    main()
