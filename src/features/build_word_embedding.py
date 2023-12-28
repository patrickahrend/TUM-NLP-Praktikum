import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from utils_embedding_functions import get_sentence_vector


class EmbeddingProcessor:
    def __init__(self, max_features=1000):
        self.tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=max_features
        )
        self.word2vec = None

    def compute_tfidf_embedding(self, train_statements, test_statements):
        tfidf_train = self.tfidf.fit_transform(train_statements).toarray()
        tfidf_test = self.tfidf.transform(test_statements).toarray()
        tfidf_train_tensor = torch.tensor(tfidf_train, dtype=torch.float)
        tfidf_test_tensor = torch.tensor(tfidf_test, dtype=torch.float)

        return tfidf_train, tfidf_test, tfidf_train_tensor, tfidf_test_tensor

    def compute_word2vec_embedding(self, train_statements, test_statements):
        tokenized_train_statements = [
            statement.split() for statement in train_statements
        ]
        tokenized_test = [statement.split() for statement in test_statements]

        self.word2vec = Word2Vec(
            tokenized_train_statements,
            min_count=1,
            workers=2,
            vector_size=1000,
            window=10,
        )
        w2v_train = pd.Series(tokenized_train_statements).apply(
            lambda x: get_sentence_vector(x, self.word2vec)
        )
        w2v_train = np.array(w2v_train.tolist())
        w2v_train_tensor = torch.tensor(w2v_train, dtype=torch.float)

        w2v_test = pd.Series(tokenized_test).apply(
            lambda x: get_sentence_vector(x, self.word2vec)
        )
        w2v_test = np.array(w2v_test.tolist())
        w2v_test_tensor = torch.tensor(w2v_test, dtype=torch.float)
        return w2v_train, w2v_test, w2v_train_tensor, w2v_test_tensor

    def compute_bert_embedding(self, train_statements, test_statements):
        pass

    def compute_fasttext_embedding(self, train_statements, test_statements):
        pass

    def compute_glove_embedding(self, train_statements, test_statements):
        pass

    def compute_gpt_embedding(self, train_statements, test_statements):
        pass

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

    # Word2Vec
    (
        w2v_train,
        w2v_test,
        w2v_train_tensor,
        w2v_test_tensor,
    ) = embedding_processor.compute_word2vec_embedding(
        df_train["Combined_Text"], df_test["Combined_Text"]
    )
    embedding_processor.save_embeddings(
        w2v_train, project_dir / "data/processed/embeddings/w2v_train.pkl"
    )
    embedding_processor.save_embeddings(
        w2v_test, project_dir / "data/processed/embeddings/w2v_test.pkl"
    )
    torch.save(
        w2v_train_tensor,
        project_dir / "data/processed/embeddings/w2v_train_tensor.pt",
    )
    torch.save(
        w2v_test_tensor,
        project_dir / "data/processed/embeddings/w2v_test_tensor.pt",
    )

    # BERT


if __name__ == "__main__":
    main()
