import os
import pickle
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

from utils_embedding_functions import (
    get_sentence_vector_custom,
    get_embeddings_bert,
    get_embeddings_gpt,
)

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")


class EmbeddingProcessor:
    def __init__(self, max_features=1000):
        self.tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=max_features
        )
        self.word2vec = None
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()
        self.openai = OpenAI()
        self.fasttext = None
        self.glove2word2vec = None

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
            lambda x: get_sentence_vector_custom(x, self.word2vec)
        )
        w2v_train = np.array(w2v_train.tolist())
        w2v_train_tensor = torch.tensor(w2v_train, dtype=torch.float)

        w2v_test = pd.Series(tokenized_test).apply(
            lambda x: get_sentence_vector_custom(x, self.word2vec)
        )
        w2v_test = np.array(w2v_test.tolist())
        w2v_test_tensor = torch.tensor(w2v_test, dtype=torch.float)
        return w2v_train, w2v_test, w2v_train_tensor, w2v_test_tensor

    def compute_bert_embedding(self, train_statements, test_statements):
        bert_train = train_statements.apply(
            lambda x: get_embeddings_bert(x, self.bert_tokenizer, self.bert_model)
        )
        bert_test = test_statements.apply(
            lambda x: get_embeddings_bert(x, self.bert_tokenizer, self.bert_model)
        )

        bert_train = np.array(bert_train.tolist())
        bert_test = np.array(bert_test.tolist())

        bert_train_tensor = torch.tensor(bert_train, dtype=torch.float)
        bert_test_tensor = torch.tensor(bert_test, dtype=torch.float)

        return bert_train, bert_test, bert_train_tensor, bert_test_tensor

    def compute_fasttext_embedding(self, train_statements, test_statements):
        tokenized_train_statements = [
            statement.split() for statement in train_statements
        ]
        tokenized_test = [statement.split() for statement in test_statements]

        self.fasttext = FastText(
            tokenized_train_statements,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
        )
        ft_train = pd.Series(tokenized_train_statements).apply(
            lambda x: self.fasttext.wv.get_sentence_vector(x)
        )
        ft_test = pd.Series(tokenized_test).apply(
            lambda x: self.fasttext.wv.get_sentence_vector(x)
        )
        ft_train = np.array(ft_train.tolist())
        ft_test = np.array(ft_test.tolist())

        ft_train_tensor = torch.tensor(ft_train, dtype=torch.float)
        ft_test_tensor = torch.tensor(ft_test, dtype=torch.float)

        return ft_train, ft_test, ft_train_tensor, ft_test_tensor

    def compute_glove_embedding(self, train_statements, test_statements):
        tokenized_train_statements = [
            statement.split() for statement in train_statements
        ]
        tokenized_test = [statement.split() for statement in test_statements]

        project_dir = Path(__file__).resolve().parents[2]
        glove_input_file = project_dir / "data/external/glove.6B.300d.txt"
        word2vec_output_file = project_dir / "data/external/glove.6B.300d.word2vec.txt"
        glove2word2vec(str(glove_input_file), str(word2vec_output_file))
        self.glove2word2vec = KeyedVectors.load_word2vec_format(
            str(word2vec_output_file), binary=False
        )

        glove_train = pd.Series(tokenized_train_statements).apply(
            lambda x: get_sentence_vector_custom(x, self.glove2word2vec, is_glove=True)
        )
        glove_test = pd.Series(tokenized_test).apply(
            lambda x: get_sentence_vector_custom(x, self.glove2word2vec, is_glove=True)
        )

        glove_train = np.array(glove_train.tolist())
        glove_test = np.array(glove_test.tolist())

        glove_train_tensor = torch.tensor(glove_train, dtype=torch.float)
        glove_test_tensor = torch.tensor(glove_test, dtype=torch.float)

        return glove_train, glove_test, glove_train_tensor, glove_test_tensor

    def compute_gpt_embedding(self, train_statements, test_statements):
        gpt_train = train_statements.apply(lambda x: get_embeddings_gpt(x, self.openai))
        gpt_test = test_statements.apply(lambda x: get_embeddings_gpt(x, self.openai))

        gpt_train = np.array(gpt_train.tolist())
        gpt_test = np.array(gpt_test.tolist())

        gpt_train_tensor = torch.tensor(gpt_train, dtype=torch.float)
        gpt_test_tensor = torch.tensor(gpt_test, dtype=torch.float)

        return gpt_train, gpt_test, gpt_train_tensor, gpt_test_tensor

    def save_embeddings(self, obj, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)


def main():
    project_dir = Path(__file__).resolve().parents[2]
    embedding_processor = EmbeddingProcessor()
    #
    # # load data and combine text and process description
    # df_train = pd.read_csv(
    #     project_dir / "data/processed/training_data_preprocessed.csv"
    # )
    # df_test = pd.read_csv(
    #     project_dir / "data/evaluation/gold_standard_preprocessed.csv"
    # )
    # df_train["Combined_Text"] = df_train["Process_description"] + " " + df_train["Text"]
    # df_test["Combined_Text"] = df_test["Process_description"] + " " + df_test["Text"]
    #
    # # TF-IDF
    # print("Computing TF-IDF embeddings...")
    # (
    #     tfidf_train,
    #     tfidf_test,
    #     tfidf_train_tensor,
    #     tfidf_test_tensor,
    # ) = embedding_processor.compute_tfidf_embedding(
    #     df_train["Combined_Text"], df_test["Combined_Text"]
    # )
    # embedding_processor.save_embeddings(
    #     tfidf_train, project_dir / "data/processed/embeddings/tfidf_train.pkl"
    # )
    # embedding_processor.save_embeddings(
    #     tfidf_test, project_dir / "data/processed/embeddings/tfidf_test.pkl"
    # )
    #
    # torch.save(
    #     tfidf_train_tensor,
    #     project_dir / "data/processed/embeddings/tfidf_train_tensor.pt",
    # )
    # torch.save(
    #     tfidf_test_tensor,
    #     project_dir / "data/processed/embeddings/tfidf_test_tensor.pt",
    # )
    #
    # # Word2Vec
    # print("Computing Word2Vec embeddings...")
    # (
    #     w2v_train,
    #     w2v_test,
    #     w2v_train_tensor,
    #     w2v_test_tensor,
    # ) = embedding_processor.compute_word2vec_embedding(
    #     df_train["Combined_Text"], df_test["Combined_Text"]
    # )
    # embedding_processor.save_embeddings(
    #     w2v_train, project_dir / "data/processed/embeddings/w2v_train.pkl"
    # )
    # embedding_processor.save_embeddings(
    #     w2v_test, project_dir / "data/processed/embeddings/w2v_test.pkl"
    # )
    # torch.save(
    #     w2v_train_tensor,
    #     project_dir / "data/processed/embeddings/w2v_train_tensor.pt",
    # )
    # torch.save(
    #     w2v_test_tensor,
    #     project_dir / "data/processed/embeddings/w2v_test_tensor.pt",
    # )
    #
    # # # BERT
    # print("Computing BERT embeddings...")
    # (
    #     bert_train,
    #     bert_test,
    #     bert_train_tensor,
    #     bert_test_tensor,
    # ) = embedding_processor.compute_bert_embedding(
    #     df_train["Combined_Text"], df_test["Combined_Text"]
    # )
    # embedding_processor.save_embeddings(
    #     bert_train, project_dir / "data/processed/embeddings/bert_train.pkl"
    # )
    # embedding_processor.save_embeddings(
    #     bert_test, project_dir / "data/processed/embeddings/bert_test.pkl"
    # )
    # torch.save(
    #     bert_train_tensor,
    #     project_dir / "data/processed/embeddings/bert_train_tensor.pt",
    # )
    # torch.save(
    #     bert_test_tensor,
    #     project_dir / "data/processed/embeddings/bert_test_tensor.pt",
    # )
    #
    # # GPT ADA
    # print("Computing GPT embeddings...")
    # (
    #     gpt_train,
    #     gpt_test,
    #     gpt_train_tensor,
    #     gpt_test_tensor,
    # ) = embedding_processor.compute_gpt_embedding(
    #     df_train["Combined_Text"], df_test["Combined_Text"]
    # )
    # embedding_processor.save_embeddings(
    #     gpt_train, project_dir / "data/processed/embeddings/gpt_train.pkl"
    # )
    # embedding_processor.save_embeddings(
    #     gpt_test, project_dir / "data/processed/embeddings/gpt_test.pkl"
    # )
    # torch.save(
    #     gpt_train_tensor,
    #     project_dir / "data/processed/embeddings/gpt_train_tensor.pt",
    # )
    # torch.save(
    #     gpt_test_tensor,
    #     project_dir / "data/processed/embeddings/gpt_test_tensor.pt",
    # )
    #
    # # FastText
    # print("Computing FastText embeddings...")
    # (
    #     ft_train,
    #     ft_test,
    #     ft_train_tensor,
    #     ft_test_tensor,
    # ) = embedding_processor.compute_fasttext_embedding(
    #     df_train["Combined_Text"], df_test["Combined_Text"]
    # )
    # embedding_processor.save_embeddings(
    #     ft_train, project_dir / "data/processed/embeddings/ft_train.pkl"
    # )
    # embedding_processor.save_embeddings(
    #     ft_test, project_dir / "data/processed/embeddings/ft_test.pkl"
    # )
    # torch.save(
    #     ft_train_tensor,
    #     project_dir / "data/processed/embeddings/ft_train_tensor.pt",
    # )
    # torch.save(
    #     ft_test_tensor,
    #     project_dir / "data/processed/embeddings/ft_test_tensor.pt",
    # )
    #
    # # Glove
    # print("Computing GloVe embeddings...")
    # (
    #     glove_train,
    #     glove_test,
    #     glove_train_tensor,
    #     glove_test_tensor,
    # ) = embedding_processor.compute_glove_embedding(
    #     df_train["Combined_Text"], df_test["Combined_Text"]
    # )
    # embedding_processor.save_embeddings(
    #     glove_train, project_dir / "data/processed/embeddings/glove_train.pkl"
    # )
    # embedding_processor.save_embeddings(
    #     glove_test, project_dir / "data/processed/embeddings/glove_test.pkl"
    # )
    # torch.save(
    #     glove_train_tensor,
    #     project_dir / "data/processed/embeddings/glove_train_tensor.pt",
    # )
    # torch.save(
    #     glove_test_tensor,
    #     project_dir / "data/processed/embeddings/glove_test_tensor.pt",
    # )


if __name__ == "__main__":
    main()
