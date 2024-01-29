import logging
import os
import pickle
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

#  Custom imports (need the full path for api.py)
from src.features.utils_embedding_functions import (
    get_sentence_vector_custom,
    get_embeddings_bert,
    get_embeddings_gpt,
    load_model_if_exists,
    load_training_data,
    save_pickle,
)

load_dotenv(find_dotenv())

openai.api_key = os.getenv("OPENAI_API_KEY")


class EmbeddingProcessor:
    """
    This class is responsible for processing embeddings for text data.
    It supports multiple types of embeddings including TF-IDF, Word2Vec, GloVe, FastText, BERT, and GPT.
    """

    def __init__(self):
        """
        Initializes the EmbeddingProcessor object.
        """
        project_dir = Path(__file__).resolve().parents[2]

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()
        self.openai = OpenAI()

        self.tfidf = load_model_if_exists(
            project_dir / "models/embeddings/word2vec.pkl"
        )

        self.word2vec = load_model_if_exists(
            project_dir / "models/embeddings/word2vec.pkl"
        )

        self.fasttext = load_model_if_exists(
            project_dir / "models/embeddings/fasttext.pkl"
        )
        self.glove = load_model_if_exists(project_dir / "models/embeddings/glove.pkl")

    def compute_tfidf_embedding(self, text: str) -> np.ndarray:
        """
        Computes the TF-IDF embedding for the given text.

        Parameters:
        text (str): The text to compute the embedding for.

        Returns:
        np.ndarray: The computed TF-IDF embedding.
        """
        tfidf_text = self.tfidf.transform(text).toarray()
        return tfidf_text

    def compute_word2vec_embedding(self, text: str) -> np.ndarray:
        """
        Computes the Word2Vec embedding for the given text.

        Parameters:
        text (str): The text to compute the embedding for.

        Returns:
        np.ndarray: The computed Word2Vec embedding.
        """

        def get_embeddings(statements: str) -> np.ndarray:
            tokenized_statements = [statement.split() for statement in statements]
            embeddings = pd.Series(tokenized_statements).apply(
                lambda x: get_sentence_vector_custom(x, self.word2vec)
            )
            return np.array(embeddings.tolist())

        w2v_text = get_embeddings(text)

        return w2v_text

    def compute_bert_embedding(self, text: str) -> np.ndarray:
        """
        Computes the BERT embedding for the given text.

        Parameters:
        text (str): The text to compute the embedding for.

        Returns:
        np.ndarray: The computed BERT embedding.
        """

        def get_embeddings(statements: pd.Series):
            embeddings = statements.apply(
                lambda x: get_embeddings_bert(x, self.bert_tokenizer, self.bert_model)
            )
            return np.array(embeddings.tolist())

        bert_text = get_embeddings(pd.Series(text))

        return bert_text

    def compute_fasttext_embedding(self, text: str) -> np.ndarray:
        """
        Computes the FastText embedding for the given text.

        Parameters:
        text (str): The text to compute the embedding for.

        Returns:
        np.ndarray: The computed FastText embedding.
        """

        def get_embeddings(statements):
            tokenized_statements = [statement.split() for statement in statements]
            embeddings = pd.Series(tokenized_statements).apply(
                lambda x: self.fasttext.wv.get_sentence_vector(x)
            )
            return np.array(embeddings.tolist())

        ft_text = get_embeddings(text)

        return ft_text

    def compute_glove_embedding(self, text: str) -> np.ndarray:
        """
        Computes the GloVe embedding for the given text.

        Parameters:
        text (str): The text to compute the embedding for.

        Returns:
        np.ndarray: The computed GloVe embedding.
        """

        def get_embeddings(statements):
            tokenized_statements = [statement.split() for statement in statements]
            embeddings = pd.Series(tokenized_statements).apply(
                lambda x: get_sentence_vector_custom(x, self.glove, is_glove=True)
            )
            return np.array(embeddings.tolist())

        glove_text = get_embeddings(text)

        return glove_text

    def compute_gpt_embedding(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Computes the GPT embedding for the given text.

        Parameters:
        text (str): The text to compute the embedding for.

        Returns:
        np.ndarray: The computed GPT embedding.
        """

        def get_embeddings(statements: pd.Series):
            embeddings = statements.apply(lambda x: get_embeddings_gpt(x, self.openai))
            return np.array(embeddings.tolist())

        gpt_text = get_embeddings(pd.Series(text))

        return gpt_text

    def train_model(self, embedding_type: str) -> None:
        """
        Trains the specified embedding model on the training data.

        Parameters:
        embedding_type (str): The type of the embedding model to train.
        """
        # Load the data needed for training the model
        tokenized_sentences, raw_sentences = load_training_data()

        # Train the model and save it to the instance for later use
        if embedding_type == "word2vec":
            self.word2vec = self.train_word2vec(tokenized_sentences)
        elif embedding_type == "glove":
            self.glove = self.train_glove()
        elif embedding_type == "fasttext":
            self.fasttext = self.train_fasttext(tokenized_sentences)
        elif embedding_type == "tfidf":
            self.tfidf = self.train_tfidf(raw_sentences)
        else:
            raise ValueError(f"Unknown model type for training: {embedding_type}")

    def train_word2vec(self, sentences: list) -> Word2Vec:
        """
        Trains a Word2Vec model on the given sentences.

        Parameters:
        sentences (list): The sentences to train the model on.

        Returns:
        Word2Vec: The trained Word2Vec model.
        """
        project_dir = Path(__file__).resolve().parents[2]
        self.word2vec = Word2Vec(
            sentences, vector_size=300, window=5, min_count=1, workers=4
        )
        logging.info("Word2Vec model loaded.")
        save_pickle(self.word2vec, project_dir / "models/embeddings/word2vec.pkl")
        return self.word2vec

    def train_glove(self) -> KeyedVectors:
        """
        Loads the GloVe model from the given path.

        Returns:
        KeyedVectors: The loaded GloVe model.
        """
        project_dir = Path(__file__).resolve().parents[2]
        glove_input_file = project_dir / "data/external/glove.6B.300d.txt"
        word2vec_output_file = project_dir / "data/external/glove.6B.300d.word2vec.txt"
        glove2word2vec(str(glove_input_file), str(word2vec_output_file))
        self.glove = KeyedVectors.load_word2vec_format(
            str(word2vec_output_file), binary=False
        )
        logging.info("GloVe model loaded.")
        save_pickle(self.glove, project_dir / "models/embeddings/glove.pkl")

        return self.glove

    def train_fasttext(self, sentences: list) -> FastText:
        """
        Trains a FastText model on the given sentences.

        Parameters:
        sentences (list): The sentences to train the model on.

        Returns:
        FastText: The trained FastText model.
        """
        self.fasttext = FastText(
            sentences, vector_size=300, window=5, min_count=1, workers=4
        )
        project_dir = Path(__file__).resolve().parents[2]
        logging.info("FastText model loaded.")
        save_pickle(self.fasttext, project_dir / "models/embeddings/fasttext.pkl")
        return self.fasttext

    def train_tfidf(self, sentences: list) -> TfidfVectorizer:
        """
        Trains a TF-IDF model on the given sentences.

        Parameters:
        sentences (list): The sentences to train the model on.

        Returns:
        TfidfVectorizer: The trained TF-IDF model.
        """
        self.tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=1300
        )
        self.tfidf.fit_transform(sentences).toarray()
        project_dir = Path(__file__).resolve().parents[2]
        logging.info("TF-IDF model loaded.")
        save_pickle(self.tfidf, project_dir / "models/embeddings/tfidf.pkl")

        return self.tfidf

    def embed_new_text(
        self, proc_desc: str, legal_text: str, embedding_type: str, dataset_type: str
    ) -> np.ndarray:
        """
        Embeds new text using the specified embedding model and dataset type.

        Parameters:
        proc_desc (str): The process description.
        legal_text (str): The legal text.
        embedding_type (str): The type of the embedding model to use.
        dataset_type (str): The type of the dataset.

        Returns:
        np.ndarray: The computed embedding.
        """
        if dataset_type == "separate":
            proc_desc_vector = self.get_embedding(proc_desc, embedding_type)
            legal_text_vector = self.get_embedding(legal_text, embedding_type)
            combined_vector = np.concatenate(
                (proc_desc_vector, legal_text_vector), axis=1
            )
        elif dataset_type == "combined":
            combined_text = proc_desc + " " + legal_text
            combined_vector = self.get_embedding(combined_text, embedding_type)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return combined_vector.reshape(1, -1)

    def get_embedding(self, text: str, embedding_type: str) -> np.ndarray:
        """
        Gets the embedding for the given text using the specified embedding model.

        Parameters:
        text (str): The text to get the embedding for.
        embedding_type (str): The type of the embedding model to use.

        Returns:
        np.ndarray: The computed embedding.
        """
        if embedding_type in ["glove", "fasttext", "word2vec"]:
            embedding_model = getattr(self, f"{embedding_type}", None)

            # If the model is None, train or load the model first
            if embedding_model is None:
                self.train_model(embedding_type)
                embedding_model = getattr(self, f"{embedding_type}")

            # Check again after training/loading. If still None, raise an error
            if embedding_model is None:
                raise ValueError(
                    f"The model for {embedding_type} could not be loaded or trained."
                )

            glove_true = embedding_type == "glove"
            embedding_model = getattr(self, f"{embedding_type}")
            embedding_vector = get_sentence_vector_custom(
                text.split(), embedding_model, glove_true
            )

        elif embedding_type == "tfidf":
            self.train_model(embedding_type)
            embedding_vector = self.tfidf.transform([text]).toarray()

        elif embedding_type in ["bert", "gpt"]:
            embedding_method = getattr(self, f"compute_{embedding_type}_embedding")
            embedding_vector = embedding_method(text)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        return embedding_vector
