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

from .utils_embedding_functions import (
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
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.bert_model.eval()
        self.openai = OpenAI()
        self.word2vec = None
        self.fasttext = None
        self.glove2word2vec = None

    def compute_tfidf_embedding(self, proc_desc, legal_text):
        tfidf_train_legal_text = self.tfidf.fit_transform(legal_text).toarray()
        tfidf_train_proc_desc = self.tfidf.transform(proc_desc).toarray()

        return (
            tfidf_train_proc_desc,
            tfidf_train_legal_text,
        )

    def compute_word2vec_embedding(
        self, train_proc_desc, train_legal_text, test_proc_desc, test_legal_text
    ):
        combined_train_statements = train_proc_desc + " " + train_legal_text
        tokenized_combined_train = [
            statement.split() for statement in combined_train_statements
        ]

        self.word2vec = Word2Vec(
            tokenized_combined_train,
            min_count=1,
            workers=2,
            vector_size=1000,
            window=10,
        )

        def get_embeddings(statements):
            tokenized_statements = [statement.split() for statement in statements]
            embeddings = pd.Series(tokenized_statements).apply(
                lambda x: get_sentence_vector_custom(x, self.word2vec)
            )
            return np.array(embeddings.tolist())

        w2v_train_proc_desc = get_embeddings(train_proc_desc)
        w2v_train_legal_text = get_embeddings(train_legal_text)
        w2v_test_proc_desc = get_embeddings(test_proc_desc)
        w2v_test_legal_text = get_embeddings(test_legal_text)

        w2v_train_proc_desc_tensor = torch.tensor(
            w2v_train_proc_desc, dtype=torch.float
        )
        w2v_train_legal_text_tensor = torch.tensor(
            w2v_train_legal_text, dtype=torch.float
        )
        w2v_test_proc_desc_tensor = torch.tensor(w2v_test_proc_desc, dtype=torch.float)
        w2v_test_legal_text_tensor = torch.tensor(
            w2v_test_legal_text, dtype=torch.float
        )

        return (
            w2v_train_proc_desc,
            w2v_train_legal_text,
            w2v_test_proc_desc,
            w2v_test_legal_text,
            w2v_train_proc_desc_tensor,
            w2v_train_legal_text_tensor,
            w2v_test_proc_desc_tensor,
            w2v_test_legal_text_tensor,
        )

    def compute_bert_embedding(self, proc_desc, legal_text):
        def get_embeddings(statements):
            embeddings = statements.apply(
                lambda x: get_embeddings_bert(x, self.bert_tokenizer, self.bert_model)
            )
            return np.array(embeddings.tolist())

        bert_proc_desc = get_embeddings(pd.Series(proc_desc))
        bert_legal_text = get_embeddings(pd.Series(legal_text))

        return (
            bert_proc_desc,
            bert_legal_text,
        )

    def compute_fasttext_embedding(
        self, train_proc_desc, train_legal_text, test_proc_desc, test_legal_text
    ):
        combined_train_statements = train_proc_desc + train_legal_text
        tokenized_combined_train = [
            statement.split() for statement in combined_train_statements
        ]

        self.fasttext = FastText(
            tokenized_combined_train,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
        )

        def get_embeddings(statements):
            tokenized_statements = [statement.split() for statement in statements]
            embeddings = pd.Series(tokenized_statements).apply(
                lambda x: self.fasttext.wv.get_sentence_vector(x)
            )
            return np.array(embeddings.tolist())

        ft_train_proc_desc = get_embeddings(train_proc_desc)
        ft_train_legal_text = get_embeddings(train_legal_text)
        ft_test_proc_desc = get_embeddings(test_proc_desc)
        ft_test_legal_text = get_embeddings(test_legal_text)

        ft_train_proc_desc_tensor = torch.tensor(ft_train_proc_desc, dtype=torch.float)
        ft_train_legal_text_tensor = torch.tensor(
            ft_train_legal_text, dtype=torch.float
        )
        ft_test_proc_desc_tensor = torch.tensor(ft_test_proc_desc, dtype=torch.float)
        ft_test_legal_text_tensor = torch.tensor(ft_test_legal_text, dtype=torch.float)

        return (
            ft_train_proc_desc,
            ft_train_legal_text,
            ft_test_proc_desc,
            ft_test_legal_text,
            ft_train_proc_desc_tensor,
            ft_train_legal_text_tensor,
            ft_test_proc_desc_tensor,
            ft_test_legal_text_tensor,
        )

    def compute_glove_embedding(
        self, train_proc_desc, train_legal_text, test_proc_desc, test_legal_text
    ):
        project_dir = Path(__file__).resolve().parents[2]
        glove_input_file = project_dir / "data/external/glove.6B.300d.txt"
        word2vec_output_file = project_dir / "data/external/glove.6B.300d.word2vec.txt"
        glove2word2vec(str(glove_input_file), str(word2vec_output_file))
        self.glove2word2vec = KeyedVectors.load_word2vec_format(
            str(word2vec_output_file), binary=False
        )

        def get_embeddings(statements):
            tokenized_statements = [statement.split() for statement in statements]
            embeddings = pd.Series(tokenized_statements).apply(
                lambda x: get_sentence_vector_custom(
                    x, self.glove2word2vec, is_glove=True
                )
            )
            return np.array(embeddings.tolist())

        glove_train_proc_desc = get_embeddings(train_proc_desc)
        glove_train_legal_text = get_embeddings(train_legal_text)
        glove_test_proc_desc = get_embeddings(test_proc_desc)
        glove_test_legal_text = get_embeddings(test_legal_text)

        glove_train_proc_desc_tensor = torch.tensor(
            glove_train_proc_desc, dtype=torch.float
        )
        glove_train_legal_text_tensor = torch.tensor(
            glove_train_legal_text, dtype=torch.float
        )
        glove_test_proc_desc_tensor = torch.tensor(
            glove_test_proc_desc, dtype=torch.float
        )
        glove_test_legal_text_tensor = torch.tensor(
            glove_test_legal_text, dtype=torch.float
        )

        return (
            glove_train_proc_desc,
            glove_train_legal_text,
            glove_test_proc_desc,
            glove_test_legal_text,
            glove_train_proc_desc_tensor,
            glove_train_legal_text_tensor,
            glove_test_proc_desc_tensor,
            glove_test_legal_text_tensor,
        )

    def compute_gpt_embedding(
        self,
        pros_desc,
        legal_text,
    ):
        def get_embeddings(statements):
            embeddings = statements.apply(lambda x: get_embeddings_gpt(x, self.openai))
            return np.array(embeddings.tolist())

        gpt_proc_desc = get_embeddings(pd.Series(pros_desc))
        gpt_legal_text = get_embeddings(pd.Series(legal_text))

        return (
            gpt_proc_desc,
            gpt_legal_text,
        )

    def save_embeddings(self, obj, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

    def embed_new_text(self, proc_desc, legal_text, embedding_type):
        ## edge case weil model auf allem trainiert wird .. einzelen methoden werden dafür noch geschrieben
        if (
            embedding_type == "glove"
            or embedding_type == "fasttext"
            or embedding_type == "word2vec"
        ):
            # Assuming you have a method to convert text to GloVe vectors
            proc_desc_vector = self.text_to_glove_vector(proc_desc)
            legal_text_vector = self.text_to_glove_vector(legal_text)
            combined_vector = np.concatenate((proc_desc_vector, legal_text_vector))
        # ... handle other embedding types ...
        else:
            compute_method = getattr(self, f"compute_{embedding_type}_embedding")
            # Pass the new text as the train data and disregard the test data
            embedding_train_proc_desc, embedding_train_legal_text = compute_method(
                proc_desc, legal_text
            )
            return (
                embedding_train_proc_desc,
                embedding_train_legal_text,
            )  # Only return the first element (embedding vectors)


#
#
# def main():
#     project_dir = Path(__file__).resolve().parents[2]
#     embedding_processor = EmbeddingProcessor()
#
#     # load data and combine text and process description
#     df_train = pd.read_csv(
#         project_dir / "data/processed/training_data_preprocessed.csv"
#     )
#     df_test = pd.read_csv(
#         project_dir / "data/evaluation/gold_standard_preprocessed.csv"
#     )
#
#     # df_train["Combined_Text"] = df_train["Process_description"] + " " + df_train["Text"]
#     # df_test["Combined_Text"] = df_test["Process_description"] + " " + df_test["Text"]
#
#     # TF-IDF
#     print("Computing TF-IDF embeddings with separate process description and text ...")
#     (
#         tfidf_train_proc_desc,
#         tfidf_train_legal_text,
#         tfidf_test_proc_desc,
#         tfidf_test_legal_text,
#         tfidf_train_proc_desc_tensor,
#         tfidf_train_legal_text_tensor,
#         tfidf_test_proc_desc_tensor,
#         tfidf_test_legal_text_tensor,
#     ) = embedding_processor.compute_tfidf_embedding(
#         df_train["Process_description"],
#         df_train["Text"],
#         df_test["Process_description"],
#         df_test["Text"],
#     )
#     embedding_processor.save_embeddings(
#         tfidf_train_proc_desc,
#         project_dir / "data/processed/embeddings/tfidf_train_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         tfidf_train_legal_text,
#         project_dir / "data/processed/embeddings/tfidf_train_legal_text.pkl",
#     )
#     embedding_processor.save_embeddings(
#         tfidf_test_proc_desc,
#         project_dir / "data/processed/embeddings/tfidf_test_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         tfidf_test_legal_text,
#         project_dir / "data/processed/embeddings/tfidf_test_legal_text.pkl",
#     )
#
#     torch.save(
#         tfidf_train_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/tfidf_train_proc_desc_tensor.pt",
#     )
#     torch.save(
#         tfidf_test_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/tfidf_test_proc_desc_tensor.pt",
#     )
#     torch.save(
#         tfidf_test_legal_text_tensor,
#         project_dir / "data/processed/embeddings/tfidf_test_legal_text_tensor.pt",
#     )
#
#     # Word2Vec
#     print(
#         "Computing Word2Vec embeddings with separate process description and text ..."
#     )
#     (
#         w2v_train_proc_desc,
#         w2v_train_legal_text,
#         w2v_test_proc_desc,
#         w2v_test_legal_text,
#         w2v_train_proc_desc_tensor,
#         w2v_train_legal_text_tensor,
#         w2v_test_proc_desc_tensor,
#         w2v_test_legal_text_tensor,
#     ) = embedding_processor.compute_word2vec_embedding(
#         df_train["Process_description"],
#         df_train["Text"],
#         df_test["Process_description"],
#         df_test["Text"],
#     )
#     embedding_processor.save_embeddings(
#         w2v_train_proc_desc,
#         project_dir / "data/processed/embeddings/w2v_train_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         w2v_train_legal_text,
#         project_dir / "data/processed/embeddings/w2v_train_legal_text.pkl",
#     )
#     embedding_processor.save_embeddings(
#         w2v_test_proc_desc,
#         project_dir / "data/processed/embeddings/w2v_test_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         w2v_test_legal_text,
#         project_dir / "data/processed/embeddings/w2v_test_legal_text.pkl",
#     )
#
#     torch.save(
#         w2v_train_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/w2v_train_proc_desc_tensor.pt",
#     )
#     torch.save(
#         w2v_train_legal_text_tensor,
#         project_dir / "data/processed/embeddings/w2v_train_legal_text_tensor.pt",
#     )
#     torch.save(
#         w2v_test_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/w2v_test_proc_desc_tensor.pt",
#     )
#     torch.save(
#         w2v_test_legal_text_tensor,
#         project_dir / "data/processed/embeddings/w2v_test_legal_text_tensor.pt",
#     )
#
#     # # BERT
#     print("Computing BERT embeddings with separate process description and text ...")
#     (
#         bert_train_proc_desc,
#         bert_train_legal_text,
#         bert_test_proc_desc,
#         bert_test_legal_text,
#         bert_train_proc_desc_tensor,
#         bert_train_legal_text_tensor,
#         bert_test_proc_desc_tensor,
#         bert_test_legal_text_tensor,
#     ) = embedding_processor.compute_bert_embedding(
#         df_train["Process_description"],
#         df_train["Text"],
#         df_test["Process_description"],
#         df_test["Text"],
#     )
#     embedding_processor.save_embeddings(
#         bert_train_proc_desc,
#         project_dir / "data/processed/embeddings/bert_train_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         bert_train_legal_text,
#         project_dir / "data/processed/embeddings/bert_train_legal_text.pkl",
#     )
#     embedding_processor.save_embeddings(
#         bert_test_proc_desc,
#         project_dir / "data/processed/embeddings/bert_test_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         bert_test_legal_text,
#         project_dir / "data/processed/embeddings/bert_test_legal_text.pkl",
#     )
#
#     torch.save(
#         bert_train_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/bert_train_proc_desc_tensor.pt",
#     )
#     torch.save(
#         bert_train_legal_text_tensor,
#         project_dir / "data/processed/embeddings/bert_train_legal_text_tensor.pt",
#     )
#
#     torch.save(
#         bert_test_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/bert_test_proc_desc_tensor.pt",
#     )
#     torch.save(
#         bert_test_legal_text_tensor,
#         project_dir / "data/processed/embeddings/bert_test_legal_text_tensor.pt",
#     )
#
#     # GPT ADA
#     print("Computing GPT embeddings with separate process description and text ...")
#     (
#         gpt_train_proc_desc,
#         gpt_train_legal_text,
#         gpt_test_proc_desc,
#         gpt_test_legal_text,
#         gpt_train_proc_desc_tensor,
#         gpt_train_legal_text_tensor,
#         gpt_test_proc_desc_tensor,
#         gpt_test_legal_text_tensor,
#     ) = embedding_processor.compute_gpt_embedding(
#         df_train["Process_description"],
#         df_train["Text"],
#         df_test["Process_description"],
#         df_test["Text"],
#     )
#     embedding_processor.save_embeddings(
#         gpt_train_proc_desc,
#         project_dir / "data/processed/embeddings/gpt_train_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         gpt_train_legal_text,
#         project_dir / "data/processed/embeddings/gpt_train_legal_text.pkl",
#     )
#     embedding_processor.save_embeddings(
#         gpt_test_proc_desc,
#         project_dir / "data/processed/embeddings/gpt_test_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         gpt_test_legal_text,
#         project_dir / "data/processed/embeddings/gpt_test_legal_text.pkl",
#     )
#     torch.save(
#         gpt_train_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/gpt_train_proc_desc_tensor.pt",
#     )
#     torch.save(
#         gpt_train_legal_text_tensor,
#         project_dir / "data/processed/embeddings/gpt_train_legal_text_tensor.pt",
#     )
#     torch.save(
#         gpt_test_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/gpt_test_proc_desc_tensor.pt",
#     )
#     torch.save(
#         gpt_test_legal_text_tensor,
#         project_dir / "data/processed/embeddings/gpt_test_legal_text_tensor.pt",
#     )
#
#     # FastText
#     print(
#         "Computing FastText embeddings with separate process description and text ..."
#     )
#     (
#         ft_train_proc_desc,
#         ft_train_legal_text,
#         ft_test_proc_desc,
#         ft_test_legal_text,
#         ft_train_proc_desc_tensor,
#         ft_train_legal_text_tensor,
#         ft_test_proc_desc_tensor,
#         ft_test_legal_text_tensor,
#     ) = embedding_processor.compute_fasttext_embedding(
#         df_train["Process_description"],
#         df_train["Text"],
#         df_test["Process_description"],
#         df_test["Text"],
#     )
#     embedding_processor.save_embeddings(
#         ft_train_proc_desc,
#         project_dir / "data/processed/embeddings/ft_train_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         ft_train_legal_text,
#         project_dir / "data/processed/embeddings/ft_train_legal_text.pkl",
#     )
#     embedding_processor.save_embeddings(
#         ft_test_proc_desc,
#         project_dir / "data/processed/embeddings/ft_test_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         ft_test_legal_text,
#         project_dir / "data/processed/embeddings/ft_test_legal_text.pkl",
#     )
#
#     torch.save(
#         ft_train_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/ft_train_proc_desc_tensor.pt",
#     )
#     torch.save(
#         ft_train_legal_text_tensor,
#         project_dir / "data/processed/embeddings/ft_train_legal_text_tensor.pt",
#     )
#     torch.save(
#         ft_test_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/ft_test_proc_desc_tensor.pt",
#     )
#     torch.save(
#         ft_test_legal_text_tensor,
#         project_dir / "data/processed/embeddings/ft_test_legal_text_tensor.pt",
#     )
#
#     # Glove
#     print("Computing GloVe embeddings with separate process description and text ...")
#     (
#         glove_train_proc_desc,
#         glove_train_legal_text,
#         glove_test_proc_desc,
#         glove_test_legal_text,
#         glove_train_proc_desc_tensor,
#         glove_train_legal_text_tensor,
#         glove_test_proc_desc_tensor,
#         glove_test_legal_text_tensor,
#     ) = embedding_processor.compute_glove_embedding(
#         df_train["Process_description"],
#         df_train["Text"],
#         df_test["Process_description"],
#         df_test["Text"],
#     )
#     embedding_processor.save_embeddings(
#         glove_train_proc_desc,
#         project_dir / "data/processed/embeddings/glove_train_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         glove_train_legal_text,
#         project_dir / "data/processed/embeddings/glove_train_legal_text.pkl",
#     )
#     embedding_processor.save_embeddings(
#         glove_test_proc_desc,
#         project_dir / "data/processed/embeddings/glove_test_proc_desc.pkl",
#     )
#     embedding_processor.save_embeddings(
#         glove_test_legal_text,
#         project_dir / "data/processed/embeddings/glove_test_legal_text.pkl",
#     )
#
#     torch.save(
#         glove_train_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/glove_train_proc_desc_tensor.pt",
#     )
#     torch.save(
#         glove_train_legal_text_tensor,
#         project_dir / "data/processed/embeddings/glove_train_legal_text_tensor.pt",
#     )
#     torch.save(
#         glove_test_proc_desc_tensor,
#         project_dir / "data/processed/embeddings/glove_test_proc_desc_tensor.pt",
#     )
#     torch.save(
#         glove_test_legal_text_tensor,
#         project_dir / "data/processed/embeddings/glove_test_legal_text_tensor.pt",
#     )
#
#
# if __name__ == "__main__":
#     main()
