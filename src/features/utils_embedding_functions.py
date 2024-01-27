import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_sentence_vector_custom(statement, model, is_glove=False) -> np.ndarray:
    if is_glove:
        # For GloVe, the model doesn't have a `wv` property
        words = [word for word in statement if word in model.key_to_index]
        if len(words) >= 1:
            return np.mean(model[words], axis=0).reshape(1, -1)
        else:
            return np.zeros(model.vector_size)
    else:
        # For Word2Vec
        words = [word for word in statement if word in model.wv.key_to_index]
        if len(words) >= 1:
            return np.mean(model.wv[words], axis=0).reshape(1, -1)
        else:
            return np.zeros(model.vector_size)


def get_embeddings_bert(statement, tokenizer, model) -> torch.Tensor:
    bert_input = tokenizer(
        statement, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    with torch.no_grad():
        output = model(**bert_input)

    ## use hidden state again, as pooler output did not perform well
    embeddings_vector = output.last_hidden_state.mean(dim=1).squeeze()
    return embeddings_vector


def get_embeddings_gpt(statement, client):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=statement,
            encoding_format="float",
        )
        first_embedding = response.data[0]
        embeddings_vector = first_embedding.embedding
        return embeddings_vector
    except Exception as e:
        print(e)
        return None

def load_model_if_exists(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def save_pickle(obj, filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_training_data():
    # Load the training data needed for models like Word2Vec, GloVe, and FastText
    project_dir = Path(__file__).resolve().parents[2]
    training_data_path = project_dir / "data/processed/training_data_preprocessed.csv"
    training_data = pd.read_csv(training_data_path)

    # combine process description and legal text to train embeddings on whole corpus
    text = training_data["Process_description"] + " " + training_data["Text"]

    # Tokenize sentences for fasttext and w2v
    tokenized_sentences = [statement.split() for statement in text]

    # Raw sentences for TF-IDF
    raw_sentences = text.tolist()

    return tokenized_sentences, raw_sentences