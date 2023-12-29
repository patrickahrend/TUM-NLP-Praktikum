import numpy as np
import torch


def get_sentence_vector(statement, model) -> np.ndarray:
    words = [word for word in statement if word in model.wv.key_to_index]
    if len(words) >= 1:
        return np.mean(model.wv[words], axis=0)
    else:
        return np.zeros(model.vector_size)


def get_embeddings_bert(statement, tokenizer, model) -> torch.Tensor:
    bert_input = tokenizer(
        statement, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    with torch.no_grad():
        output = model(**bert_input)
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
