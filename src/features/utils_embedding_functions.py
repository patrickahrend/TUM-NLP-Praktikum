import numpy as np
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

    ## we are interested in the pooled output, not the hidden states as these are the full sentence embedding
    pooled_output = output.pooler_output
    return pooled_output


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
