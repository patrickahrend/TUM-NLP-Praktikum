import numpy as np


def get_sentence_vector(statement, model) -> np.ndarray:
    words = [word for word in statement if word in model.wv.key_to_index]
    if len(words) >= 1:
        return np.mean(model.wv[words], axis=0)
    else:
        return np.zeros(model.vector_size)
