import numpy as np
from transformers import AutoModel
from numpy.linalg import norm

import config


# trust_remote_code is needed to use the encode method
embedding_model = AutoModel.from_pretrained(
    config.EMBEDDING_MODEL_CHECKPOINT,
    trust_remote_code=True,
    revision=config.EMBEDDING_MODEL_HASH
)


def embed_sentences(*sentences: str, max_length=2048) -> np.ndarray:
    embeddings = embedding_model.encode(sentences, max_length=max_length)
    return embeddings


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return (a @ b.T) / (norm(a)*norm(b))
