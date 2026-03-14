from __future__ import annotations

import hashlib
from typing import List

from memrl.providers.base import BaseEmbedder, EmbedderError


class HashEmbedder(BaseEmbedder):
    def __init__(self, embedding_dim: int = 384) -> None:
        super().__init__(max_text_len=0)
        self.embedding_dim = embedding_dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings: List[List[float]] = []
        for text in texts:
            if not isinstance(text, str):
                raise EmbedderError("HashEmbedder expects string inputs")
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = []
            for i in range(0, len(digest), 4):
                chunk = digest[i:i + 4]
                if len(chunk) < 4:
                    chunk = chunk.ljust(4, b"\0")
                value = int.from_bytes(chunk, "little")
                normalized = (value % 1000) / 500.0 - 1.0
                values.append(normalized)
            embedding = [values[i % len(values)] for i in range(self.embedding_dim)]
            embeddings.append(embedding)
        return embeddings
