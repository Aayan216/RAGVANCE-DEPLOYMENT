import numpy as np
from typing import List, Dict, Any
from django.conf import settings


class Embedder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self._model = SentenceTransformer(model_name)
            print(f"[INFO] Loaded embedding model: {model_name}")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts. Returns (n, 384) float32 array."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 384)
        embeddings = self._model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single query. Returns (1, 384) float32 array."""
        embedding = self._model.encode([text], convert_to_numpy=True)
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        return 384