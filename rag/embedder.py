import glob
import json
import os
import numpy as np
from typing import List, Dict, Any
from django.conf import settings


class Embedder:
    """Sentence embedding facade with two backends.

    ``torch`` (default, local/dev/tests):
        sentence-transformers + PyTorch. Importing that stack costs
        ~450MB RSS (torch + sklearn + pandas + model), which exceeds the
        Render free 512MB instance limit at runtime.

    ``onnx`` (production, EMBEDDING_BACKEND=onnx):
        the exact same all-MiniLM-L6-v2 sentence-transformers model exported
        to ONNX at build time (scripts/export_onnx.py), executed with
        onnxruntime + the tokenizers library only. No torch / sklearn / pandas
        import, ~1-3s init, numerically identical vectors (see
        tests/test_embedder_onnx.py parity checks).
    """

    _instance = None
    _model = None
    _session = None
    _tokenizer = None
    _max_seq_length = 256

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None and self._session is None:
            backend = getattr(settings, "EMBEDDING_BACKEND", "torch")
            if backend == "onnx":
                self._init_onnx()
            else:
                self._init_torch()

    def _init_torch(self):
        from sentence_transformers import SentenceTransformer
        model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def _init_onnx(self):
        from tokenizers import Tokenizer
        import onnxruntime as ort

        snapshot = self._resolve_snapshot()
        tokenizer_path = os.path.join(snapshot, "tokenizer.json")
        model_path = os.path.join(snapshot, "model.onnx")

        max_len = 256
        sbert_cfg = os.path.join(snapshot, "sentence_bert_config.json")
        try:
            with open(sbert_cfg, encoding="utf-8") as handle:
                max_len = int(json.load(handle).get("max_seq_length", 256))
        except Exception:
            pass
        self._max_seq_length = max_len

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_len)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", pad_type_id=0)
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        print(f"[INFO] Loaded embedding model (onnx): {model_path}")

    @staticmethod
    def _hub_cache_candidates():
        hubs = []
        for value in (
            os.environ.get("HF_HUB_CACHE"),
            os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            os.path.expanduser("~/.cache/huggingface/hub"),
        ):
            if value and value not in hubs and os.path.isdir(value):
                hubs.append(value)
        return hubs

    @staticmethod
    def _resolve_snapshot():
        """Locate the snapshot dir holding model.onnx + tokenizer.json."""
        explicit = getattr(settings, "EMBEDDING_ONNX_PATH", None) or None
        if explicit:
            snapshot = os.path.dirname(os.path.abspath(explicit))
            if not os.path.isfile(explicit):
                raise RuntimeError(f"EMBEDDING_ONNX_PATH not found: {explicit}")
            return snapshot

        model_name = getattr(settings, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        bare = model_name.split("/")[-1]
        dir_names = []
        if "/" in model_name:
            dir_names.append("models--" + model_name.replace("/", "--"))
        dir_names.append("models--sentence-transformers--" + bare)

        matches = []
        hubs = Embedder._hub_cache_candidates()
        for hub in hubs:
            for dir_name in dir_names:
                pattern = os.path.join(hub, dir_name, "snapshots", "*")
                for snap in glob.glob(pattern):
                    if os.path.isfile(os.path.join(snap, "model.onnx")) and os.path.isfile(
                        os.path.join(snap, "tokenizer.json")
                    ):
                        matches.append(snap)
        if not matches:
            raise RuntimeError(
                "ONNX embedding model not found (model.onnx + tokenizer.json). "
                "It is exported at build time by scripts/export_onnx.py. "
                f"Searched hubs: {hubs or 'none'} for {dir_names}"
            )
        matches.sort(
            key=lambda path: os.path.getmtime(os.path.join(path, "model.onnx")),
            reverse=True,
        )
        return matches[0]

    def _embed_onnx(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 384)
        outputs = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            encodings = self._tokenizer.encode_batch(
                [str(t) for t in texts[start : start + batch_size]]
            )
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
            (hidden,) = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            mask = attention_mask.astype(np.float32)[..., None]
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
            pooled = pooled / np.maximum(
                np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
            )
            outputs.append(pooled.astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts. Returns (n, 384) float32 array."""
        if self._session is not None:
            return self._embed_onnx(texts)
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 384)
        embeddings = self._model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Generate embedding for a single query. Returns (1, 384) float32 array."""
        if self._session is not None:
            return self._embed_onnx([text])
        embedding = self._model.encode([text], convert_to_numpy=True)
        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        return 384
