import io
import os
import shutil
import tempfile
import zipfile
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from django.conf import settings


class VectorStore:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or str(getattr(settings, "VECTOR_STORE_PATH", "vector_store"))
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index_path = os.path.join(self.persist_dir, "faiss.index")
        self.meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = None
        self.metadata = []
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load()
        elif self._restore_from_cloud():
            self.load()
        else:
            self.index = faiss.IndexFlatIP(384)
            self.metadata = []

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> List[int]:
        """Add embeddings to index. Returns list of FAISS IDs assigned."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        
        start_id = self.index.ntotal
        self.index.add(embeddings)
        
        for i, meta in enumerate(metadata):
            meta["faiss_id"] = start_id + i
            self.metadata.append(meta)
        
        return list(range(start_id, start_id + len(embeddings)))

    def search(self, query_embedding: np.ndarray, k: int = 5, doc_ids: List[int] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors. Optionally filter by document IDs."""
        if self.index.ntotal == 0:
            return []
        
        fetch_k = min(k * 3, self.index.ntotal) if doc_ids else min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, fetch_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                if doc_ids and meta.get("doc_id") not in doc_ids:
                    continue
                meta["score"] = float(score)
                results.append(meta)
                if len(results) >= k:
                    break
        
        return results

    def get_by_id(self, faiss_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata by FAISS ID."""
        for meta in self.metadata:
            if meta.get("faiss_id") == faiss_id:
                return meta
        return None

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved FAISS index ({self.index.ntotal} vectors) to {self.persist_dir}")
        self._mirror_to_cloud()

    VECTOR_BUNDLE_KEY = "vector_index.zip"
    VECTOR_BUCKET = "ragvance-vectors"

    def _mirror_to_cloud(self):
        """Best-effort: upload faiss.index + metadata.pkl as ONE zip object.

        A single upsert means the remote pair is replaced atomically — one
        upload cannot succeed while the other fails. Failures only warn and
        never raise; the local files remain the source of truth.
        """
        try:
            from backend import supabase_storage as storage

            if not storage.is_enabled():
                return
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(self.index_path, "faiss.index")
                bundle.write(self.meta_path, "metadata.pkl")
            storage.upload_file(
                self.VECTOR_BUNDLE_KEY, buffer.getvalue(), bucket=self.VECTOR_BUCKET
            )
        except Exception as exc:
            print(f"[WARN] FAISS cloud mirror skipped: {type(exc).__name__}")

    def _restore_from_cloud(self):
        """Download the bundle; materialize BOTH local files only on success.

        Runs only when the local pair is missing. Extraction happens in a temp
        directory inside persist_dir (same filesystem); files are moved into
        place only after both members unpack cleanly. A failure between the
        two moves self-heals on the next start (pair incomplete -> restore
        runs again).
        """
        try:
            from backend import supabase_storage as storage

            if not storage.is_enabled():
                return False
            data = storage.download_file(self.VECTOR_BUNDLE_KEY, bucket=self.VECTOR_BUCKET)
            workdir = tempfile.mkdtemp(dir=self.persist_dir, prefix=".restore_")
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as bundle:
                    names = set(bundle.namelist())
                    if "faiss.index" not in names or "metadata.pkl" not in names:
                        print("[WARN] FAISS restore skipped: incomplete bundle")
                        return False
                    bundle.extract("faiss.index", workdir)
                    bundle.extract("metadata.pkl", workdir)
                os.replace(os.path.join(workdir, "faiss.index"), self.index_path)
                os.replace(os.path.join(workdir, "metadata.pkl"), self.meta_path)
                print(f"[INFO] Restored FAISS bundle from Supabase Storage")
                return True
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        except Exception as exc:
            print(f"[WARN] FAISS restore skipped: {type(exc).__name__}")
            return False

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded FAISS index ({self.index.ntotal} vectors) from {self.persist_dir}")

    def clear(self):
        self.index = faiss.IndexFlatIP(384)
        self.metadata = []
        self.save()

    def delete_document(self, doc_id: int):
        """Remove all vectors belonging to a document and rebuild the index."""
        indices_to_remove = set()
        for i, meta in enumerate(self.metadata):
            if meta.get("doc_id") == doc_id:
                indices_to_remove.add(i)

        if not indices_to_remove:
            return

        keep_vectors = []
        keep_metadata = []
        for i in range(self.index.ntotal):
            if i not in indices_to_remove:
                keep_vectors.append(self.index.reconstruct(i))
                meta = self.metadata[i].copy()
                meta["faiss_id"] = len(keep_metadata)
                keep_metadata.append(meta)

        if keep_vectors:
            vectors = np.array(keep_vectors, dtype=np.float32)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        else:
            self.index = faiss.IndexFlatIP(384)

        self.metadata = keep_metadata
        self.save()

    def __len__(self):
        return self.index.ntotal if self.index else 0