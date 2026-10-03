import os
import subprocess
import sys
import time

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, r"D:\v5\RAG-p2")
os.chdir(r"D:\v5\RAG-p2")

PASSED = 0
FAILED = 0


def ok(label, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print("PASS:", label, flush=True)
    else:
        FAILED += 1
        print("FAIL:", label, ("| " + str(extra)) if extra != "" else "", flush=True)


import django

django.setup()

import numpy as np
from django.conf import settings

from rag.embedder import Embedder

ok("default EMBEDDING_BACKEND is torch (local/dev/tests unchanged)",
   settings.EMBEDDING_BACKEND == "torch", settings.EMBEDDING_BACKEND)

# ---------------- ensure build-time ONNX model exists ----------------
snap = None
try:
    snap = Embedder._resolve_snapshot()
except RuntimeError:
    proc = subprocess.run(
        [sys.executable, "scripts/export_onnx.py"],
        capture_output=True, text=True, timeout=600,
        cwd=r"D:\v5\RAG-p2",
    )
    ok("export script produced verified model.onnx",
       proc.returncode == 0 and "verified max_abs_diff" in proc.stdout,
       (proc.stdout + proc.stderr)[-1500:])
    snap = Embedder._resolve_snapshot()
onnx_path = os.path.join(snap, "model.onnx")
ok("model.onnx present and plausible size",
   os.path.isfile(onnx_path) and os.path.getsize(onnx_path) >= 40_000_000,
   onnx_path)
ok("tokenizer.json present next to model.onnx",
   os.path.isfile(os.path.join(snap, "tokenizer.json")), snap)

TEXTS = [
    "What is the Zephyr protocol checklist requirement for relay nodes?",
    "short",
    "Marigold nutrient guidelines require pH between 5.8 and 6.2 with an ideal "
    "target of 6.0 and electrical conductivity near 1.4 mS/cm for lettuce crops. "
    "Water temperature must stay at 20 degrees Celsius throughout the growth "
    "cycle for stable dissolved oxygen across every growing channel.",
    "Caf\u00e9 na\u00efve \u2014 \u4e2d\u6587\u6d4b\u8bd5 \U0001f393 emoji",
    "pH 5.8-6.2, EC 1.4 mS/cm, t=20C; e.g. 42% (w/v) @ 37C -- 101%+7?",
    "   ",
    "word " * 300,
]

# ---------------- torch (sentence-transformers) reference ----------------
Embedder._instance = None
settings.EMBEDDING_BACKEND = "torch"
t0 = time.perf_counter()
torch_emb = Embedder().embed(TEXTS)
torch_ms = (time.perf_counter() - t0) * 1000
torch_empty = Embedder().embed([])
torch_query = Embedder().embed_query("What is the Zephyr protocol?")
ok("torch backend shape", torch_emb.shape == (len(TEXTS), 384), torch_emb.shape)
ok("torch backend dtype float32", torch_emb.dtype == np.float32, torch_emb.dtype)
ok("torch embed_query shape", torch_query.shape == (1, 384), torch_query.shape)
ok("torch empty input shape", torch_empty.shape == (0, 384), torch_empty.shape)

# ---------------- onnx backend ----------------
Embedder._instance = None
settings.EMBEDDING_BACKEND = "onnx"
t0 = time.perf_counter()
onnx_emb = Embedder().embed(TEXTS)
onnx_ms = (time.perf_counter() - t0) * 1000
onnx_empty = Embedder().embed([])
onnx_query = Embedder().embed_query("What is the Zephyr protocol?")
same = Embedder() is Embedder()
settings.EMBEDDING_BACKEND = "torch"
Embedder._instance = None

ok("onnx backend shape", onnx_emb.shape == (len(TEXTS), 384), onnx_emb.shape)
ok("onnx backend dtype float32", onnx_emb.dtype == np.float32, onnx_emb.dtype)
ok("onnx embed_query shape", onnx_query.shape == (1, 384), onnx_query.shape)
ok("onnx empty input shape", onnx_empty.shape == (0, 384), onnx_empty.shape)
ok("onnx singleton identity", same is True)

# ---------------- parity: same vectors as sentence-transformers ----------------
torch_norms = np.linalg.norm(torch_emb, axis=1)
onnx_norms = np.linalg.norm(onnx_emb, axis=1)
ok("torch vectors L2-normalized", np.allclose(torch_norms, 1.0, atol=1e-4),
   np.round(torch_norms, 5))
ok("onnx vectors L2-normalized", np.allclose(onnx_norms, 1.0, atol=1e-4),
   np.round(onnx_norms, 5))

max_abs = float(np.abs(torch_emb - onnx_emb).max())
ok(f"torch vs onnx max abs diff < 1e-3 (got {max_abs:.6f})", max_abs < 1e-3, max_abs)

dots = (torch_emb * onnx_emb).sum(axis=1)
row_cos = dots / (torch_norms * onnx_norms)
ok(f"torch vs onnx min row cosine > 0.9999 (got {row_cos.min():.6f})",
   float(row_cos.min()) > 0.9999, np.round(row_cos, 6))

q_cos = float(
    (torch_query[0] @ onnx_query[0])
    / (np.linalg.norm(torch_query[0]) * np.linalg.norm(onnx_query[0]))
)
ok(f"embed_query cosine > 0.9999 (got {q_cos:.6f})", q_cos > 0.9999, q_cos)

ok(f"onnx embed latency sane ({onnx_ms:.0f}ms for {len(TEXTS)} texts)",
   onnx_ms < 5000, onnx_ms)

# ---------------- EMBEDDING_ONNX_PATH override ----------------
prev_path = settings.EMBEDDING_ONNX_PATH
settings.EMBEDDING_ONNX_PATH = onnx_path
settings.EMBEDDING_BACKEND = "onnx"
Embedder._instance = None
try:
    overridden = Embedder().embed([TEXTS[0]])
    ok("EMBEDDING_ONNX_PATH override works and matches",
       overridden.shape == (1, 384)
       and np.allclose(overridden[0], onnx_emb[0], atol=1e-5),
       overridden.shape)
finally:
    settings.EMBEDDING_ONNX_PATH = prev_path
    settings.EMBEDDING_BACKEND = "torch"
    Embedder._instance = None

# ---------------- subprocess: runtime works with torch/ST unavailable ----------------
# Mimics the Render production image after `pip uninstall sentence-transformers torch`:
# the onnx embedder and the chunker import path must both work without them.
CHILD = r"""
import os, sys
sys.path.insert(0, r"D:\v5\RAG-p2")
os.chdir(r"D:\v5\RAG-p2")

class Blocker:
    BLOCKED = {"sentence_transformers", "torch"}
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError("blocked in test: " + name, name=name)
        return None

sys.meta_path.insert(0, Blocker())
os.environ["EMBEDDING_BACKEND"] = "onnx"
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()
import numpy as np
from rag.embedder import Embedder
v = Embedder().embed(["hello world", "second sentence"])
print("EMBED_SHAPE=", v.shape, flush=True)
print("EMBED_NORM=", round(float(np.linalg.norm(v[0])), 5), flush=True)
from rag.chunker import TextChunker
chunks = TextChunker().chunk_pages([{"content": "alpha beta gamma delta. " * 40, "page_number": 1, "file_name": "x.txt"}])
print("CHUNKS=", len(chunks), flush=True)
import langchain_text_splitters
print("LANGCHAIN_SPLITTERS_OK", flush=True)
heavy = sorted(set(m.split(".")[0] for m in sys.modules) & {"torch", "sentence_transformers", "sklearn", "pandas", "transformers"})
print("HEAVY=", ",".join(heavy), flush=True)
print("CHILD_DONE", flush=True)
"""

child = subprocess.run(
    [sys.executable, "-c", CHILD],
    cwd=r"D:\v5\RAG-p2", capture_output=True, text=True, timeout=300,
)
out = child.stdout + child.stderr


def child_flag(name):
    line = next((l for l in out.splitlines() if l.startswith(name)), None)
    return line.split("=", 1)[1].strip() if line else None


ok("child (torch/ST blocked) exited 0", child.returncode == 0, out[-2000:])
ok("onnx embed works without torch/ST", child_flag("EMBED_SHAPE=") == "(2, 384)",
   child_flag("EMBED_SHAPE=") or out[-1500:])
ok("child embedding normalized", child_flag("EMBED_NORM=") in ("1.0", "1.0"),
   child_flag("EMBED_NORM="))
ok("TextChunker works without torch/ST",
   child_flag("CHUNKS=") not in (None, "0"), child_flag("CHUNKS="))
ok("langchain_text_splitters imports without torch/ST",
   "LANGCHAIN_SPLITTERS_OK" in out, out[-1500:])
ok("torch/sentence-transformers/sklearn/pandas absent in child",
   child_flag("HEAVY=") in ("", "transformers"), child_flag("HEAVY="))
ok("child completed", "CHILD_DONE" in out, out[-1500:])

print(f"\n===== {PASSED}/{PASSED + FAILED} passed =====", flush=True)
sys.exit(1 if FAILED else 0)
