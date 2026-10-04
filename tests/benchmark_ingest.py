"""Ingestion scaling benchmark for MAX_CHUNKS_PER_DOCUMENT.

Measures the three heavy stages of document processing against the
gunicorn --timeout 180 budget:

  1. near_duplicate_mask  - O(n^2) Python loops (views.py:300)
  2. Embedder.embed       - batched ONNX/torch inference (views.py:297)
  3. TextChunker.chunk_pages + DB bulk insert of chunk rows

Prints a per-stage table and a verdict on whether the provisional
MAX_CHUNKS_PER_DOCUMENT=5000 fits the budget. No files are uploaded and
no vectors are persisted; a temp Document row is used only to time the
Chunk bulk insert and is deleted afterwards.
"""
import os
import sys
import time

sys.path.insert(0, r"D:\v5\RAG-p2")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django

django.setup()

import numpy as np
from django.conf import settings

from rag import TextChunker
from rag.cleaning import near_duplicate_mask
from rag.embedder import Embedder

PASSED = 0
FAILED = 0


def ok(label, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print("PASS -", label, flush=True)
    else:
        FAILED += 1
        print("FAIL -", label, (":: " + str(extra)) if extra else "", flush=True)


def distinct_texts(n, seed=3):
    rng = np.random.default_rng(seed)
    words = ["sampling", "aliasing", "window", "residual", "tolerance",
             "calibration", "monsoon", "sediment", "reactor", "cipher",
             "harvest", "weld", "glacier", "substrate", "compass"]
    out = []
    for _ in range(n):
        k = 60 + int(rng.integers(0, 30))
        picks = rng.integers(0, len(words), size=k)
        out.append(" ".join(words[i] for i in picks))
    return out


print("=" * 72, flush=True)
print(f"Settings: CHUNK_SIZE={settings.CHUNK_SIZE} CHUNK_OVERLAP={settings.CHUNK_OVERLAP} "
      f"MAX_CHUNKS_PER_DOCUMENT={settings.MAX_CHUNKS_PER_DOCUMENT}", flush=True)
print(f"Gunicorn timeout budget: 180s (deployment; not modified)", flush=True)
print("=" * 72, flush=True)

try:
    # ---------- 1. near_duplicate_mask scaling (embeddings random/dissimilar
    #    = worst case: nothing matches, every pair is scored) ----------
    print("\n[1] near_duplicate_mask scaling", flush=True)
    dedup_times = {}
    for n in (500, 1000, 2000, 4000, int(settings.MAX_CHUNKS_PER_DOCUMENT)):
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((n, 384)).astype(np.float32)
        texts = distinct_texts(n, seed=n)
        t0 = time.perf_counter()
        keep = near_duplicate_mask(emb, texts)
        dt = time.perf_counter() - t0
        dedup_times[n] = dt
        print(f"    n={n:5d}  {dt:7.2f}s  (kept {sum(keep)}/{n})", flush=True)

    ceil = int(settings.MAX_CHUNKS_PER_DOCUMENT)
    ok(f"dedup at ceiling (n={ceil}) < 60s",
       dedup_times[ceil] < 60, dedup_times[ceil])

    # ---------- 2. Embedder throughput ----------
    print("\n[2] Embedder.embed throughput", flush=True)
    embedder = Embedder()
    sample = distinct_texts(500, seed=9)
    t0 = time.perf_counter()
    vecs = embedder.embed(sample)
    dt = time.perf_counter() - t0
    rate = len(sample) / dt
    est_ceiling = ceil / rate
    print(f"    embedded {len(sample)} chunks in {dt:.2f}s -> {rate:.1f} chunks/s",
          flush=True)
    print(f"    estimated embed time for {ceil} chunks: {est_ceiling:.1f}s", flush=True)
    ok("embedder output shape matches count/dim",
       vecs.shape == (500, 384), vecs.shape)
    ok("embed throughput >= 20 chunks/s", rate >= 20, rate)
    ok(f"estimated embed for ceiling < 120s", est_ceiling < 120, est_ceiling)

    # ---------- 3. TextChunker throughput (80-page document) ----------
    print("\n[3] TextChunker.chunk_pages (80 pages)", flush=True)
    page_text = " ".join(distinct_texts(30, seed=11))
    pages = [{"page_number": p + 1, "content": page_text,
              "file_name": "bench.pdf"} for p in range(80)]
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE,
                          chunk_overlap=settings.CHUNK_OVERLAP)
    t0 = time.perf_counter()
    chunks = chunker.chunk_pages(pages)
    dt_chunk = time.perf_counter() - t0
    print(f"    {len(chunks)} chunks from 80 pages in {dt_chunk:.2f}s", flush=True)
    ok("chunker handles 80 pages < 10s", dt_chunk < 10, dt_chunk)
    ok("chunker produced chunks", len(chunks) > 0, len(chunks))

    # ---------- 4. DB stage: bulk insert of ceiling-sized Chunk rows ----------
    print("\n[4] Chunk bulk insert (ceiling-sized document)", flush=True)
    from backend.models import Chunk, Document
    doc = Document.objects.create(title="_bench_tmp", file_type="txt",
                                  chunk_count=0)
    try:
        rows = [Chunk(document=doc, content=texts_i[:400],
                      embedding_id=i, page_number=(i % 80) + 1, chunk_index=i)
                for i, texts_i in enumerate(distinct_texts(ceil, seed=21))]
        t0 = time.perf_counter()
        Chunk.objects.bulk_create(rows, batch_size=1000)
        dt_db = time.perf_counter() - t0
        print(f"    bulk_create {ceil} rows in {dt_db:.2f}s", flush=True)
        ok(f"bulk insert of {ceil} chunks < 15s", dt_db < 15, dt_db)
    finally:
        pk = doc.pk
        doc.delete()
        assert not Chunk.objects.filter(document_id=pk).exists()

    # ---------- 5. total budget ----------
    total_est = est_ceiling + dedup_times[ceil] + dt_chunk + dt_db + 15
    print("\n[5] end-to-end estimate for a ceiling-sized document", flush=True)
    print(f"    embed {est_ceiling:.1f}s + dedup {dedup_times[ceil]:.1f}s "
          f"+ chunk {dt_chunk:.1f}s + db {dt_db:.1f}s + 15s save/margin "
          f"= {total_est:.1f}s", flush=True)
    ok(f"estimated total < 150s (180s gunicorn budget, 30s headroom)",
       total_est < 150, total_est)
    verdict = "CONFIRMED" if total_est < 150 else "NEEDS REDUCTION"
    print(f"\n    MAX_CHUNKS_PER_DOCUMENT={ceil} -> {verdict} "
          f"(estimate {total_est:.1f}s of 180s)", flush=True)

except Exception:
    import traceback
    traceback.print_exc()
    ok("benchmark ran without unexpected crash", False)

print(f"\n{PASSED}/{PASSED + FAILED} passed", flush=True)
sys.exit(1 if FAILED else 0)
