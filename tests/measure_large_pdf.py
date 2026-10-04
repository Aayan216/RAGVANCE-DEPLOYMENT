"""Plan section 12: real 80-page PDF measurement on a live server.

Uploads a TEMP COPY of the provided 80-page Time Series material (the user's
own processed document stays untouched), processes it through the real HTTP
pipeline, records wall-clock metrics, verifies listings/sources/FAISS
bookkeeping, then deletes the temp copy and proves exact rollback.

Metrics captured:
  - upload wall time (multipart POST -> 302)
  - process wall time (POST /process/<id>/ -> synchronous success JSON)
  - chunk count + ceiling headroom vs MAX_CHUNKS_PER_DOCUMENT
  - FAISS ntotal / DB chunk rows before, during, after
  - delete wall time + rollback verification
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\v5\RAG-p2")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django

django.setup()

import requests
from django.conf import settings

from backend.models import Chunk, Document

SRC_TITLE = "test"                      # user's live document
TMP_TITLE = "ZZ 80p Measurement"        # temp copy, always cleaned up
PORT = os.environ.get("MEASURE_PORT", "8018")
BASE = f"http://127.0.0.1:{PORT}"

PASSED = 0
FAILED = 0


def ok(label, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"PASS - {label}", flush=True)
    else:
        FAILED += 1
        print(f"FAIL - {label}" + (f" :: {extra}" if extra else ""), flush=True)


def faiss_count():
    from rag.vector_store import VectorStore
    return len(VectorStore())


src_doc = Document.objects.filter(title=SRC_TITLE).first()
assert src_doc is not None and src_doc.file, "source 80-page document missing"
SRC_PATH = src_doc.file.path
assert os.path.exists(SRC_PATH), SRC_PATH
PAGES = 80
try:
    import fitz
    _d = fitz.open(SRC_PATH)
    PAGES = _d.page_count
    _d.close()
except Exception:
    pass
FILE_BYTES = os.path.getsize(SRC_PATH)
from config.settings import MAX_CHUNKS_PER_DOCUMENT
MAX_CHUNKS = int(MAX_CHUNKS_PER_DOCUMENT)

base_docs = Document.objects.count()
base_rows = Chunk.objects.count()
base_faiss = None

SERVER = None
server_out = None
r = None

try:
    base_faiss = faiss_count()
    print(f"[baseline] docs={base_docs} chunk_rows={base_rows} "
          f"faiss={base_faiss} | src: {PAGES}p {FILE_BYTES:,}B "
          f"chunks={src_doc.chunk_count}", flush=True)

    server_out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "_measure_server.log"), "w",
                      encoding="utf-8", errors="replace")
    SERVER = subprocess.Popen(
        [sys.executable, "manage.py", "runserver", f"127.0.0.1:{PORT}",
         "--noreload"],
        cwd=str(settings.BASE_DIR), env=dict(os.environ),
        stdout=server_out, stderr=subprocess.STDOUT)

    t0 = time.time()
    up = False
    while time.time() - t0 < 60:
        try:
            if requests.get(BASE + "/", timeout=3).status_code == 200:
                up = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    ok("server booted", up, f"{time.time() - t0:.1f}s")
    if not up:
        raise SystemExit("server failed to boot")

    s = requests.Session()
    s.get(BASE + "/", timeout=20)
    csrf = s.cookies.get("csrftoken", "")
    H = {"X-CSRFToken": csrf}

    # ---- upload temp copy ----
    with open(SRC_PATH, "rb") as fh:
        payload = fh.read()
    t0 = time.time()
    r = s.post(BASE + "/", timeout=120,
               data={"title": TMP_TITLE, "csrfmiddlewaretoken": csrf},
               files={"file": ("timeseries_80p_copy.pdf", payload,
                               "application/pdf")},
               headers={"Referer": BASE + "/"}, allow_redirects=False)
    up_s = time.time() - t0
    doc = Document.objects.filter(title=TMP_TITLE).first()
    ok("upload -> 302 + row", r.status_code == 302 and doc is not None,
       (r.status_code, doc))
    if not doc:
        raise SystemExit("upload failed")

    # ---- process (synchronous) ----
    t0 = time.time()
    r = s.post(BASE + f"/process/{doc.id}/", headers=H, timeout=600)
    proc_s = time.time() - t0
    ctype = r.headers.get("Content-Type", "")
    data = r.json() if "application/json" in ctype else {}
    ok("process -> success JSON", r.status_code == 200
       and data.get("status") == "success" and data.get("chunks", 0) > 0,
       (r.status_code, str(data)[:200]))

    chunks = int(data.get("chunks", 0) or 0)
    ok(f"chunks ({chunks}) under ceiling ({MAX_CHUNKS})",
       0 < chunks < MAX_CHUNKS, (chunks, MAX_CHUNKS))

    mid_rows = Chunk.objects.count()
    mid_faiss = faiss_count()
    ok("chunk rows delta == reported chunks",
       mid_rows - base_rows == chunks, (base_rows, mid_rows, chunks))
    ok("FAISS delta == reported chunks",
       mid_faiss - base_faiss == chunks, (base_faiss, mid_faiss, chunks))

    # ---- listings + sources over HTTP ----
    fid = None
    for path in ("/", "/practice/", "/mock-test/"):
        body = s.get(BASE + path, timeout=20).text
        ok(f"listing {path} shows temp title", TMP_TITLE in body, path)
    from rag.vector_store import VectorStore
    vs = VectorStore()
    metas = [m for m in vs.metadata if m.get("doc_id") == doc.id]
    ok("metadata entries == chunks", len(metas) == chunks, len(metas))
    if metas:
        fid = metas[0].get("faiss_id")
        r = s.get(BASE + f"/sources/?ids={fid}", timeout=20)
        j = r.json() if r.status_code == 200 else {}
        ok("sources endpoint returns text for new vector",
           bool(j.get("sources")), (r.status_code, str(j)[:120]))

    # ---- re-process idempotency ----
    r = s.post(BASE + f"/process/{doc.id}/", headers=H, timeout=120)
    data2 = r.json() if "application/json" in r.headers.get(
        "Content-Type", "") else {}
    ok("re-process -> already_processed",
       r.status_code == 200 and data2.get("status") == "already_processed",
       str(data2)[:120])

    # ---- delete + rollback ----
    fp = doc.file.path
    t0 = time.time()
    r = s.post(BASE + f"/delete/{doc.id}/", headers=H, timeout=120)
    del_s = time.time() - t0
    data3 = r.json() if "application/json" in r.headers.get(
        "Content-Type", "") else {}
    ok("delete -> success JSON", r.status_code == 200
       and data3.get("success") is not False, (r.status_code, str(data3)[:150]))
    ok("temp file removed from disk", not os.path.exists(fp), fp)

    after_docs = Document.objects.count()
    after_rows = Chunk.objects.count()
    after_faiss = faiss_count()
    ok("docs back to baseline", after_docs == base_docs,
       (base_docs, after_docs))
    ok("chunk rows back to baseline", after_rows == base_rows,
       (base_rows, after_rows))
    ok("FAISS back to baseline", after_faiss == base_faiss,
       (base_faiss, after_faiss))
    user_ok = Document.objects.filter(id=src_doc.id, processed=True).exists()
    ok("user's original doc untouched", user_ok)

    print("\n" + "=" * 66, flush=True)
    print("SECTION 12 - REAL 80-PAGE PDF MEASUREMENT", flush=True)
    print("=" * 66, flush=True)
    print(f"  source file              : {FILE_BYTES:,} bytes, {PAGES} pages",
          flush=True)
    print(f"  upload (multipart)       : {up_s:.1f}s", flush=True)
    print(f"  process (sync HTTP)      : {proc_s:.1f}s", flush=True)
    print(f"  delete                   : {del_s:.1f}s", flush=True)
    print(f"  chunks produced          : {chunks} (dedup applied; "
          f"ceiling {MAX_CHUNKS}, headroom "
          f"{MAX_CHUNKS - chunks})", flush=True)
    print(f"  FAISS                    : {base_faiss} -> {mid_faiss} -> "
          f"{after_faiss}", flush=True)
    print(f"  DB chunk rows            : {base_rows} -> {mid_rows} -> "
          f"{after_rows}", flush=True)
    print(f"  gunicorn prod timeout    : 180s budget, process used "
          f"{proc_s:.1f}s ({proc_s / 180 * 100:.0f}%)", flush=True)
    print("=" * 66, flush=True)

except SystemExit as e:
    print(f"aborted: {e}", flush=True)
finally:
    # cleanup temp doc if anything failed mid-way
    try:
        d = Document.objects.filter(title=TMP_TITLE).first()
        if d:
            from rag.vector_store import VectorStore
            fp = d.file.path if d.file else None
            VectorStore().delete_document(d.id)
            d.delete()
            if fp and os.path.exists(fp):
                os.remove(fp)
            print(f"[cleanup] removed temp doc {TMP_TITLE}", flush=True)
    except Exception as exc:
        print(f"[cleanup] error: {exc}", flush=True)
    if SERVER:
        SERVER.terminate()
        try:
            SERVER.wait(timeout=15)
        except Exception:
            SERVER.kill()
    if server_out:
        server_out.close()

print(f"\n{PASSED}/{PASSED + FAILED} passed", flush=True)
sys.exit(0 if FAILED == 0 else 1)
