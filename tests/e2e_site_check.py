"""Full-site end-to-end check: routes, security, live HTTP journey, LLM
features, integrity, and memory.

Part A  in-process Django client: route sweep, method matrix, CSRF, XSS,
        ID/404 handling, DB+FAISS integrity, sources endpoint.
Part B  real HTTP against a live runserver subprocess: static/media, upload
        -> process -> listings -> sources -> tutor/practice/mock journeys
        (LLM-dependent steps degrade to WARN if the AI service is
        unreachable, never crash), concurrency smoke, delete + rollback.
Part C  embedder memory footprint (< 450MB working set on the 512MB budget).
Part D  config sanity (key present, never printed).

Cleanup: fixtures removed; the script NEVER calls vector_store.save() on its
own in-process instance after Part B (the server owns the disk index then);
leftover processed docs are removed with a freshly-loaded store.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, r"D:\v5\RAG-p2")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
# Measure/serve the production embedding backend (ONNX). Local torch stays
# available via the runserver subprocess (it reads its own env/.env).
os.environ.setdefault("EMBEDDING_BACKEND", "onnx")
import django

django.setup()

import requests
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from backend import views
from backend.models import (Chunk, Document, MockTest, TestAttempt,
                            TestQuestion)

PASSED = 0
FAILED = 0
WARNED = 0
SERVER = None
KNOWN_TITLES = {"E2E Journey Doc", "E2E XSS Probe"}
CREATED_TEST_IDS = []
mock_baseline_ids = set()


T_START = time.time()


def ts():
    return f"[{time.time() - T_START:6.0f}s]"


def ok(label, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"{ts()} PASS -", label, flush=True)
    else:
        FAILED += 1
        print(f"{ts()} FAIL -", label, (":: " + str(extra)) if extra else "",
              flush=True)


def warn(label, extra=""):
    global WARNED
    WARNED += 1
    print(f"{ts()} WARN -", label, (":: " + str(extra)) if extra else "",
          flush=True)


def section(title):
    print("\n" + "=" * 70, flush=True)
    print(title, flush=True)
    print("=" * 70, flush=True)


LEAK_MARKERS = ("Traceback (most recent call last)", "Exception Value",
                "Site doesn't have DEBUG")


def leak_free(text):
    return not any(m in text for m in LEAK_MARKERS)


def rss_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        buf = PMC()
        buf.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(),
                                      ctypes.byref(buf), buf.cb):
            return buf.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    try:  # fallback: PowerShell working set
        import subprocess as _sp
        out = _sp.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {os.getpid()}).WorkingSet64"],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip()) / (1024 * 1024)
    except Exception:
        return -1.0


# Self-generated fixture: never depends on user media (previous corpus files
# may be deleted between runs).
PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_e2e_fixture.pdf")
if os.path.exists(PDF_PATH):
    os.remove(PDF_PATH)
import fitz as _fitz
_gen = _fitz.open()
for _p in range(3):
    _page = _gen.new_page()
    for _i in range(20):
        _seed = _p * 20 + _i
        if _i % 2 == 0:
            _line = (f"Fixture page {_p + 1} line {_i + 1}: database management "
                     f"systems course note {2 * _seed + 1} covers index "
                     f"selection and join order for query {_seed}.")
        else:
            _line = (f"Fixture page {_p + 1} line {_i + 1}: relational schema "
                     f"design in database management systems module {_seed} "
                     f"describes keys and constraints for table {_seed + 7}.")
        _page.insert_text((72, 72 + _i * 14), _line)
_gen.save(PDF_PATH)
_gen.close()
PORT = os.environ.get("E2E_PORT", "8017")
BASE = f"http://127.0.0.1:{PORT}"

# =====================================================================
section("PART A - in-process route sweep / security / integrity")
# =====================================================================

c = Client()

# --- A1. every URL in backend.urls, correct method, no 500/leak ---
sweep = [
    ("get", "/", (200,), "html"),
    ("get", "/tutor/", (200,), "html"),
    ("get", "/practice/", (200,), "html"),
    ("get", "/mock-test/", (200,), "html"),
    ("get", "/mock-test/terminated/", (200,), "html"),
    ("get", "/mock-test/start/", (200,), "html"),       # delegates to settings GET
    ("get", "/sources/?ids=0", (200,), "json"),         # faiss id 0 may not exist
    ("get", "/sources/", (400,), "json"),
    ("post", "/tutor/ask/", (400, 500), "json"),       # empty body -> benign JSON
    ("get", "/practice/generate/", (405,), "any"),
    ("get", "/practice/submit/", (405,), "any"),
    ("get", "/mock-test/terminate/", (405,), "any"),
    ("post", "/mock-test/start/", (400, 405), "json"),  # invalid JSON body
    ("get", "/process/999999/", (405,), "json"),
    ("post", "/process/999999/", (404,), "json"),
    ("get", "/delete/999999/", (405,), "json"),
    ("post", "/delete/999999/", (404,), "json"),
    ("get", "/process/abc/", (404,), "any"),            # int converter rejects
    ("post", "/process/abc/", (404,), "any"),
    ("get", "/delete/abc/", (404,), "any"),
    ("post", "/delete/%2e%2e%2f/", (404,), "any"),      # traversal attempt
    ("get", "/mock-test/999999/", (404,), "html"),
    ("get", "/mock-test/999999/submit/", (405,), "any"),
    ("get", "/mock-test/999999/result/", (404,), "html"),
    ("get", "/mock-test/999999/analysis/", (404,), "html"),
    ("get", "/mock-test/999999/review/", (404,), "html"),
    ("get", "/definitely-missing/", (404,), "html"),
    ("delete", "/process/999999/", (405,), "any"),
    ("put", "/process/999999/", (405,), "any"),
]
for method, url, allowed, kind in sweep:
    r = getattr(c, method)(url)
    label = f"{method.upper()} {url} -> {r.status_code} (allowed {allowed})"
    text = r.content.decode("utf-8", "replace")
    good = r.status_code in allowed and leak_free(text)
    if kind == "html" and good:
        good = "</html>" in text and "text/html" in r.headers.get("Content-Type", "")
    if kind == "json" and good:
        good = "application/json" in r.headers.get("Content-Type", "")
    ok(label, good, text[:120] if not good else "")

# --- A2. CSRF enforcement on state-changing endpoints ---
ec = Client(enforce_csrf_checks=True)
r = ec.post("/process/999999/", {})
ok("CSRF: POST /process/ without token -> 403", r.status_code == 403, r.status_code)
r = ec.post("/", {"title": "x", "file": SimpleUploadedFile("a.txt", b"x")})
ok("CSRF: POST / (upload) without token -> 403", r.status_code == 403, r.status_code)

# --- A3. XSS: malicious title is escaped in rendered HTML ---
xss_title = '"><script>window.__xss=1</script>'
up = SimpleUploadedFile("e2e_xss.txt", b"probe payload for xss test",
                        content_type="text/plain")
r = c.post("/", {"title": xss_title, "file": up})
xss_doc = Document.objects.filter(title=xss_title).first()
ok("XSS probe uploaded", r.status_code == 302 and xss_doc is not None, r.status_code)
body = c.get("/").content.decode("utf-8", "replace")
ok("XSS title not injected raw into list page",
   "<script>window.__xss" not in body)
ok("XSS title HTML-escaped in list page",
   "&lt;script&gt;window.__xss" in body
   or "&quot;&gt;&lt;script&gt;" in body)
if xss_doc:
    fp = xss_doc.file.path if xss_doc.file else None
    xss_doc.delete()
    if fp and os.path.exists(fp):
        os.remove(fp)

# --- A4. integrity: DB rows vs FAISS index vs files on disk ---
docs = list(Document.objects.filter(processed=True))
chunk_rows = Chunk.objects.count()
sum_cc = sum(d.chunk_count for d in docs)
vs = views.vector_store
ntotal = len(vs)
ok(f"Chunk rows ({chunk_rows}) == sum(processed.chunk_count) ({sum_cc})",
   chunk_rows == sum_cc, (chunk_rows, sum_cc))
ok(f"FAISS ntotal ({ntotal}) == Chunk rows ({chunk_rows})",
   ntotal == chunk_rows, (ntotal, chunk_rows))
missing_files = [d.id for d in docs
                 if not d.file or not os.path.exists(d.file.path)
                 or os.path.getsize(d.file.path) == 0]
ok("every processed document has a non-empty file on disk",
   not missing_files, missing_files)
empty_processed = [d.id for d in docs if d.chunk_count == 0]
print(f"    [info] processed docs={len(docs)}, chunk rows={chunk_rows}, "
      f"FAISS={ntotal}, processed-with-0-chunks={empty_processed}", flush=True)

# --- A5. sources endpoint validation matrix ---
fid_ok = None
if vs.metadata:
    fid_ok = vs.metadata[0].get("faiss_id", 0)
for bad, expect in [("", 400), ("abc", 400), ("-3", 400),
                    ("1,2,abc", 400), (",".join(["1"] * 51), 400)]:
    r = c.get(f"/sources/?ids={bad}" if bad else "/sources/")
    ok(f"sources ids={bad[:20] or '<empty>'} -> {expect}",
       r.status_code == expect, r.status_code)
r = c.get(f"/sources/?ids=999999")
ok("sources unknown-but-valid id -> 200 with empty sources",
   r.status_code == 200 and r.json().get("sources") == [],
   (r.status_code, r.content[:80]))
if fid_ok is not None:
    r = c.get(f"/sources/?ids={fid_ok}")
    data = r.json() if r.status_code == 200 else {}
    srcs = data.get("sources", [])
    ok(f"sources real faiss id {fid_ok} -> text returned",
       r.status_code == 200 and srcs and srcs[0].get("text"),
       (r.status_code, str(srcs)[:120]))

# =====================================================================
section("PART B - live server HTTP journey")
# =====================================================================

assert os.path.exists(PDF_PATH), PDF_PATH
runserver_env = dict(os.environ)
py = sys.executable
SERVER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_e2e_server.log")
_server_out = open(SERVER_LOG, "w", encoding="utf-8", errors="replace")
SERVER = subprocess.Popen(
    [py, "manage.py", "runserver", f"127.0.0.1:{PORT}", "--noreload"],
    cwd=str(settings.BASE_DIR), env=runserver_env,
    stdout=_server_out, stderr=subprocess.STDOUT)

boot_t0 = time.time()
up = False
while time.time() - boot_t0 < 60:
    try:
        r = requests.get(BASE + "/", timeout=3)
        if r.status_code == 200:
            up = True
            break
    except requests.exceptions.RequestException:
        pass
    time.sleep(0.5)
ok(f"server booted on :{PORT}", up, f"waited {time.time() - boot_t0:.1f}s")

s = requests.Session()
try:
    if up:
        r = s.get(BASE + "/", timeout=30)
        csrf = s.cookies.get("csrftoken", "")
        ok("csrftoken cookie issued", bool(csrf))
        H = {"X-CSRFToken": csrf}

        def new_session():
            """Fresh Session with a new CSRF cookie (avoids keep-alive
            connections poisoned by earlier request timeouts)."""
            _ns = requests.Session()
            _ns.get(BASE + "/", timeout=20)
            _tok = _ns.cookies.get("csrftoken", "")
            return _ns, {"X-CSRFToken": _tok}

        doc_count_baseline = Document.objects.count()
        chunk_baseline = Chunk.objects.count()
        mock_baseline_ids = set(MockTest.objects.values_list("id", flat=True))
        ok("mock baseline captured", isinstance(mock_baseline_ids, set))

        # B1. static + JSON content types over real HTTP
        r = s.get(BASE + "/static/css/style.css", timeout=15)
        ok("static css served (200, >100 bytes)",
           r.status_code == 200 and len(r.content) > 100, r.status_code)
        for path in ("/", "/tutor/", "/practice/", "/mock-test/",
                     "/mock-test/terminated/"):
            r = s.get(BASE + path, timeout=15)
            ok(f"GET {path} -> 200 html, leak-free",
               r.status_code == 200
               and "text/html" in r.headers.get("Content-Type", "")
               and leak_free(r.text) and "</html>" in r.text,
               r.status_code)
        r = s.get(BASE + "/definitely-missing/", timeout=15)
        ok("unknown URL -> 404 custom page",
           r.status_code == 404 and leak_free(r.text), r.status_code)

        # B2. CSRF over real HTTP (no token header)
        r = requests.post(BASE + f"/process/{doc_count_baseline}/",
                          json={}, timeout=15)
        ok("live CSRF: POST without token -> 403",
           r.status_code == 403, r.status_code)

        # B3. upload journey: form POST with real multipart
        with open(PDF_PATH, "rb") as fh:
            files = {"file": ("e2e_30p.pdf", fh, "application/pdf")}
            data = {"title": "E2E Journey Doc",
                    "csrfmiddlewaretoken": csrf}
            r = s.post(BASE + "/", data=data, files=files, timeout=60,
                       headers={"Referer": BASE + "/"},
                       allow_redirects=False)
        doc = Document.objects.filter(title="E2E Journey Doc").first()
        KNOWN_TITLES.add("E2E Journey Doc")
        ok("live upload -> 302 + doc row",
           r.status_code == 302 and doc is not None, r.status_code)

        # B4. missing title -> form re-render, no doc row
        with open(PDF_PATH, "rb") as fh:
            files = {"file": ("e2e_notitle.pdf", fh, "application/pdf")}
            data = {"csrfmiddlewaretoken": csrf}
            r = s.post(BASE + "/", data=data, files=files, timeout=60,
                       headers={"Referer": BASE + "/"},
                       allow_redirects=False)
        ok("upload without title -> 200 form error, no doc row",
           r.status_code == 200
           and not Document.objects.filter(title="").exists(),
           r.status_code)

        if doc:
            # B5. process journey (first call loads the embedding model)
            t0 = time.time()
            r = s.post(BASE + f"/process/{doc.id}/", headers=H, timeout=300)
            sec = time.time() - t0
            data = r.json() if "application/json" in r.headers.get(
                "Content-Type", "") else {}
            ok("live process -> 200 success JSON",
               r.status_code == 200 and data.get("status") == "success"
               and data.get("chunks", 0) > 0, (r.status_code, str(data)[:150]))
            print(f"    [journey] process took {sec:.1f}s, "
                  f"chunks={data.get('chunks')}", flush=True)

            # B6. re-process idempotency over HTTP
            r = s.post(BASE + f"/process/{doc.id}/", headers=H, timeout=120)
            data2 = r.json() if "application/json" in r.headers.get(
                "Content-Type", "") else {}
            ok("live re-process -> already_processed JSON",
               r.status_code == 200
               and data2.get("status") == "already_processed",
               (r.status_code, str(data2)[:120]))

            # B7. listings show the document
            for path in ("/", "/practice/", "/mock-test/"):
                body = s.get(BASE + path, timeout=15).text
                ok(f"list page {path} shows processed doc title",
                   "E2E Journey Doc" in body, path)

            # B8. media file on disk + served
            ok("uploaded file exists on disk with size > 0",
               os.path.exists(doc.file.path) and os.path.getsize(doc.file.path) > 0)
            media_url = doc.file.url
            media_url = media_url if media_url.startswith("http") \
                else BASE + "/" + media_url.lstrip("/")
            r = s.get(media_url, timeout=15)
            if r.status_code == 200 and len(r.content) > 0:
                ok("media file served over HTTP", True)
            else:
                warn("media file not served over HTTP (dev static/media "
                     "serving)", f"{media_url} -> {r.status_code}")

            # B9. sources: chunk text via FAISS id from Chunk rows
            chunk = Chunk.objects.filter(document=doc).first()
            if chunk:
                r = s.get(BASE + f"/sources/?ids={chunk.embedding_id}",
                          timeout=15)
                srcs = r.json().get("sources", []) if r.status_code == 200 else []
                ok("sources lookup returns journey doc chunk",
                   r.status_code == 200 and srcs
                   and srcs[0].get("doc_id") == doc.id
                   and srcs[0].get("text"),
                   (r.status_code, str(srcs)[:150]))

            # B10. retrieval (no LLM): local embedding search.
            # The server owns the on-disk index now; refresh our in-process
            # view (load() re-reads disk) before searching.
            vs.load()
            res = views.rag_chain._retrieve_context(
                "database management systems", doc_ids=[doc.id])
            ok("local retrieval returns journey doc chunks", len(res) >= 1,
               len(res))

            # B11. tutor ask (LLM): 200 answer OR graceful degradation
            try:
                r = s.post(BASE + "/tutor/ask/",
                           json={"question": "What is this study material "
                                             "about? Answer in one sentence."},
                           headers=H, timeout=90)
                if r.status_code == 200 and r.json().get("answer"):
                    ok("tutor ask (LLM) -> 200 with answer", True)
                elif r.status_code in (503, 500) and "application/json" in \
                        r.headers.get("Content-Type", "") and leak_free(r.text):
                    warn("tutor ask degraded (AI service unreachable)",
                         f"{r.status_code}: {r.text[:100]}")
                else:
                    ok("tutor ask -> graceful response", False,
                       (r.status_code, r.text[:120]))
            except requests.exceptions.Timeout:
                warn("tutor ask timed out (AI service slow)", "90s")

            # B12. practice: generate (LLM) then submit (local grading)
            mcqs = None
            try:
                r = s.post(BASE + "/practice/generate/",
                           json={"num_questions": 3, "difficulty": "medium",
                                 "doc_ids": [doc.id]},
                           headers=H, timeout=120)
                if r.status_code == 200 and r.json().get("mcqs"):
                    mcqs = r.json()["mcqs"]
                    ok("practice generate (LLM) -> 3 MCQs",
                       len(mcqs) == 3, len(mcqs))
                else:
                    warn("practice generate degraded (AI service unreachable)",
                         f"{r.status_code}: {r.text[:100]}")
            except requests.exceptions.Timeout:
                warn("practice generate timed out", "120s")
            if mcqs:
                r = s.post(BASE + "/practice/submit/",
                           json={"mcq": mcqs[0], "selected": mcqs[0].get(
                               "correct_answer", "A")},
                           headers=H, timeout=30)
                ok("practice submit -> 200 graded result",
                   r.status_code == 200 and isinstance(r.json(), dict),
                   (r.status_code, r.text[:120]))

            # B13. mock test: create -> terminate branch.
            # create runs in a background thread while we probe whether the
            # server still answers GET / (diagnoses single-worker blocking).
            from concurrent.futures import ThreadPoolExecutor
            ex = ThreadPoolExecutor(max_workers=2)
            t1_id = None
            fut1 = ex.submit(
                s.post, BASE + "/mock-test/",
                json={"num_questions": 10, "difficulty": "easy",
                      "timer_minutes": 10, "doc_ids": [doc.id]},
                headers=H, timeout=360)
            probe = []
            pt0 = time.time()
            while not fut1.done() and time.time() - pt0 < 370:
                time.sleep(6)
                try:
                    probe.append(requests.get(BASE + "/", timeout=8).status_code)
                except requests.exceptions.RequestException as exc:
                    probe.append(type(exc).__name__)
                print(f"{ts()}    [probe during create1] {probe[-1]}",
                      flush=True)
            try:
                r = fut1.result(timeout=10)
                if r.status_code == 200 and r.json().get("redirect"):
                    t1_id = int(re.search(r"/mock-test/(\d+)/",
                                          r.json()["redirect"]).group(1))
                    CREATED_TEST_IDS.append(t1_id)
                    ok("mock test create -> redirect JSON", True, t1_id)
                else:
                    warn("mock test create degraded (AI service unreachable)",
                         f"{r.status_code}: {r.text[:100]}")
            except requests.exceptions.Timeout:
                warn("mock create (terminate flow) timed out", "360s")
            if probe and all(x == 200 for x in probe):
                ok("server answered GET / while mock create ran (threaded)",
                   True, probe)
            elif probe:
                warn("server serialized during slow mock create "
                     "(single worker)", probe[:6])
            if t1_id:
                attempt1 = None
                for _ in range(20):
                    a = TestAttempt.objects.filter(test=t1_id).first()
                    if a:
                        attempt1 = a.id
                        break
                    time.sleep(0.2)
                r = s.post(BASE + "/mock-test/terminate/",
                           json={"test_id": t1_id, "attempt_id": attempt1},
                           headers=H, timeout=30)
                ok("mock terminate -> success JSON",
                   r.status_code == 200 and r.json().get("success") is True,
                   (r.status_code, r.text[:120]))
                r = s.get(BASE + f"/mock-test/{t1_id}/", timeout=15,
                          allow_redirects=True)
                ok("terminated test redirects to terminated page",
                   r.status_code == 200 and "/mock-test/terminated" in r.url,
                   (r.status_code, r.url))

            # B14. mock test: create -> take -> submit -> result pages
            t2_id = None
            try:
                r = s.post(BASE + "/mock-test/",
                           json={"num_questions": 10, "difficulty": "easy",
                                 "timer_minutes": 10, "doc_ids": [doc.id]},
                           headers=H, timeout=360)
                if r.status_code == 200 and r.json().get("redirect"):
                    t2_id = int(re.search(r"/mock-test/(\d+)/",
                                          r.json()["redirect"]).group(1))
                    CREATED_TEST_IDS.append(t2_id)
                    ok("second mock create -> redirect JSON", True, t2_id)
                else:
                    warn("second mock create degraded",
                         f"{r.status_code}: {r.text[:100]}")
            except requests.exceptions.Timeout:
                warn("second mock create timed out", "360s")
            if t2_id:
                try:
                    r = s.get(BASE + f"/mock-test/{t2_id}/", timeout=30)
                    ok("mock take page -> 200",
                       r.status_code == 200 and "</html>" in r.text,
                       r.status_code)
                    qs = list(TestQuestion.objects.filter(test=t2_id))
                    ok("mock questions generated", len(qs) == 10, len(qs))
                    answers = {}
                    for i, q in enumerate(qs):
                        correct = q.correct_answer or "A"
                        if i % 2 == 0:
                            answers[str(q.id)] = correct if correct != "A" \
                                else "B"      # half deliberately wrong
                        else:
                            answers[str(q.id)] = correct
                    r = s.post(BASE + f"/mock-test/{t2_id}/submit/",
                               json={"answers": answers, "time_taken": 7},
                               headers=H, timeout=90)
                    data = r.json() if r.status_code == 200 else {}
                    ok("mock submit -> redirect JSON",
                       r.status_code == 200 and data.get("attempt_id"),
                       (r.status_code, r.text[:120]))
                    aid = data.get("attempt_id")
                    if aid:
                        for page in (f"/mock-test/{aid}/result/",
                                     f"/mock-test/{aid}/analysis/",
                                     f"/mock-test/{aid}/review/"):
                            try:
                                rr = s.get(BASE + page, timeout=60)
                                ok(f"GET {page} -> 200 html",
                                   rr.status_code == 200
                                   and "</html>" in rr.text
                                   and leak_free(rr.text), rr.status_code)
                            except requests.exceptions.RequestException as exc:
                                ok(f"GET {page} -> 200 html", False,
                                   type(exc).__name__)
                except requests.exceptions.RequestException as exc:
                    ok("mock take/submit flow completed", False,
                       type(exc).__name__)

            # B15. concurrency smoke on a FRESH session (rules out poisoned
            # keep-alive connections from earlier timeouts).
            ns, nh = new_session()
            idle = False
            for i in range(24):
                try:
                    if ns.get(BASE + "/", timeout=10).status_code == 200:
                        idle = True
                        break
                except requests.exceptions.RequestException:
                    pass
                print(f"{ts()}    [idle-wait] attempt {i + 1}", flush=True)
                time.sleep(5)
            if idle:
                try:
                    def _probe_get(_):
                        return requests.get(BASE + "/", timeout=30).status_code
                    with ThreadPoolExecutor(max_workers=10) as pool:
                        results = list(pool.map(_probe_get, range(20)))
                    ok("20 parallel GET / all 200",
                       all(x == 200 for x in results), results)
                except Exception as exc:  # noqa: BLE001
                    ok("concurrency smoke completed", False,
                       repr(exc)[:120])
            else:
                warn("server not idle after 240s; skipping concurrency")

            # B16. delete journey + rollback verification (fresh session)
            ns2, nh2 = new_session()
            chunks_before = Chunk.objects.filter(document=doc).count()
            r = ns2.post(BASE + f"/delete/{doc.id}/", headers=nh2, timeout=60)
            ok("live delete -> 200 success JSON",
               r.status_code == 200 and r.json().get("status") == "success",
               (r.status_code, r.text[:120]))
            ok("doc row removed",
               not Document.objects.filter(id=doc.id).exists())
            ok("chunk rows removed",
               not Chunk.objects.filter(document_id=doc.id).exists())
            ok("file removed from disk",
               not os.path.exists(doc.file.path))
            ok("DB counts back to baseline",
               Document.objects.count() == doc_count_baseline
               and Chunk.objects.count() == chunk_baseline,
               (Document.objects.count(), doc_count_baseline,
                Chunk.objects.count(), chunk_baseline))
            body = ""
            for _ in range(3):
                try:
                    body = requests.get(BASE + "/", timeout=20).text
                    break
                except requests.exceptions.RequestException:
                    time.sleep(3)
            ok("title gone from list page",
               bool(body) and "E2E Journey Doc" not in body,
               f"body len={len(body)}")
            try:
                r = ns2.post(BASE + f"/delete/{doc.id}/", headers=nh2,
                             timeout=15)
                ok("second delete -> 404 JSON",
                   r.status_code == 404, r.status_code)
            except requests.exceptions.RequestException as exc:
                ok("second delete -> 404 JSON", False, type(exc).__name__)
            print(f"    [journey] delete rolled back {chunks_before} chunks",
                  flush=True)

except Exception:
    import traceback
    traceback.print_exc()
    ok("live HTTP journey ran without unexpected crash", False)

finally:
    if SERVER is not None:
        SERVER.terminate()
        try:
            SERVER.wait(timeout=15)
        except subprocess.TimeoutExpired:
            SERVER.kill()
            SERVER.wait(timeout=10)
        SERVER = None
    try:
        _server_out.close()
    except Exception:
        pass
    try:
        with open(SERVER_LOG, encoding="utf-8", errors="replace") as fh:
            slog = fh.read()
        slow = [ln for ln in slog.splitlines()
                if re.search(r"\sin (\d{3,})\.\d+s", ln)]
        print(f"\n    [server] log: {SERVER_LOG} "
              f"(tracebacks={slog.count('Traceback')})", flush=True)
        for ln in slow:
            print(f"    [server-slow] {ln.strip()}", flush=True)
        for ln in slog.splitlines()[-25:]:
            print(f"    [server] {ln}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"    [server] log read failed: {exc}", flush=True)

# =====================================================================
section("PART C - embedder memory footprint")
# =====================================================================
try:
    from rag.embedder import Embedder
    before = rss_mb()
    t0 = time.time()
    vec = Embedder().embed(["memory footprint warmup"])
    after = rss_mb()
    ok("embedder produced vectors", vec.shape[1] == 384, vec.shape)
    print(f"    [mem] backend={settings.EMBEDDING_BACKEND} "
          f"RSS {before:.0f} MB -> {after:.0f} MB "
          f"(warmup {time.time() - t0:.1f}s)", flush=True)
    ok("RSS after embedder load < 450 MB (512MB instance budget)",
       0 < after < 450, after)
except Exception as exc:
    import traceback
    traceback.print_exc()
    ok("memory check ran", False, repr(exc))

# =====================================================================
section("PART D - config sanity")
# =====================================================================
ok("GEMINI_API_KEY present (value never printed)",
   bool(os.getenv("GEMINI_API_KEY")))
print(f"    [info] DEBUG={os.getenv('DEBUG')} "
      f"EMBEDDING_BACKEND={settings.EMBEDDING_BACKEND}", flush=True)

# --- final cleanup: any leftover fixture docs (fresh store, server dead) ---
try:
    leftovers = Document.objects.filter(title__in=KNOWN_TITLES)
    if leftovers.exists():
        n_left = leftovers.count()
        from rag.vector_store import VectorStore
        fresh = VectorStore()          # reflects the server's final disk state
        for d in leftovers:
            if d.processed:
                fresh.delete_document(d.id)
            fp = d.file.path if d.file else None
            d.delete()
            if fp and os.path.exists(fp):
                os.remove(fp)
        print(f"    [cleanup] removed {n_left} leftover docs", flush=True)
    MockTest.objects.exclude(id__in=mock_baseline_ids).delete()
    print("[cleanup] E2E fixtures removed", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[cleanup] {exc}", flush=True)

total = PASSED + FAILED
print(f"\n{PASSED}/{total} passed, {WARNED} warnings", flush=True)
sys.exit(1 if FAILED else 0)
