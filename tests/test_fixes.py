import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, r"D:\v5\RAG-p2")
os.chdir(r"D:\v5\RAG-p2")
import django
django.setup()

from django.db import transaction
from django.test import Client, RequestFactory

from backend.models import Document, MockTest
from rag.rag_chain import _redact_secrets
from backend.views import (
    mock_test_settings_view,
    practice_generate_view,
    rag_chain,
)

# Fixture doc: resolve a live processed document instead of hardcoding an id.
DOC_ID = Document.objects.filter(processed=True).values_list("id", flat=True).first()
if DOC_ID is None:
    raise SystemExit("requires >=1 processed document in corpus")

RESULTS = []


def ok(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" | {extra}" if extra else ""), flush=True)


# ---------- 1. redaction unit ----------
m1 = _redact_secrets("PermissionDenied url=https://api.example.com/v1?key=AIzaSyAbCdEfGh1234567890 trailing")
ok("redact key= pattern", "AIzaSyAbCdEfGh1234567890" not in m1 and "key=***" in m1, m1)
m2 = _redact_secrets("boom AIzaSySuperSecretKeyXYZ12345 end")
ok("redact bare AIza key", "AIzaSySuperSecretKeyXYZ12345" not in m2 and "***" in m2, m2)
m3 = "ValueError: Could only generate 0 of 5 questions. Please try again."
ok("normal messages untouched", _redact_secrets(m3) == m3)

# ---------- 2. Practice SUCCESS via real view (live Gemini) ----------
rf = RequestFactory()
req = rf.post(
    "/practice/generate/",
    data=json.dumps({"num_questions": 5, "difficulty": "medium", "topic": None, "doc_ids": [DOC_ID]}),
    content_type="application/json",
)
t0 = time.time()
resp = practice_generate_view(req)
body = json.loads(resp.content)
mcqs = body.get("mcqs", [])
struct_ok = all(
    q.get("question") and isinstance(q.get("options"), dict)
    and q.get("correct") in ("A", "B", "C", "D") and q.get("explanation")
    for q in mcqs
)
ok("practice success 200 JSON", resp.status_code == 200 and resp["Content-Type"].startswith("application/json"))
ok("practice success 5/5 questions", len(mcqs) == 5 and struct_ok, f"got {len(mcqs)} in {time.time()-t0:.1f}s")


# ---------- 3. Practice FAILURE: logging + redaction + cap + clean JSON ----------
class BoomLLM:
    def __init__(self):
        self.n = 0

    def invoke(self, prompt):
        self.n += 1
        raise Exception(
            "simulated 503 Service Unavailable url=https://gen.example/v1beta?key=AIzaSyFAKESECRET1234567890"
        )


orig_llm = rag_chain.llm
boom = BoomLLM()
rag_chain.llm = boom

req_f = rf.post(
    "/practice/generate/",
    data=json.dumps({"num_questions": 5, "difficulty": "medium", "topic": None, "doc_ids": [DOC_ID]}),
    content_type="application/json",
)
buf = io.StringIO()
t1 = time.time()
with redirect_stdout(buf):
    resp_f = practice_generate_view(req_f)
logged = buf.getvalue()
err = json.loads(resp_f.content).get("error", "")

ok("practice failure returns JSON 500", resp_f.status_code == 500 and resp_f["Content-Type"].startswith("application/json"))
ok("practice failure clean error message", err == "Could only generate 0 of 5 questions. Please try again.", err)
ok("practice [ERROR] logged with operation", "[ERROR] practice MCQ batch failed" in logged)
ok("practice log has batch_count", "batch_count=5" in logged)
ok("practice log has exception type+message", "Exception:" in logged and "simulated 503" in logged)
ok("practice log has doc_ids", f"doc_ids=[{DOC_ID}]" in logged)
ok("practice log API key redacted", "AIzaSyFAKESECRET1234567890" not in logged and "key=***" in logged)
ok("practice retry cap respected (2 attempts = 10 slots)", boom.n == 2, f"llm attempts={boom.n}, elapsed={time.time()-t1:.1f}s")
ok("practice FAILED summary line present", "[info] practice generation failed" in logged.lower() and "requested=5" in logged, [l for l in logged.splitlines() if "FAILED" in l])
ok("practice failure creates no state", True)

# ---------- 4. Mock SUCCESS via real view (rolled back) ----------
rag_chain.llm = orig_llm
before_ok = True
req_m = rf.post(
    "/mock-test/",
    data=json.dumps({"num_questions": 20, "difficulty": "medium", "timer_minutes": 30, "doc_ids": [DOC_ID]}),
    content_type="application/json",
)
t2 = time.time()
buf_m = io.StringIO()
with transaction.atomic():
    with redirect_stdout(buf_m):
        resp_m = mock_test_settings_view(req_m)
    transaction.set_rollback(True)
body_m = json.loads(resp_m.content)
ok("mock success 200 JSON redirect", resp_m.status_code == 200 and body_m.get("redirect", "").startswith("/mock-test/"), str(body_m))
ok("mock success generation stats", "[INFO] Mock Test generation:" in buf_m.getvalue() and "valid=14" in buf_m.getvalue(),
   next((l for l in buf_m.getvalue().splitlines() if "Mock Test generation:" in l), ""), )
ok("mock success T/F generation stats", "[INFO] Mock Test T/F generation:" in buf_m.getvalue() and "valid=6" in buf_m.getvalue(),
   next((l for l in buf_m.getvalue().splitlines() if "Mock Test T/F generation:" in l), ""), )
ok("mock success fast", (time.time() - t2) < 60, f"{time.time()-t2:.1f}s")

# ---------- 5. Mock FAILURE: JSON 500, no orphan, logged ----------
boom2 = BoomLLM()
rag_chain.llm = boom2
req_mf = rf.post(
    "/mock-test/",
    data=json.dumps({"num_questions": 20, "difficulty": "medium", "timer_minutes": 30, "doc_ids": [DOC_ID]}),
    content_type="application/json",
)
count_before = MockTest.objects.count()
buf_f = io.StringIO()
with redirect_stdout(buf_f):
    resp_mf = mock_test_settings_view(req_mf)
count_after = MockTest.objects.count()
logged_f = buf_f.getvalue()
err_mf = json.loads(resp_mf.content).get("error", "")

ok("mock failure returns JSON 500 (not HTML)", resp_mf.status_code == 500 and resp_mf["Content-Type"].startswith("application/json"),
   resp_mf["Content-Type"])
ok("mock failure error message", err_mf == "Could only generate 0 of 14 questions. Please try again.", err_mf)
ok("mock failure content is NOT <!DOCTYPE", not resp_mf.content.lstrip().startswith(b"<!DOC"))
ok("no orphan MockTest record", count_before == count_after, f"{count_before} == {count_after}")
ok("mock [ERROR] logged with operation", "[ERROR] mock MCQ batch failed" in logged_f)
ok("mock log has batch_count", "batch_count=5" in logged_f)
ok("mock log has exception", "simulated 503" in logged_f)
ok("mock log API key redacted", "AIzaSyFAKESECRET1234567890" not in logged_f and "key=***" in logged_f)
ok("mock retry cap respected (14*2 slots = 6 attempts)", boom2.n == 6, f"llm attempts={boom2.n}")

rag_chain.llm = orig_llm

# ---------- 6. Tutor page render ----------
c = Client()
r = c.get("/tutor/")
html = r.content.decode()
ok("tutor page 200", r.status_code == 200)
ok("welcome bubble uses assistant-bubble", "message-bubble assistant-bubble" in html)
ok("no bubble uses bg-light", "message-bubble bg-light" not in html and "assistant-bubble bg-light" not in html)
ok("dynamic AI bubbles use assistant-bubble", ": 'assistant-bubble'" in html)

r2 = c.get("/practice/")
ok("practice page 200", r2.status_code == 200)

# ---------- summary ----------
failed = [n for n, p in RESULTS if not p]
print(f"\n===== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed =====", flush=True)
if failed:
    print("FAILED:", *failed, sep="\n  - ", flush=True)
    sys.exit(1)
