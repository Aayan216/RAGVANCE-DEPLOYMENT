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

import httpx
from django.conf import settings
from django.test import RequestFactory, override_settings

from backend.models import Document
from backend.views import practice_generate_view, tutor_ask_view, rag_chain

# Fixture doc: resolve a live processed document instead of hardcoding an id.
DOC_ID = Document.objects.filter(processed=True).values_list("id", flat=True).first()
if DOC_ID is None:
    raise SystemExit("requires >=1 processed document in corpus")
from google.genai.errors import APIError, ServerError
from langchain_google_genai.chat_models import (
    GoogleAPIError,
    GoogleInvalidRequestError,
    GoogleRateLimitError,
)
from langchain_core.exceptions import ModelRateLimitError, ModelTimeoutError
from rag.llm_resilience import is_transient_llm_error, invoke_with_retry

RESULTS = []


def ok(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" | {extra}" if extra else ""), flush=True)


class Recorder:
    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


class FlakyLLM:
    def __init__(self, failures, exc):
        self.n = 0
        self.failures = failures
        self.exc = exc

    def invoke(self, prompt):
        self.n += 1
        if self.n <= self.failures:
            raise self.exc
        return "ok"


# ---------- 1. classifier: type/attribute based, never message based ----------
ok("generic Exception with 503 text is NOT transient",
   not is_transient_llm_error(Exception("simulated 503 Service Unavailable")))
ok("APIError 503 is transient", is_transient_llm_error(APIError(503, None)))
ok("APIError 429 is transient", is_transient_llm_error(APIError(429, None)))
ok("APIError 500 is transient", is_transient_llm_error(APIError(500, None)))
ok("APIError 400 is not transient", not is_transient_llm_error(APIError(400, None)))
ok("APIError 404 is not transient", not is_transient_llm_error(APIError(404, None)))
ok("ConnectionError is transient", is_transient_llm_error(ConnectionError("boom")))
ok("TimeoutError is transient", is_transient_llm_error(TimeoutError("slow")))
ok("httpx.TimeoutException is transient", is_transient_llm_error(httpx.TimeoutException("slow")))
ok("httpx.ConnectError is transient", is_transient_llm_error(httpx.ConnectError("nope")))
ok("ValueError is not transient", not is_transient_llm_error(ValueError("x")))
ok("None is not transient", not is_transient_llm_error(None))


class WithStatus:
    status = 503


class WithStringCode:
    code = "503"


api_core_cls = type("ServiceUnavailable", (Exception,), {"__module__": "google.api_core.exceptions"})
ok("attr status=503 is transient", is_transient_llm_error(WithStatus()))
ok("string code '503' is not transient", not is_transient_llm_error(WithStringCode()))
ok("google.api_core ServiceUnavailable name is transient", is_transient_llm_error(api_core_cls("down")))
ok("langchain GoogleRateLimitError is transient", is_transient_llm_error(GoogleRateLimitError("429")))
ok("langchain ModelRateLimitError is transient", is_transient_llm_error(ModelRateLimitError("429")))
ok("langchain ModelTimeoutError is transient", is_transient_llm_error(ModelTimeoutError("slow")))
ok("langchain GoogleInvalidRequestError is not transient",
   not is_transient_llm_error(GoogleInvalidRequestError("bad")))
ok("langchain GoogleAPIError 503 is transient", is_transient_llm_error(GoogleAPIError(503, None)))
ok("langchain GoogleAPIError 500 is transient", is_transient_llm_error(GoogleAPIError(500, None)))
ok("google.genai ServerError 503 is transient", is_transient_llm_error(ServerError(503, None)))


# ---------- 2. invoke_with_retry behavior ----------
rec = Recorder()
llm = FlakyLLM(2, APIError(503, None))
out = invoke_with_retry(llm, "p", sleep=rec, max_attempts=3, base_delay=0.5)
ok("transient fail x2 then success returns result", out == "ok" and llm.n == 3, f"n={llm.n}")
ok("exponential backoff delays", rec.delays == [0.5, 1.0], str(rec.delays))

rec2 = Recorder()
llm2 = FlakyLLM(99, APIError(503, None))
raised = None
try:
    invoke_with_retry(llm2, "p", sleep=rec2, max_attempts=3, base_delay=1.0)
except Exception as exc:
    raised = exc
ok("exhaustion re-raises original exception", isinstance(raised, APIError) and raised.code == 503, str(raised))
ok("exhaustion used all attempts", llm2.n == 3, f"n={llm2.n}")
ok("exhaustion slept twice", rec2.delays == [1.0, 2.0], str(rec2.delays))

rec3 = Recorder()
llm3 = FlakyLLM(99, Exception("simulated 503 Service Unavailable"))
raised3 = None
try:
    invoke_with_retry(llm3, "p", sleep=rec3, max_attempts=5, base_delay=1.0)
except Exception as exc:
    raised3 = exc
ok("generic exception never retried", llm3.n == 1 and rec3.delays == [], f"n={llm3.n} delays={rec3.delays}")
ok("generic exception message preserved", isinstance(raised3, Exception) and "simulated 503" in str(raised3))

rec4 = Recorder()
llm4 = FlakyLLM(99, APIError(429, None))
raised4 = None
try:
    invoke_with_retry(llm4, "p", sleep=rec4, max_attempts=1, base_delay=1.0)
except Exception as exc:
    raised4 = exc
ok("max_attempts=1 means single attempt", llm4.n == 1 and isinstance(raised4, APIError) and rec4.delays == [])

rec5 = Recorder()
llm5 = FlakyLLM(99, ValueError("bad"))
raised5 = None
try:
    invoke_with_retry(llm5, "p", sleep=rec5, max_attempts=4, base_delay=1.0)
except Exception as exc:
    raised5 = exc
ok("non-transient typed error not retried", llm5.n == 1 and isinstance(raised5, ValueError) and rec5.delays == [])

ok("settings GEMINI_MAX_RETRIES default", settings.GEMINI_MAX_RETRIES == 2, str(settings.GEMINI_MAX_RETRIES))
ok("settings GEMINI_RETRY_BASE_DELAY default", settings.GEMINI_RETRY_BASE_DELAY == 1.0, str(settings.GEMINI_RETRY_BASE_DELAY))

with override_settings(GEMINI_MAX_RETRIES=1):
    rec6 = Recorder()
    llm6 = FlakyLLM(1, APIError(503, None))
    out6 = invoke_with_retry(llm6, "p", sleep=rec6)
    ok("settings-driven attempt count (1 retry = 2 attempts)", out6 == "ok" and llm6.n == 2, f"n={llm6.n}")


# ---------- 3. practice end-to-end: typed transient -> exact 500 + [ERROR] ----------
class TransientGeminiError(Exception):
    def __init__(self, message="simulated quota exceeded"):
        super().__init__(message)
        self.code = 503


class TypedBoomLLM:
    def __init__(self):
        self.n = 0

    def invoke(self, prompt):
        self.n += 1
        raise TransientGeminiError()


rf = RequestFactory()
orig_llm = rag_chain.llm
typed = TypedBoomLLM()
rag_chain.llm = typed

req = rf.post(
    "/practice/generate/",
    data=json.dumps({"num_questions": 5, "difficulty": "medium", "topic": None, "doc_ids": [DOC_ID]}),
    content_type="application/json",
)
buf = io.StringIO()
t0 = time.time()
with override_settings(GEMINI_RETRY_BASE_DELAY=0):
    with redirect_stdout(buf):
        resp = practice_generate_view(req)
logged = buf.getvalue()
err = json.loads(resp.content).get("error", "")
rag_chain.llm = orig_llm

ok("typed transient practice returns JSON 500",
   resp.status_code == 500 and resp["Content-Type"].startswith("application/json"))
ok("typed transient exact failure message",
   err == "Could only generate 0 of 5 questions. Please try again.", err)
ok("typed transient [ERROR] logged", "[ERROR] practice MCQ batch failed" in logged)
ok("typed transient log has exception type+message",
   "TransientGeminiError:" in logged and "simulated quota exceeded" in logged)
ok("typed transient FAILED summary unchanged",
   "llm_calls=2" in logged and "valid=0" in logged and "retries=1" in logged,
   [l for l in logged.splitlines() if "FAILED" in l])
ok("typed transient attempt count (2 calls x 3 attempts)", typed.n == 6, f"n={typed.n}")
ok("typed transient run fast", (time.time() - t0) < 30, f"{time.time()-t0:.1f}s")


# ---------- 4. tutor: transient -> 503, generic -> 500 ----------
class TutorTransientLLM:
    def __init__(self):
        self.n = 0

    def invoke(self, prompt):
        self.n += 1
        raise TransientGeminiError()


class TutorGenericLLM:
    def __init__(self):
        self.n = 0

    def invoke(self, prompt):
        self.n += 1
        raise Exception("tutor boom")


req_t = rf.post("/tutor/ask/", data=json.dumps({"question": "What is ATP?"}),
                content_type="application/json")
rag_chain.llm = TutorTransientLLM()
with override_settings(GEMINI_RETRY_BASE_DELAY=0):
    resp_t = tutor_ask_view(req_t)
body_t = json.loads(resp_t.content)
ok("tutor transient returns 503", resp_t.status_code == 503, str(resp_t.status_code))
ok("tutor transient JSON error body",
   resp_t["Content-Type"].startswith("application/json") and "error" in body_t, str(body_t))

rag_chain.llm = TutorGenericLLM()
resp_g = tutor_ask_view(req_t)
body_g = json.loads(resp_g.content)
ok("tutor generic returns 500 with original message",
   resp_g.status_code == 500 and body_g.get("error") == "tutor boom", str(body_g))

rag_chain.llm = orig_llm

# ---------- summary ----------
failed = [n for n, p in RESULTS if not p]
print(f"\n===== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed =====", flush=True)
if failed:
    print("FAILED:", *failed, sep="\n  - ", flush=True)
    sys.exit(1)
