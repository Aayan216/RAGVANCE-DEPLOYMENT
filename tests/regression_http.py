import os
import sys
import json

sys.path.insert(0, r"D:\v5\RAG-p2")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django

django.setup()

from django.test import Client
from django.utils import timezone

from backend import views
from backend.models import Document, MockTest, TestAttempt, TestQuestion, UserAnswer
from mock_test import MockTestService

# Fixture doc: resolve a live processed document instead of hardcoding an id
# (the corpus is not guaranteed to contain any specific id).
DOC_ID = Document.objects.filter(processed=True).values_list("id", flat=True).first()
if DOC_ID is None:
    raise SystemExit("requires >=1 processed document in corpus")

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


class FakeChain:
    def __init__(self):
        self.counters = {"mcq": 0, "true_false": 0}

    def generate_mock_questions_batch(self, difficulty="medium", doc_ids=None, count=5, question_type="mcq"):
        out = []
        for _ in range(count):
            i = self.counters[question_type]
            self.counters[question_type] += 1
            if question_type == "true_false":
                out.append({
                    "question_type": "true_false", "question": f"TF statement {i}.",
                    "options": {"A": "True", "B": "False"}, "correct": "A" if i % 2 == 0 else "B",
                    "explanation": "Because.", "topic": "Networking",
                })
            else:
                out.append({
                    "question": f"MCQ question {i}?",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct": "ABCD"[i % 4], "explanation": "Because.", "topic": "Networking",
                })
        return out


c = Client()

# ============================================================
# 1. Tutor (LIVE - real Gemini + RAG)
# ============================================================
print("--- tutor live ---", flush=True)
r = c.post("/tutor/ask/", data=json.dumps({"question": "What is a database management system?"}),
           content_type="application/json")
ok("tutor ask -> 200", r.status_code == 200, r.status_code)
data = r.json()
if r.status_code == 200:
    ok("tutor answer non-empty", isinstance(data.get("answer"), str) and len(data.get("answer", "")) > 20,
       str(data)[:200])
    srcs = data.get("sources", [])
    ok("tutor sources present", isinstance(srcs, list) and len(srcs) >= 1, len(srcs) if isinstance(srcs, list) else type(srcs))
    if srcs:
        s = srcs[0]
        ok("source keys present", all(k in s for k in ("doc_id", "chunk_index", "file_name", "text")), list(s.keys()))
        ok("source text non-empty", bool(s.get("text")))

r = c.post("/tutor/ask/", data=json.dumps({"question": "   "}), content_type="application/json")
ok("tutor empty question -> 400", r.status_code == 400 and "required" in r.json().get("error", ""), r.status_code)
ok("tutor GET -> 405", c.get("/tutor/ask/").status_code == 405)

# ============================================================
# 2. Practice scoring (no API cost)
# ============================================================
print("--- practice scoring ---", flush=True)
mcq = {"question": "2+2?", "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
       "correct": "B", "explanation": "Basic arithmetic.", "topic": "Math"}
r = c.post("/practice/submit/", data=json.dumps({"mcq": mcq, "selected": "B"}), content_type="application/json")
d = r.json()
ok("practice correct answer -> is_correct True", r.status_code == 200 and d.get("is_correct") is True, d)
ok("practice returns correct_answer + explanation", d.get("correct_answer") == "B" and d.get("explanation") == "Basic arithmetic.")

r = c.post("/practice/submit/", data=json.dumps({"mcq": mcq, "selected": "A"}), content_type="application/json")
ok("practice wrong answer -> is_correct False", r.json().get("is_correct") is False)

r = c.post("/practice/submit/", data=json.dumps({"mcq": mcq, "selected": ""}), content_type="application/json")
ok("practice empty answer -> is_correct False", r.json().get("is_correct") is False)

# ============================================================
# 3. Mock Test submit flow (HTTP) + state machine
# ============================================================
print("--- mock submit FSM ---", flush=True)
orig_service = views.mock_test_service
views.mock_test_service = MockTestService(rag_chain=FakeChain())
created = []
try:
    # create test 1 (submit flow)
    r = c.post("/mock-test/", data=json.dumps({"num_questions": 10, "difficulty": "medium",
                                               "timer_minutes": 30, "doc_ids": [DOC_ID]}),
               content_type="application/json")
    ok("create test -> 200 redirect", r.status_code == 200 and r.json().get("redirect", "").startswith("/mock-test/"), r.content[:150])
    tid1 = int(r.json()["redirect"].strip("/").split("/")[1])
    created.append(tid1)
    att1 = TestAttempt.objects.filter(test_id=tid1).first()
    ok("attempt auto-created active", att1 is not None and att1.status == "active")

    qs = list(TestQuestion.objects.filter(test_id=tid1))
    answers = {str(q.id): q.correct_answer for q in qs}  # all correct, JSON string keys
    r = c.post(f"/mock-test/{tid1}/submit/",
               data=json.dumps({"answers": answers, "time_taken": 42}), content_type="application/json")
    ok("submit -> 200 + result redirect", r.status_code == 200 and r.json().get("redirect") == f"/mock-test/{att1.id}/result/", r.content[:200])
    att1.refresh_from_db()
    ok("attempt completed score 10/10", att1.status == "completed" and att1.score == 10, (att1.status, att1.score))
    ok("percentage 100 + time recorded", att1.percentage == 100.0 and att1.time_taken_seconds == 42, (att1.percentage, att1.time_taken_seconds))
    ok("test status completed", MockTest.objects.get(id=tid1).status == "completed")

    rr = c.get(f"/mock-test/{att1.id}/result/")
    ok("result page 200 + shows 100%", rr.status_code == 200 and "100" in rr.content.decode())

    r = c.post(f"/mock-test/{tid1}/submit/", data=json.dumps({"answers": {}, "time_taken": 1}),
               content_type="application/json")
    ok("double submit -> 400", r.status_code == 400 and "already" in r.json().get("error", ""), r.content[:150])

    r = c.get(f"/mock-test/{tid1}/")
    ok("take view on completed -> redirect to result", r.status_code == 302 and f"/mock-test/{att1.id}/result/" in r.url, getattr(r, "url", r.status_code))

    # create test 2 (terminate flow)
    r = c.post("/mock-test/", data=json.dumps({"num_questions": 10, "difficulty": "easy",
                                               "timer_minutes": 10, "doc_ids": [DOC_ID]}),
               content_type="application/json")
    tid2 = int(r.json()["redirect"].strip("/").split("/")[1])
    created.append(tid2)
    att2 = TestAttempt.objects.filter(test_id=tid2).first()

    r = c.post("/mock-test/terminate/", data=json.dumps({"test_id": tid2, "attempt_id": att2.id}),
               content_type="application/json")
    ok("terminate -> success + redirect", r.status_code == 200 and r.json().get("success") is True
       and r.json().get("redirect") == "/mock-test/terminated/", r.content[:150])
    att2.refresh_from_db()
    ok("attempt terminated", att2.status == "terminated")
    ok("test terminated", MockTest.objects.get(id=tid2).status == "terminated")

    rt = c.get("/mock-test/terminated/")
    ok("terminated page 200", rt.status_code == 200)

    r = c.post(f"/mock-test/{tid2}/submit/", data=json.dumps({"answers": {}, "time_taken": 1}),
               content_type="application/json")
    ok("submit after terminate -> 400", r.status_code == 400 and "terminated" in r.json().get("error", ""), r.content[:150])

    r = c.get(f"/mock-test/{tid2}/")
    ok("take view on terminated -> redirect to terminated page", r.status_code == 302 and r.url == "/mock-test/terminated/", getattr(r, "url", r.status_code))

    # error paths
    r = c.post("/mock-test/terminate/", data=json.dumps({}), content_type="application/json")
    ok("terminate missing ids -> 400", r.status_code == 400 and "Missing" in r.json().get("error", ""))

    r = c.post("/mock-test/terminate/", data=json.dumps({"test_id": 999999, "attempt_id": 999998}), content_type="application/json")
    ok("terminate bogus ids -> 500 JSON (no HTML)", r.status_code == 500 and r["Content-Type"].startswith("application/json") and not r.content.lstrip().startswith(b"<!DOC"), (r.status_code, r["Content-Type"]))

    r = c.post(f"/mock-test/{tid1}/submit/", data="not-json{", content_type="application/json")
    ok("submit malformed JSON -> JSON error (no HTML)", r.status_code in (400, 500) and r["Content-Type"].startswith("application/json") and not r.content.lstrip().startswith(b"<!DOC"), (r.status_code, r["Content-Type"]))

    r = c.post("/mock-test/999999/submit/", data=json.dumps({"answers": {}, "time_taken": 0}), content_type="application/json")
    ok("submit unknown test -> JSON error (no HTML)", r.status_code in (400, 500) and r["Content-Type"].startswith("application/json") and not r.content.lstrip().startswith(b"<!DOC"), (r.status_code, r["Content-Type"]))

finally:
    views.mock_test_service = orig_service
    for tid in created:
        MockTest.objects.filter(id=tid).delete()

print(f"\n{PASSED}/{PASSED + FAILED} passed", flush=True)
sys.exit(1 if FAILED else 0)
