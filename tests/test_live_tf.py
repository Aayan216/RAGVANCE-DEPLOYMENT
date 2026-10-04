import os
import sys

sys.path.insert(0, r"D:\v5\RAG-p2")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django

django.setup()

from django.utils import timezone

from backend.models import Document, MockTest, TestQuestion, TestAttempt, UserAnswer
from backend.views import mock_test_service
from rag.batch_generation import validate_true_false, validate_mcq

# Fixture doc: resolve a live processed document instead of hardcoding an id.
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


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def check_common(qs, n, expected_type=None):
    ok("exact count", len(qs) == n, len(qs))
    ok("correct in A/B/C/D", all(q.correct_answer in ("A", "B", "C", "D") for q in qs))
    ok("explanation non-empty", all(q.explanation and q.explanation.strip() for q in qs))
    ok("question non-empty", all(q.question_text and q.question_text.strip() for q in qs))
    ok("source_chunks non-empty", all(q.source_chunk_ids for q in qs),
       [q.id for q in qs if not q.source_chunk_ids])
    ok("topic non-empty", all(q.topic and q.topic.strip() for q in qs))
    if expected_type:
        ok(f"all type={expected_type}", all(q.question_type == expected_type for q in qs),
           sorted({q.question_type for q in qs}))
    texts = [norm(q.question_text) for q in qs]
    ok("no duplicate questions (normalized)", len(set(texts)) == len(texts),
       [t for t in texts if texts.count(t) > 1])


def make_att(test):
    return TestAttempt.objects.create(
        test=test, started_at=timezone.now(),
        total_questions=test.num_questions, status="active",
    )


# ============ LIVE 1: mcq regression 10q ============
print("=== LIVE 1: mcq 10 ===", flush=True)
t1 = mock_test_service.create_test(num_questions=10, difficulty="medium", timer_minutes=10, doc_ids=[DOC_ID], question_type="mcq")
qs1 = list(TestQuestion.objects.filter(test=t1).order_by("id"))
check_common(qs1, 10, "mcq")
ok("mcq: 4 options filled", all(q.option_a and q.option_b and q.option_c and q.option_d for q in qs1))
ok("mcq: C/D not True/False pair", not (qs1[0].option_c in ("True", "False") and qs1[0].option_d in ("True", "False")))
ok("mcq: validate_mcq passes all", all(validate_mcq({
    "question": q.question_text, "options": {"A": q.option_a, "B": q.option_b, "C": q.option_c, "D": q.option_d},
    "correct": q.correct_answer, "explanation": q.explanation, "topic": q.topic}) for q in qs1))

# ============ LIVE 2: true_false 10q ============
print("=== LIVE 2: true_false 10 ===", flush=True)
t2 = mock_test_service.create_test(num_questions=10, difficulty="medium", timer_minutes=10, doc_ids=[DOC_ID], question_type="true_false")
qs2 = list(TestQuestion.objects.filter(test=t2).order_by("id"))
check_common(qs2, 10, "true_false")
ok("tf: C/D empty", all(q.option_c == "" and q.option_d == "" for q in qs2),
   [(q.id, repr(q.option_c), repr(q.option_d)) for q in qs2 if q.option_c or q.option_d])
ok("tf: options canonical True/False", all(q.option_a == "True" and q.option_b == "False" for q in qs2),
   [(q.id, q.option_a, q.option_b) for q in qs2 if not (q.option_a == "True" and q.option_b == "False")])
ok("tf: correct in A/B", all(q.correct_answer in ("A", "B") for q in qs2))
ok("tf: validate_true_false passes all", all(validate_true_false({
    "question_type": "true_false", "question": q.question_text,
    "options": {"A": q.option_a, "B": q.option_b},
    "correct": q.correct_answer, "explanation": q.explanation, "topic": q.topic}) for q in qs2))
ok("tf: statements differ from MCQ text", not (set(norm(q.question_text) for q in qs1) & set(norm(q.question_text) for q in qs2)))

# ============ LIVE 3: both 10q ============
print("=== LIVE 3: both 10 ===", flush=True)
t3 = mock_test_service.create_test(num_questions=10, difficulty="medium", timer_minutes=10, doc_ids=[DOC_ID], question_type="both")
qs3 = list(TestQuestion.objects.filter(test=t3).order_by("id"))
check_common(qs3, 10)
types = [q.question_type for q in qs3]
tf_pos = [i + 1 for i, x in enumerate(types) if x == "true_false"]
ok("both: exactly 3 tf + 7 mcq", types.count("true_false") == 3 and types.count("mcq") == 7, types)
ok("both: tf at positions [3,6,8]", tf_pos == [3, 6, 8], tf_pos)
ok("both: mcq rows have C/D", all(q.option_c and q.option_d for q in qs3 if q.question_type == "mcq"))
ok("both: tf rows have empty C/D", all(q.option_c == "" and q.option_d == "" for q in qs3 if q.question_type == "true_false"))
ok("both: tf rows canonical True/False", all(q.option_a == "True" and q.option_b == "False" for q in qs3 if q.question_type == "true_false"))
ok("both: tf correct A/B", all(q.correct_answer in ("A", "B") for q in qs3 if q.question_type == "true_false"))
ok("both: mcq correct A-D", all(q.correct_answer in ("A", "B", "C", "D") for q in qs3 if q.question_type == "mcq"))

# ============ LIVE 4: view-level POST (no question_type in payload) ============
print("=== LIVE 4: view POST, forced both ===", flush=True)
from django.test import Client

vc = Client()
r = vc.post("/mock-test/", data=f'{{"num_questions": 10, "difficulty": "medium", "timer_minutes": 30, "doc_ids": [{DOC_ID}]}}', content_type="application/json")
ok("view POST 200 + redirect", r.status_code == 200 and r.json().get("redirect", "").startswith("/mock-test/"), r.content[:200])
v_test_id = int(r.json()["redirect"].strip("/").split("/")[1])
v_types = list(TestQuestion.objects.filter(test_id=v_test_id).order_by("id").values_list("question_type", flat=True))
ok("view POST: 10 stored = 7 mcq + 3 tf", len(v_types) == 10 and v_types.count("mcq") == 7 and v_types.count("true_false") == 3, v_types)
ok("view POST: tf at positions [3,6,8]", [i + 1 for i, x in enumerate(v_types) if x == "true_false"] == [3, 6, 8], v_types)
ok("view POST: attempt auto-created", TestAttempt.objects.filter(test_id=v_test_id).exists())

# stale client field is ignored (cannot force mcq-only)
r = vc.post("/mock-test/", data=f'{{"num_questions": 10, "difficulty": "medium", "timer_minutes": 30, "doc_ids": [{DOC_ID}], "question_type": "mcq"}}', content_type="application/json")
ok("view POST stale question_type ignored -> 200", r.status_code == 200, r.content[:200])
v2_id = int(r.json()["redirect"].strip("/").split("/")[1])
v2_types = list(TestQuestion.objects.filter(test_id=v2_id).order_by("id").values_list("question_type", flat=True))
ok("view POST stale 'mcq' still stored as 7+3", v2_types.count("true_false") == 3 and v2_types.count("mcq") == 7, v2_types)

# ============ end-to-end submit on live both-test ============
print("=== live submit ===", flush=True)
a3 = make_att(t3)
ans = {q.id: q.correct_answer for q in qs3}
att = mock_test_service.submit_attempt(t3.id, ans, 5)
ok("live submit: 10/10", att.score == 10 and att.status == "completed", f"score={att.score}")
res = mock_test_service.get_attempt_result(att.id)
ok("live result: 10 correct", len(res["correct"]) == 10)

for t in (t1, t2, t3):
    TestAttempt.objects.filter(test=t).delete()
    t.delete()
for tid in (v_test_id, v2_id):
    MockTest.objects.filter(id=tid).delete()

print(f"\n{PASSED}/{PASSED + FAILED} passed", flush=True)
sys.exit(1 if FAILED else 0)
