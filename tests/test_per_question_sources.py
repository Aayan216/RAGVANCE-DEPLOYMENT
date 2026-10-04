import json
import os
import re
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, r"D:\v5\RAG-p2")
os.chdir(r"D:\v5\RAG-p2")
import django
django.setup()

from backend.models import Document
from backend.views import rag_chain

# Fixture doc: resolve a live processed document instead of hardcoding an id.
DOC_ID = Document.objects.filter(processed=True).values_list("id", flat=True).first()
if DOC_ID is None:
    raise SystemExit("requires >=1 processed document in corpus")

RESULTS = []


def ok(name, cond, extra=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" | {extra}" if extra else ""), flush=True)


def make_question(i, source_chunks=None):
    q = {
        "question": f"Question {i}?",
        "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "correct": "A",
        "explanation": "Because.",
        "topic": f"T{i}",
    }
    if source_chunks is not None:
        q["source_chunks"] = source_chunks
    return q


class StubResponse:
    def __init__(self, content):
        self.content = content


class StubLLM:
    def __init__(self, questions):
        self.questions = questions
        self.n = 0

    def invoke(self, prompt):
        self.n += 1
        return StubResponse(json.dumps({"questions": self.questions}))


results = rag_chain._retrieve_context("key concepts", doc_ids=[DOC_ID])
allowed = [r["faiss_id"] for r in results]
ok("retrieval yields >=3 chunks", len(allowed) >= 3, str(allowed))
a, b, cc = allowed[0], allowed[1], allowed[2]

orig_llm = rag_chain.llm

# ---------- 1. per-question distinct sets (B3 success metric) ----------
stub = StubLLM([make_question(1, [a]), make_question(2, [b, cc])])
rag_chain.llm = stub
try:
    out = rag_chain.generate_mcq_batch(topic="key concepts", doc_ids=[DOC_ID], count=2)
finally:
    rag_chain.llm = orig_llm
ok("batch returns 2 questions", len(out) == 2, str(len(out)))
ok("per-question sets preserved",
   out[0]["source_chunks"] == [a] and out[1]["source_chunks"] == [b, cc],
   str([q["source_chunks"] for q in out]))
ok("per-question sets differ (B3 metric)", out[0]["source_chunks"] != out[1]["source_chunks"])
ok("batch called the model once", stub.n == 1, f"n={stub.n}")

# ---------- 2. foreign ids dropped, strings coerced, order kept ----------
stub2 = StubLLM([make_question(1, [cc, 999999, str(a), "abc", True])])
rag_chain.llm = stub2
try:
    out2 = rag_chain.generate_mcq_batch(topic="key concepts", doc_ids=[DOC_ID], count=1)
finally:
    rag_chain.llm = orig_llm
ok("foreign/non-int ids dropped, strings coerced, order kept",
   out2[0]["source_chunks"] == [cc, a], str(out2[0]["source_chunks"]))

# ---------- 3. fallback: missing / invalid -> full retrieval batch ----------
stub3 = StubLLM([make_question(1), make_question(2, ["abc"])])
rag_chain.llm = stub3
try:
    out3 = rag_chain.generate_mcq_batch(topic="key concepts", doc_ids=[DOC_ID], count=2)
finally:
    rag_chain.llm = orig_llm
ok("missing source_chunks -> full retrieval fallback",
   out3[0]["source_chunks"] == allowed, str(out3[0]["source_chunks"]))
ok("all-invalid source_chunks -> full retrieval fallback",
   out3[1]["source_chunks"] == allowed, str(out3[1]["source_chunks"]))

# ---------- 4. mock batch: cite ids parsed from the prompt's own labels ----------
label_map = {(m["doc_id"], m["chunk_index"]): m["faiss_id"]
             for m in rag_chain.vector_store.metadata}


class PromptLabelStub:
    """Mimics the real model: cites chunk ids from the [doc_N:chunk_M] labels it sees."""

    def __init__(self, question_type=None):
        self.question_type = question_type
        self.n = 0
        self.seen_fids = []

    def invoke(self, prompt):
        self.n += 1
        pairs = re.findall(r"\[doc_(\d+):chunk_(\d+)\]", prompt)
        fids = []
        for doc_s, chunk_s in pairs:
            fid = label_map.get((int(doc_s), int(chunk_s)))
            if fid is not None and fid not in fids:
                fids.append(fid)
        self.seen_fids = fids
        q1 = make_question(1, fids[0:1])
        q2 = make_question(2, fids[1:2] or fids[0:1])
        if self.question_type:
            q1 = dict(q1, question_type=self.question_type)
            q2 = dict(q2, question_type=self.question_type)
            if self.question_type == "true_false":
                q1["options"] = {"A": "True", "B": "False"}
                q2["options"] = {"A": "True", "B": "False"}
                return StubResponse(json.dumps({"questions": [q1]}))
        return StubResponse(json.dumps({"questions": [q1, q2]}))


stub4 = PromptLabelStub()
rag_chain.llm = stub4
try:
    out4 = rag_chain.generate_mock_questions_batch(doc_ids=[DOC_ID], count=2)
finally:
    rag_chain.llm = orig_llm
ok("mock: prompt carried context labels", len(stub4.seen_fids) >= 2, str(stub4.seen_fids))
ok("mock: per-question sets non-empty and distinct",
   all(q["source_chunks"] for q in out4) and out4[0]["source_chunks"] != out4[1]["source_chunks"],
   str([q["source_chunks"] for q in out4]))
ok("mock: question_type stamped", all(q.get("question_type") == "mcq" for q in out4))

# ---------- 5. T/F batch: same behavior ----------
stub5 = PromptLabelStub(question_type="true_false")
rag_chain.llm = stub5
try:
    out5 = rag_chain.generate_mock_questions_batch(doc_ids=[DOC_ID], count=1, question_type="true_false")
finally:
    rag_chain.llm = orig_llm
ok("T/F: per-question set preserved from prompt labels",
   len(out5) == 1 and out5[0]["source_chunks"] and
   set(out5[0]["source_chunks"]).issubset(set(stub5.seen_fids)),
   str(out5[0]["source_chunks"]))
ok("T/F: question_type stamped", out5[0].get("question_type") == "true_false")

# ---------- 6. _attach_sources unit: bools, duplicates, non-list ----------
attach = rag_chain._attach_sources
fake_results = [{"faiss_id": 10}, {"faiss_id": 11}]
q = {"source_chunks": [11, 11, True, "10", None]}
attach([q], fake_results)
ok("unit: dedupe + bool rejected + str coerced", q["source_chunks"] == [11, 10], str(q["source_chunks"]))

q2 = {"source_chunks": 42}
attach([q2], fake_results)
ok("unit: non-list falls back", q2["source_chunks"] == [10, 11], str(q2["source_chunks"]))

q3 = {}
attach([q3], fake_results)
ok("unit: absent key falls back", q3["source_chunks"] == [10, 11], str(q3["source_chunks"]))

ok("unit: never empty with retrieval results", all(
    isinstance(qx.get("source_chunks"), list) and qx["source_chunks"] for qx in (q, q2, q3)))

# ---------- summary ----------
failed = [n for n, p in RESULTS if not p]
print(f"\n===== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed =====", flush=True)
if failed:
    print("FAILED:", *failed, sep="\n  - ", flush=True)
    sys.exit(1)
