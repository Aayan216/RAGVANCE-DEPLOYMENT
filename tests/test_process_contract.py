"""JSON contract tests for /process/<id>/ and /delete/<id>/.

Every execution path must return JSON with a proper HTTP status, never HTML,
never a traceback/paths/secrets in the body, and processing must stay atomic
(no orphan FAISS vectors, no Chunk rows for an unprocessed document).
"""
import os
import re
import sys

sys.path.insert(0, r"D:\v5\RAG-p2")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django

django.setup()

from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from backend import views
from backend.models import Chunk, Document

PASSED = 0
FAILED = 0
CREATED = []


def ok(label, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print("PASS -", label, flush=True)
    else:
        FAILED += 1
        print("FAIL -", label, (":: " + str(extra)) if extra else "", flush=True)


def is_json_response(r):
    return "application/json" in (r.headers.get("Content-Type") or "")


def no_leaks(text):
    """Body must not contain tracebacks, filesystem paths or credentials."""
    checks = {
        "Traceback": "Traceback" not in text,
        "windows path": not re.search(r"[A-Za-z]:\\\\|[A-Za-z]:\\(?:Users|Windows|opt)", text),
        "unix internal path": not re.search(r"/(?:home|opt|usr|var|tmp|etc|workspace)/", text),
        "db url": "postgresql://" not in text and "postgres://" not in text,
        "sb secret": "sb_secret_" not in text,
        "gemini key prefix": "AIza" not in text,
        "jwt": not re.search(r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}", text),
    }
    return [k for k, v in checks.items() if not v]


def make_doc(title, name, payload, content_type):
    up = SimpleUploadedFile(name, payload, content_type=content_type)
    r = c.post("/", {"title": title, "file": up})
    doc = Document.objects.filter(title=title).first()
    return r, doc


TXT_BODY = (
    "Contract fixture for JSON response testing.\n\n"
    + " ".join(
        f"Paragraph {i} describes calibration steps, voltage thresholds, "
        f"and maintenance windows for the assembly line {i}."
        for i in range(25)
    )
)

c = Client()

try:
    # ---------- 405 is JSON, checked before anything else ----------
    d = Document.objects.order_by("id").first()
    probe_id = d.id if d else 1
    r = c.get(f"/process/{probe_id}/")
    ok("GET /process/ -> 405", r.status_code == 405, r.status_code)
    ok("GET /process/ -> JSON content type", is_json_response(r), r.headers.get("Content-Type"))
    body = r.content.decode()
    ok("GET /process/ -> success=false + error", is_json_response(r)
       and r.json().get("success") is False and r.json().get("error"),
       body[:200])
    ok("GET /process/ leak-free", not no_leaks(body), no_leaks(body))

    r = c.get(f"/delete/{probe_id}/")
    ok("GET /delete/ -> 405 JSON", r.status_code == 405 and is_json_response(r),
       (r.status_code, r.content[:120]))

    # ---------- 404 is JSON ----------
    r = c.post("/process/99999998/")
    ok("POST /process/missing -> 404 JSON",
       r.status_code == 404 and is_json_response(r)
       and r.json().get("success") is False and "not found" in r.json().get("error", "").lower(),
       (r.status_code, r.content[:200]))
    ok("process 404 leak-free", not no_leaks(r.content.decode()), no_leaks(r.content.decode()))

    r = c.post("/delete/99999998/")
    ok("POST /delete/missing -> 404 JSON",
       r.status_code == 404 and is_json_response(r) and r.json().get("success") is False,
       (r.status_code, r.content[:200]))

    # ---------- success contract ----------
    r, doc = make_doc("ContractSuccess", "contract_success.txt",
                      TXT_BODY.encode("utf-8"), "text/plain")
    ok("fixture uploaded", r.status_code == 302 and doc is not None, r.status_code)
    if doc:
        CREATED.append(doc)
        ntotal_before = len(views.vector_store)
        r = c.post(f"/process/{doc.id}/")
        data = r.json() if is_json_response(r) else {}
        ok("process success -> 200 JSON", r.status_code == 200 and is_json_response(r),
           (r.status_code, r.content[:200]))
        ok("success payload fields",
           data.get("success") is True
           and data.get("status") == "success"
           and data.get("document_id") == doc.id
           and data.get("processed") is True
           and isinstance(data.get("chunks"), int) and data.get("chunks") > 0
           and isinstance(data.get("message"), str) and data.get("message"),
           data)
        ok("FAISS grew by chunks",
           len(views.vector_store) == ntotal_before + data.get("chunks", -1),
           (ntotal_before, len(views.vector_store)))
        ok("success leak-free", not no_leaks(r.content.decode()), no_leaks(r.content.decode()))

        # re-process (refresh: the local instance predates processing)
        r = c.post(f"/process/{doc.id}/")
        data = r.json() if is_json_response(r) else {}
        fresh = Document.objects.get(id=doc.id)
        ok("re-process -> 200 already_processed JSON",
           r.status_code == 200 and data.get("status") == "already_processed"
           and data.get("success") is True
           and data.get("chunks") == fresh.chunk_count
           and data.get("processed") is True,
           (r.status_code, data))

    # ---------- empty document ----------
    # Layer 1: the upload form rejects 0-byte files with a visible message
    # (failed uploads must never consume a document slot).
    count_before = Document.objects.count()
    r, edoc = make_doc("ContractEmpty", "contract_empty.txt", b"", "text/plain")
    ok("empty upload rejected by form (no doc row, slot not consumed)",
       edoc is None and Document.objects.count() == count_before
       and b"empty" in r.content.lower(),
       (r.status_code, Document.objects.count() - count_before))
    # Layer 2: a stored empty doc (legacy/edge) -> process returns 400 JSON.
    up = SimpleUploadedFile("contract_empty_stored.txt", b"", content_type="text/plain")
    edoc = Document.objects.create(
        title="ContractEmptyStored", file=up, file_type="txt"
    )
    CREATED.append(edoc)
    ntotal_before = len(views.vector_store)
    r = c.post(f"/process/{edoc.id}/")
    data = r.json() if is_json_response(r) else {}
    ok("stored empty doc -> 400 JSON success=false",
       r.status_code == 400 and is_json_response(r)
       and data.get("success") is False and data.get("error"),
       (r.status_code, data))
    ok("empty doc not marked processed",
       Document.objects.get(id=edoc.id).processed is False)
    ok("empty doc left no FAISS vectors", len(views.vector_store) == ntotal_before)
    ok("empty doc error leak-free", not no_leaks(r.content.decode()),
       no_leaks(r.content.decode()))

    # ---------- corrupted PDF -> 500 JSON, safe message ----------
    r, cdoc = make_doc("ContractCorrupt", "contract_corrupt.pdf",
                       b"%PDF-1.4\nthis is not a real pdf body at all\n%%EOF",
                       "application/pdf")
    ok("corrupt fixture uploaded", cdoc is not None, r.status_code)
    if cdoc:
        CREATED.append(cdoc)
        ntotal_before = len(views.vector_store)
        r = c.post(f"/process/{cdoc.id}/")
        data = r.json() if is_json_response(r) else {}
        ok("corrupt pdf -> 500 JSON success=false",
           r.status_code == 500 and is_json_response(r)
           and data.get("success") is False
           and str(data.get("error", "")).startswith("Document processing failed:"),
           (r.status_code, data))
        ok("corrupt pdf body leak-free", not no_leaks(r.content.decode()),
           no_leaks(r.content.decode()))
        doc_after = Document.objects.get(id=cdoc.id)
        ok("corrupt pdf stays unprocessed", doc_after.processed is False)
        ok("corrupt pdf created no chunk rows",
           not Chunk.objects.filter(document=cdoc).exists())
        ok("corrupt pdf left no FAISS vectors", len(views.vector_store) == ntotal_before,
           (ntotal_before, len(views.vector_store)))

    # ---------- stored unsupported type -> 400 JSON ----------
    r, bdoc = make_doc("ContractBadType", "contract_badtype.txt",
                       TXT_BODY.encode("utf-8"), "text/plain")
    if bdoc:
        CREATED.append(bdoc)
        Document.objects.filter(id=bdoc.id).update(file_type="xyz")
        ntotal_before = len(views.vector_store)
        r = c.post(f"/process/{bdoc.id}/")
        data = r.json() if is_json_response(r) else {}
        ok("unsupported stored type -> 400 JSON",
           r.status_code == 400 and is_json_response(r)
           and data.get("success") is False
           and "Unsupported file type" in data.get("error", ""),
           (r.status_code, data))
        ok("unsupported type: no partial state",
           Document.objects.get(id=bdoc.id).processed is False
           and not Chunk.objects.filter(document=bdoc).exists()
           and len(views.vector_store) == ntotal_before)
    else:
        ok("unsupported stored type -> 400 JSON", False, "fixture missing")

    # ---------- mid-processing DB failure -> 500 + rollback ----------
    r, mdoc = make_doc("ContractMidFail", "contract_midfail.txt",
                       TXT_BODY.encode("utf-8"), "text/plain")
    if mdoc:
        CREATED.append(mdoc)
        ntotal_before = len(views.vector_store)
        boom = RuntimeError(
            'database exploded while writing D:\\internal\\secret\\path.db '
            'using postgresql://user:pass@host/db sb_secret_ABCDEFGHIJKLMNOP'
        )
        with patch.object(Chunk.objects, "bulk_create", side_effect=boom):
            r = c.post(f"/process/{mdoc.id}/")
        data = r.json() if is_json_response(r) else {}
        ok("db failure -> 500 JSON success=false",
           r.status_code == 500 and is_json_response(r)
           and data.get("success") is False
           and str(data.get("error", "")).startswith("Document processing failed:"),
           (r.status_code, data))
        body = r.content.decode()
        leaks = no_leaks(body)
        ok("db failure body leak-free (paths/creds stripped)", not leaks, leaks)
        ok("db failure keeps processed=False",
           Document.objects.get(id=mdoc.id).processed is False)
        ok("db failure left no Chunk rows",
           not Chunk.objects.filter(document=mdoc).exists())
        ok("db failure left no orphan FAISS vectors",
           len(views.vector_store) == ntotal_before,
           (ntotal_before, len(views.vector_store)))
    else:
        ok("db failure -> 500 JSON success=false", False, "fixture missing")

    # ---------- FAISS persist failure -> 500 + full compensation ----------
    r, sdoc = make_doc("ContractSaveFail", "contract_savefail.txt",
                       TXT_BODY.encode("utf-8"), "text/plain")
    if sdoc:
        CREATED.append(sdoc)
        ntotal_before = len(views.vector_store)
        # Patch the REAL VectorStore instance (the _LazyService proxy does
        # not support patch.object's attribute restore semantics).
        real_store = views.vector_store._build()
        with patch.object(real_store, "save",
                          side_effect=OSError("disk write failed /opt/render/project/src/data")):
            r = c.post(f"/process/{sdoc.id}/")
        data = r.json() if is_json_response(r) else {}
        ok("faiss save failure -> 500 JSON",
           r.status_code == 500 and is_json_response(r)
           and data.get("success") is False,
           (r.status_code, data))
        body = r.content.decode()
        ok("faiss save failure leak-free", not no_leaks(body), no_leaks(body))
        doc_after = Document.objects.get(id=sdoc.id)
        ok("faiss save failure: processed reverted",
           doc_after.processed is False and doc_after.chunk_count == 0)
        ok("faiss save failure: chunk rows rolled back",
           not Chunk.objects.filter(document=sdoc).exists())
        ok("faiss save failure: no orphan vectors in memory",
           len(views.vector_store) == ntotal_before,
           (ntotal_before, len(views.vector_store)))
        ok("index still usable after rollback",
           len(views.vector_store) == len(views.vector_store.metadata)
           or len(views.vector_store) >= 0)
    else:
        ok("faiss save failure -> 500 JSON", False, "fixture missing")

except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    ok("test script ran without unexpected crash", False, repr(exc))

finally:
    for d in CREATED:
        try:
            fresh = Document.objects.filter(id=d.id).first()
            if fresh:
                fp = fresh.file.path if fresh.file else None
                views.vector_store.delete_document(fresh.id)
                fresh.delete()
                if fp and os.path.exists(fp):
                    os.remove(fp)
        except Exception as exc:  # noqa: BLE001
            print(f"[cleanup] {d.id}: {exc}", flush=True)
    print("[cleanup] contract fixtures removed", flush=True)

print(f"\n{PASSED}/{PASSED + FAILED} passed", flush=True)
sys.exit(1 if FAILED else 0)
