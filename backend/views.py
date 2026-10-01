import json
import os
import sys
import threading
import time
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.conf import settings
from django.utils import timezone

from backend.models import Document, Chunk, MockTest, TestAttempt, UserAnswer
from backend.forms import DocumentUploadForm, TutorQuestionForm
from backend import supabase_storage
from rag import (
    FileParser,
    TextChunker,
    Embedder,
    VectorStore,
    RAGChain,
    MCQGenerator,
)
from rag.cleaning import clean_pages, near_duplicate_mask
from rag.llm_resilience import is_transient_llm_error
# mock_test (pulls pandas + sklearn) is imported lazily by the proxies below.


def _rss_mb():
    """Current process resident set size in MB, or None if unavailable."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            proc = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            ctypes.windll.kernel32.K32GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            if not ctypes.windll.kernel32.K32GetProcessMemoryInfo(
                proc, ctypes.byref(counters), counters.cb
            ):
                return None
            return counters.WorkingSetSize / (1024 * 1024)
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * (os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))
    except Exception:
        return None


def _log_rss(stage):
    rss = _rss_mb()
    if rss is not None:
        print(f"[MEM] rss={rss:.1f} MB ({stage})", flush=True)


class _LazyService:
    """Defer building a heavy service until its first use.

    Keeps gunicorn/Django boot free of torch/sentence_transformers/pandas so
    the Render free instance passes its health check under the 512MB limit.
    Attribute access, calls and len() all transparently trigger the build.
    """

    def __init__(self, name, factory):
        self._name = name
        self._factory = factory
        self._obj = None
        self._lock = threading.Lock()

    def _build(self):
        # Read state via __dict__ so a missing attribute can never recurse
        # back into __getattr__ while the object is still being constructed.
        state = self.__dict__
        if state["_obj"] is None:
            with state["_lock"]:
                if state["_obj"] is None:
                    state["_obj"] = state["_factory"]()
                    _log_rss(f"loaded {state['_name']}")
        return state["_obj"]

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return getattr(self._build(), item)

    def __setattr__(self, item, value):
        # Route attribute writes to the real service so that test/service
        # patches like `rag_chain.llm = stub` take effect on the object whose
        # methods actually run (matches the pre-proxy eager behavior).
        if item in ("_name", "_factory", "_obj", "_lock"):
            object.__setattr__(self, item, value)
        else:
            setattr(self._build(), item, value)

    def __call__(self, *args, **kwargs):
        return self._build()(*args, **kwargs)

    def __bool__(self):
        # Always truthy so `proxy or Fallback()` keeps the proxy (a real
        # service could be falsy via len(), which would build it eagerly).
        return True

    def __len__(self):
        return len(self._build())


def _build_mock_test_service():
    from mock_test import MockTestService
    return MockTestService(rag_chain)


def _build_analyzer():
    from mock_test import PerformanceAnalyzer
    return PerformanceAnalyzer()


# Initialize services (lazily: built on first request that needs them)
vector_store = _LazyService("vector_store", lambda: VectorStore())
embedder = _LazyService("embedder", lambda: Embedder())
rag_chain = _LazyService("rag_chain", lambda: RAGChain(vector_store, embedder))
mcq_generator = _LazyService("mcq_generator", lambda: MCQGenerator(rag_chain))
mock_test_service = _LazyService("mock_test_service", _build_mock_test_service)
analyzer = _LazyService("analyzer", _build_analyzer)

_log_rss("imported backend.views")


def upload_view(request):
    """File upload page with document list."""
    if request.method == "POST":
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            doc = form.save(commit=False)
            if not doc.title:
                doc.title = doc.file.name
            doc.file_type = FileParser.detect_file_type(doc.file.name)
            doc.save()
            if supabase_storage.is_enabled():
                try:
                    with open(doc.file.path, "rb") as handle:
                        supabase_storage.upload_file(doc.file.name, handle.read())
                except Exception as exc:
                    messages.warning(
                        request,
                        f"Saved, but cloud storage sync failed ({type(exc).__name__}).",
                    )
            messages.success(request, f"Uploaded: {doc.title}")
            return redirect("upload")
    else:
        form = DocumentUploadForm()
    
    documents = Document.objects.all()
    return render(request, "upload.html", {"form": form, "documents": documents})


@require_http_methods(["POST"])
def process_document_view(request, doc_id):
    """Process uploaded document: parse, chunk, embed, store."""
    doc = get_object_or_404(Document, id=doc_id)
    
    if doc.processed:
        return JsonResponse({"status": "already_processed", "chunks": doc.chunk_count})
    
    try:
        # Parse (local file if present; otherwise pull a temp copy from Storage)
        file_path, temp_path = supabase_storage.materialize_file(
            doc.file.name, doc.file.path
        )
        try:
            pages = FileParser.parse(file_path, doc.file_type)
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        # Clean (whitespace/dehyphenation + repeated header/footer removal)
        pages = clean_pages(pages)

        # Chunk
        chunker = TextChunker(
            chunk_size=getattr(settings, "CHUNK_SIZE", 500),
            chunk_overlap=getattr(settings, "CHUNK_OVERLAP", 50),
        )
        chunks = chunker.chunk_pages(pages)

        if not chunks:
            return JsonResponse({"status": "error", "message": "No text extracted from document"})

        # Embed
        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed(texts)

        # Drop near-duplicate chunks (same document only) before indexing
        keep = near_duplicate_mask(embeddings, texts)
        if not all(keep):
            chunks = [c for c, k in zip(chunks, keep) if k]
            embeddings = embeddings[[i for i, k in enumerate(keep) if k]]
        if not chunks:
            return JsonResponse({"status": "error", "message": "No text extracted from document"})
        
        # Prepare metadata
        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "text": chunk["content"],
                "doc_id": doc.id,
                "chunk_index": chunk["chunk_index"],
                "page_number": chunk["page_number"],
                "file_name": chunk["file_name"],
            })
        
        # Store in FAISS
        faiss_ids = vector_store.add(embeddings, metadata)
        vector_store.save()
        
        # Save to Django models
        chunk_objects = [
            Chunk(
                document=doc,
                content=chunk["content"],
                embedding_id=faiss_ids[i],
                page_number=chunk["page_number"],
                chunk_index=chunk["chunk_index"],
            )
            for i, chunk in enumerate(chunks)
        ]
        Chunk.objects.bulk_create(chunk_objects)
        
        # Update document
        doc.processed = True
        doc.chunk_count = len(chunks)
        doc.save()
        
        return JsonResponse({"status": "success", "chunks": len(chunks)})
    
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})


@require_http_methods(["POST"])
def delete_document_view(request, doc_id):
    """Permanently delete a document: DB record, chunks, vector embeddings, and file."""
    doc = get_object_or_404(Document, id=doc_id)

    try:
        doc_title = doc.title

        # 1. Remove from FAISS vector store
        vector_store.delete_document(doc.id)

        # 2. Delete uploaded file from disk and best-effort from Supabase Storage
        if doc.file:
            relative_name = doc.file.name
            doc.file.delete(save=False)
            if relative_name and supabase_storage.is_enabled():
                try:
                    supabase_storage.delete_file(relative_name)
                except Exception:
                    pass

        # 3. Delete document record (CASCADE deletes chunks too)
        doc.delete()

        return JsonResponse({
            "status": "success",
            "message": f'"{doc_title}" was permanently deleted.',
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def tutor_view(request):
    """Tutor mode - Q&A interface."""
    form = TutorQuestionForm()
    return render(request, "tutor.html", {"form": form})


@csrf_exempt
@require_http_methods(["POST"])
def tutor_ask_view(request):
    """AJAX endpoint for tutor questions."""
    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip()
        
        if not question:
            return JsonResponse({"error": "Question is required"}, status=400)
        
        result = rag_chain.tutor_query(question)
        return JsonResponse(result)

    except Exception as e:
        if is_transient_llm_error(e):
            return JsonResponse(
                {"error": "The AI service is temporarily unavailable. Please try again."},
                status=503,
            )
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
def sources_view(request):
    """Direct lookup of source chunks by FAISS ids. No LLM involved."""
    raw = request.GET.get("ids", "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return JsonResponse({"error": "ids parameter is required"}, status=400)
    if len(parts) > 50:
        return JsonResponse({"error": "too many ids (max 50)"}, status=400)

    ids = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            return JsonResponse({"error": f"invalid id: {part}"}, status=400)
        if value < 0:
            return JsonResponse({"error": f"invalid id: {part}"}, status=400)
        ids.append(value)

    sources = []
    for fid in ids:
        meta = vector_store.get_by_id(fid)
        if not meta:
            continue
        sources.append({
            "faiss_id": meta.get("faiss_id", fid),
            "doc_id": meta.get("doc_id"),
            "chunk_index": meta.get("chunk_index"),
            "page_number": meta.get("page_number"),
            "file_name": meta.get("file_name"),
            "text": meta.get("text", meta.get("content", "")),
        })
    return JsonResponse({"sources": sources})


def practice_view(request):
    """Practice mode - MCQ generation and answering."""
    documents = Document.objects.filter(processed=True)
    return render(request, "practice.html", {"documents": documents})


@require_http_methods(["POST"])
def practice_generate_view(request):
    """Generate MCQs for practice."""
    try:
        data = json.loads(request.body)
        num_questions = int(data.get("num_questions", 5))
        difficulty = data.get("difficulty", "medium")
        topic = data.get("topic", None)
        doc_ids = data.get("doc_ids", None)

        if not doc_ids:
            return JsonResponse({"error": "Please select at least one study material."}, status=400)

        valid_ids = list(
            Document.objects.filter(id__in=doc_ids, processed=True).values_list("id", flat=True)
        )
        if not valid_ids:
            return JsonResponse({"error": "Please select at least one study material."}, status=400)

        mcqs = mcq_generator.generate_practice_set(num_questions, difficulty, topic, doc_ids=valid_ids)
        return JsonResponse({"mcqs": mcqs})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["POST"])
def practice_submit_view(request):
    """Evaluate practice MCQ answers."""
    try:
        data = json.loads(request.body)
        mcq = data.get("mcq", {})
        selected = data.get("selected", "")

        result = mcq_generator.evaluate_answer(mcq, selected)
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def mock_test_settings_view(request):
    """Mock test configuration page."""
    documents = Document.objects.filter(processed=True)

    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid request"}, status=400)

        num_questions = int(data.get("num_questions", 20))
        difficulty = data.get("difficulty", "medium")
        timer_minutes = int(data.get("timer_minutes", 30))
        doc_ids = data.get("doc_ids", [])

        valid_ids = list(
            Document.objects.filter(id__in=doc_ids, processed=True).values_list("id", flat=True)
        )
        if not valid_ids:
            return JsonResponse({"error": "Please select at least one study material."}, status=400)

        if num_questions not in [10, 20, 30, 50]:
            return JsonResponse({"error": "Invalid number of questions."}, status=400)
        if difficulty not in ["easy", "medium", "hard"]:
            return JsonResponse({"error": "Invalid difficulty."}, status=400)
        if timer_minutes not in [10, 20, 30, 60]:
            return JsonResponse({"error": "Invalid timer."}, status=400)

        try:
            test = mock_test_service.create_test(
                num_questions=num_questions,
                difficulty=difficulty,
                timer_minutes=timer_minutes,
                doc_ids=valid_ids,
                question_type="both",
            )
        except ValueError as e:
            return JsonResponse({"error": str(e)}, status=500)

        attempt = TestAttempt.objects.create(
            test=test,
            started_at=timezone.now(),
            total_questions=num_questions,
            status="active",
        )
        test.status = "active"
        test.save()

        return JsonResponse({"redirect": f"/mock-test/{test.id}/"})

    return render(request, "mock_test/settings.html", {"documents": documents})


def mock_test_start_view(request):
    """Start mock test - same as settings POST for convenience."""
    return mock_test_settings_view(request)


def mock_test_take_view(request, test_id):
    """Take the mock test - timer UI with questions."""
    test = get_object_or_404(MockTest, id=test_id)
    attempt = TestAttempt.objects.filter(test=test).first()

    if test.status == "completed" and attempt:
        return redirect("mock_test_result", attempt_id=attempt.id)
    if test.status == "terminated":
        return redirect("mock_test_terminated")
    if attempt and attempt.status == "completed":
        return redirect("mock_test_result", attempt_id=attempt.id)
    if attempt and attempt.status == "terminated":
        return redirect("mock_test_terminated")

    questions = mock_test_service.get_test_questions(test_id)

    return render(request, "mock_test/take_test.html", {
        "test": test,
        "questions": questions,
        "timer_seconds": test.timer_minutes * 60,
        "attempt_id": attempt.id if attempt else None,
    })


@require_http_methods(["POST"])
def mock_test_submit_view(request, test_id):
    """Submit and grade mock test."""
    try:
        data = json.loads(request.body)
        answers = data.get("answers", {})
        time_taken = int(data.get("time_taken", 0))

        answers = {int(k): v for k, v in answers.items()}

        attempt = mock_test_service.submit_attempt(test_id, answers, time_taken)
        return JsonResponse({
            "attempt_id": attempt.id,
            "redirect": f"/mock-test/{attempt.id}/result/",
        })
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["POST"])
def mock_test_terminate_view(request):
    """Terminate an active exam (e.g., fullscreen exit)."""
    try:
        data = json.loads(request.body)
        test_id = data.get("test_id")
        attempt_id = data.get("attempt_id")

        if not test_id or not attempt_id:
            return JsonResponse({"error": "Missing test_id or attempt_id"}, status=400)

        mock_test_service.terminate_attempt(test_id, attempt_id)
        return JsonResponse({
            "success": True,
            "redirect": "/mock-test/terminated/",
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def mock_test_terminated_view(request):
    """Show exam terminated page."""
    return render(request, "mock_test/terminated.html")


def mock_test_result_view(request, attempt_id):
    """Show test result summary."""
    attempt = get_object_or_404(TestAttempt, id=attempt_id)
    answers = UserAnswer.objects.filter(attempt=attempt)
    wrong_count = answers.filter(is_correct=False).exclude(selected_option="").count()
    unanswered_count = answers.filter(selected_option="").count()
    return render(request, "mock_test/result.html", {
        "attempt": attempt,
        "wrong_count": wrong_count,
        "unanswered_count": unanswered_count,
    })


def mock_test_analysis_view(request, attempt_id):
    """Show performance analysis with charts."""
    attempt = get_object_or_404(TestAttempt, id=attempt_id)
    chart_data = analyzer.get_chart_data(attempt)
    return render(request, "mock_test/analysis.html", {
        "attempt": attempt,
        "chart_data": json.dumps(chart_data),
    })


def mock_test_review_view(request, attempt_id):
    """Review wrong answers with explanations."""
    attempt = get_object_or_404(TestAttempt, id=attempt_id)
    result = mock_test_service.get_attempt_result(attempt_id)
    return render(request, "mock_test/review.html", {
        "attempt": attempt,
        "wrong_questions": result["wrong"],
    })