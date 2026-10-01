import os
import json
import re
import itertools
from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from django.conf import settings
from .vector_store import VectorStore
from .embedder import Embedder
from .query_pool import build_section_queries
from .llm_resilience import invoke_with_retry
from .sanitize import scrub_citation_markers
from .grounding import filter_supporting_results
from .analytics import log_retrieval


def _redact_secrets(text: str) -> str:
    """Prevent API keys from leaking into logs."""
    text = re.sub(r"(?i)([?&]key=)[^&\s\"']+", r"\1***", text)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "***", text)
    return text


TUTOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a study assistant. Answer the question using ONLY the provided context.
Give a direct, plain-text answer. No markdown. No bullet points. No bold. No formatting.
Keep it short and to the point.
The context may contain sections from several documents; use whichever section actually addresses the question.
If the question's premise contradicts the context (wrong fact, date, subject or edition), do not refuse.
Answer by stating what the context actually says and briefly correct the premise.
Refuse only when the context genuinely has no information about the question; then say "I don't have enough information about this."

Context:
{context}

Question: {question}

Answer:"""
)

MCQ_PROMPT = PromptTemplate(
    input_variables=["context", "topic", "difficulty"],
    template="""Generate a multiple-choice question from the context.

Context:
{context}

Topic: {topic}
Difficulty: {difficulty}

Requirements:
- 4 options (A, B, C, D)
- One correct answer
- Include explanation of why the correct answer is right
- Question should test understanding, not just memorization

Output format (JSON):
{{
    "question": "Question text here",
    "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
    "correct": "A",
    "explanation": "Why the correct answer is right...",
    "topic": "Topic name"
}}"""
)

MOCK_PROMPT = PromptTemplate(
    input_variables=["context", "difficulty"],
    template="""Generate an exam-style multiple-choice question from the context.

Context:
{context}
Difficulty: {difficulty}

Requirements:
- 4 options (A, B, C, D)
- One correct answer
- Include explanation
- Tag with a specific topic/concept
- {difficulty} level: easy=basic recall, medium=application, hard=analysis/synthesis

Output format (JSON):
{{
    "question": "Question text here",
    "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
    "correct": "A",
    "explanation": "Why the correct answer is right...",
    "topic": "Specific topic/concept"
}}"""
)

MCQ_BATCH_PROMPT = PromptTemplate(
    input_variables=["context", "topic", "difficulty", "count"],
    template="""Generate exactly {count} multiple-choice questions from the context.

Context:
{context}

Topic: {topic}
Difficulty: {difficulty}

Requirements:
- Exactly {count} questions
- Each question: 4 options (A, B, C, D), one correct answer, explanation, topic tag
- {difficulty} level: easy=basic recall, medium=application, hard=analysis/synthesis
- Each question must focus on a DIFFERENT concept from the context where possible;
  do not ask several questions about the same idea
- Ground every question ONLY in the provided context; never invent facts,
  concepts, or details that are not in the context
- Question should test understanding, not just memorization
- "source_chunks": integers listing ONLY the context chunk ids [doc_N:chunk_M]
  that support this question — the fewest chunks that justify the correct answer;
  each question must cite its own supporting chunks, not every context chunk

Output format (JSON only, no other text):
{{
    "questions": [
        {{
            "question": "Question text here",
            "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
            "correct": "A",
            "explanation": "Why the correct answer is right...",
            "topic": "Topic name",
            "source_chunks": [13, 42]
        }}
    ]
}}"""
)

MOCK_BATCH_PROMPT = PromptTemplate(
    input_variables=["context", "difficulty", "count"],
    template="""Generate exactly {count} exam-style multiple-choice questions from the context.

Context:
{context}
Difficulty: {difficulty}

Requirements:
- Exactly {count} questions
- Each question: 4 options (A, B, C, D), one correct answer, explanation, specific topic tag
- {difficulty} level: easy=basic recall, medium=application, hard=analysis/synthesis
- Each question must focus on a DIFFERENT concept from the context where possible;
  do not ask several questions about the same idea
- Ground every question ONLY in the provided context; never invent facts,
  concepts, or details that are not in the context
- Questions should test understanding, not just memorization
- "source_chunks": integers listing ONLY the context chunk ids [doc_N:chunk_M]
  that support this question — the fewest chunks that justify the correct answer;
  each question must cite its own supporting chunks, not every context chunk

Output format (JSON only, no other text):
{{
    "questions": [
        {{
            "question": "Question text here",
            "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
            "correct": "A",
            "explanation": "Why the correct answer is right...",
            "topic": "Specific topic/concept",
            "source_chunks": [13, 42]
        }}
    ]
}}"""
)

TRUE_FALSE_BATCH_PROMPT = PromptTemplate(
    input_variables=["context", "difficulty", "count"],
    template="""Generate exactly {count} true/false exam questions from the context.

Context:
{context}
Difficulty: {difficulty}

Requirements:
- Exactly {count} questions
- Each question is a factual statement that is clearly TRUE or clearly FALSE according to the context
- Ground every statement ONLY in the provided context; do not rely on outside knowledge
- The statement must be directly supported or contradicted by the supplied context
- Each statement must have exactly one correct answer
- Avoid ambiguous wording, double negatives, and statements that are
  technically true under one interpretation and false under another
- {difficulty} level: easy=basic recall, medium=application, hard=analysis/synthesis
- Include an explanation grounded in the context for each statement
- Tag each with a specific topic/concept
- "source_chunks": integers listing ONLY the context chunk ids [doc_N:chunk_M]
  that support or contradict this statement — the fewest chunks that suffice;
  each statement must cite its own supporting chunks, not every context chunk

Output format (JSON only, no other text):
{{
    "questions": [
        {{
            "question_type": "true_false",
            "question": "Statement here",
            "options": {{"A": "True", "B": "False"}},
            "correct": "A",
            "explanation": "Explanation grounded in the context...",
            "topic": "Specific topic/concept",
            "source_chunks": [13, 42]
        }}
    ]
}}"""
)


class RAGChain:
    def __init__(self, vector_store: VectorStore = None, embedder: Embedder = None):
        self.vector_store = vector_store or VectorStore()
        self.embedder = embedder or Embedder()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        model_name = getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
        from langchain_google_genai import ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
        )
        self.top_k = getattr(settings, "TOP_K_RETRIEVAL", 5)
        # Deterministic rotation over the corpus-derived query pool for mock and
        # practice generation (thread-safe; persists across batches/requests so
        # different generation runs advance through the pool instead of always
        # restarting at seed 0).
        self._mock_seed_counter = itertools.count()
        self._practice_seed_counter = itertools.count()

    def _next_pool_query(self, doc_ids: List[int], counter,
                         offset_half: bool = False) -> Optional[str]:
        """Corpus-derived retrieval query for one generation batch."""
        pool = build_section_queries(self.vector_store, doc_ids)
        if not pool:
            return None
        idx = next(counter)
        if offset_half:
            idx += len(pool) // 2
        return pool[idx % len(pool)]

    def next_practice_query(self, doc_ids: List[int] = None) -> Optional[str]:
        return self._next_pool_query(doc_ids, self._practice_seed_counter)

    def _next_mock_query(self, doc_ids: List[int] = None, offset_half: bool = False) -> Optional[str]:
        return self._next_pool_query(doc_ids, self._mock_seed_counter, offset_half)

    @staticmethod
    def _extract_text(response) -> str:
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            return " ".join(parts).strip()
        return str(content)

    def _invoke(self, prompt):
        return invoke_with_retry(self.llm, prompt)

    def _retrieve_context(self, query: str, doc_ids: List[int] = None,
                          operation: str = "retrieve") -> List[Dict[str, Any]]:
        query_emb = self.embedder.embed_query(query)
        results = self.vector_store.search(query_emb, k=self.top_k, doc_ids=doc_ids)
        log_retrieval(
            operation,
            [r.get("score") for r in results],
            doc_ids=doc_ids,
            docs=[r.get("doc_id") for r in results],
        )
        return results

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        parts = []
        for r in results:
            doc_id = r.get("doc_id", "?")
            chunk_idx = r.get("chunk_index", "?")
            text = r.get("text", r.get("content", ""))
            parts.append(f"[doc_{doc_id}:chunk_{chunk_idx}] {text}")
        return "\n\n".join(parts)

    def tutor_query(self, question: str) -> Dict[str, Any]:
        """Tutor mode: Answer question with citations."""
        results = self._retrieve_context(question, operation="tutor")
        if not results:
            return {"answer": "No relevant documents found. Please upload study materials first.", "sources": []}
        
        context = self._format_context(results)
        prompt = TUTOR_PROMPT.format(context=context, question=question)
        response = self._invoke(prompt)
        answer = scrub_citation_markers(self._extract_text(response))
        
        supported = filter_supporting_results(answer, results)
        sources = []
        for r in supported:
            sources.append({
                "doc_id": r.get("doc_id"),
                "chunk_index": r.get("chunk_index"),
                "page_number": r.get("page_number"),
                "file_name": r.get("file_name"),
                "text": r.get("text", r.get("content", ""))[:200],
                "score": r.get("score"),
            })
        
        return {
            "answer": answer,
            "sources": sources,
        }

    @staticmethod
    def _extract_json(text: str) -> str:
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        return clean.strip()

    def generate_mcq(self, topic: str = None, difficulty: str = "medium", doc_ids: List[int] = None) -> Dict[str, Any]:
        """Practice mode: Generate MCQ from context."""
        query = topic or "key concepts"
        results = self._retrieve_context(query, doc_ids=doc_ids, operation="practice")
        if not results:
            return {"error": "No content available. Upload documents first."}
        
        context = self._format_context(results)
        prompt = MCQ_PROMPT.format(context=context, topic=topic or "General", difficulty=difficulty)
        response = self._invoke(prompt)
        raw_text = self._extract_text(response)
        clean_text = self._extract_json(raw_text)
        
        try:
            mcq = json.loads(clean_text)
            mcq["source_chunks"] = [r.get("faiss_id") for r in results]
            return mcq
        except json.JSONDecodeError:
            return {"error": "Failed to parse MCQ response", "raw": raw_text}

    def generate_mock_question(self, difficulty: str = "medium", doc_ids: List[int] = None) -> Dict[str, Any]:
        """Mock test mode: Generate exam question."""
        results = self._retrieve_context(self._next_mock_query(doc_ids), doc_ids=doc_ids,
                                         operation="mock")
        if not results:
            return {"error": "No content available. Upload documents first."}
        
        context = self._format_context(results)
        prompt = MOCK_PROMPT.format(context=context, difficulty=difficulty)
        response = self._invoke(prompt)
        raw_text = self._extract_text(response)
        clean_text = self._extract_json(raw_text)
        
        try:
            mcq = json.loads(clean_text)
            mcq["source_chunks"] = [r.get("faiss_id") for r in results]
            return mcq
        except json.JSONDecodeError:
            return {"error": "Failed to parse question", "raw": raw_text}

    def _parse_questions_list(self, text: str, expected: int) -> List[Dict[str, Any]]:
        """Parse a batch JSON response into a list of MCQ dicts."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict):
            questions = data.get("questions")
        elif isinstance(data, list):
            questions = data
        else:
            return []
        if not isinstance(questions, list):
            return []
        return [q for q in questions if isinstance(q, dict)][:expected]

    @staticmethod
    def _attach_sources(questions: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> None:
        """Per-question source attribution.

        Keeps model-cited chunk ids that are integers within the retrieved set
        (order-preserving, deduplicated); falls back to the whole retrieval
        batch when a question cites nothing valid, so source_chunks is never
        empty when retrieval returned results.
        """
        batch_ids = [r.get("faiss_id") for r in results]
        allowed = {i for i in batch_ids if isinstance(i, int) and not isinstance(i, bool)}
        for question in questions:
            cited = question.get("source_chunks")
            clean: List[int] = []
            if isinstance(cited, list):
                for item in cited:
                    if isinstance(item, bool):
                        continue
                    try:
                        fid = int(item)
                    except (TypeError, ValueError):
                        continue
                    if fid in allowed and fid not in clean:
                        clean.append(fid)
            question["source_chunks"] = clean or list(batch_ids)

    def generate_mcq_batch(
        self,
        topic: str = None,
        difficulty: str = "medium",
        doc_ids: List[int] = None,
        count: int = 5,
    ) -> List[Dict[str, Any]]:
        """Practice mode: generate multiple MCQs in one LLM call."""
        query = topic or "key concepts"
        results = self._retrieve_context(query, doc_ids=doc_ids, operation="practice")
        if not results:
            return []

        context = self._format_context(results)
        prompt = MCQ_BATCH_PROMPT.format(
            context=context,
            topic=topic or "General",
            difficulty=difficulty,
            count=count,
        )
        try:
            response = self._invoke(prompt)
        except Exception as exc:
            print(
                f"[ERROR] practice MCQ batch failed | batch_count={count} | doc_ids={doc_ids} | "
                + _redact_secrets(f"{type(exc).__name__}: {exc}")
            )
            return []
        raw_text = self._extract_text(response)
        clean_text = self._extract_json(raw_text)
        questions = self._parse_questions_list(clean_text, count)
        self._attach_sources(questions, results)
        return questions

    def generate_mock_questions_batch(
        self,
        difficulty: str = "medium",
        doc_ids: List[int] = None,
        count: int = 5,
        question_type: str = "mcq",
    ) -> List[Dict[str, Any]]:
        """Mock test mode: generate multiple exam questions in one LLM call.

        question_type is "mcq" or "true_false"; retrieval (FAISS + doc_ids
        filtering) is identical for both types.
        """
        is_true_false = question_type == "true_false"
        results = self._retrieve_context(
            self._next_mock_query(doc_ids, offset_half=is_true_false),
            doc_ids=doc_ids,
            operation="mock",
        )
        if not results:
            return []

        prompt_template = TRUE_FALSE_BATCH_PROMPT if is_true_false else MOCK_BATCH_PROMPT
        context = self._format_context(results)
        prompt = prompt_template.format(
            context=context,
            difficulty=difficulty,
            count=count,
        )
        try:
            response = self._invoke(prompt)
        except Exception as exc:
            print(
                f"[ERROR] mock {'true_false' if is_true_false else 'MCQ'} batch failed | batch_count={count} | doc_ids={doc_ids} | "
                + _redact_secrets(f"{type(exc).__name__}: {exc}")
            )
            return []
        raw_text = self._extract_text(response)
        clean_text = self._extract_json(raw_text)
        questions = self._parse_questions_list(clean_text, count)
        for question in questions:
            question["question_type"] = question_type
        self._attach_sources(questions, results)
        return questions