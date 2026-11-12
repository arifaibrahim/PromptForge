"""PromptForge Offline prompt-engineering engine.

This module is intentionally self-contained. It loads and indexes the four local
PDF rulebooks, retrieves task-relevant pages, communicates only with a local
Ollama server, manages clarification/refinement, and performs two separate
verification passes before returning a prompt.

Expected project layout:
    PromptForge-Offline/
        prompt_engine.py
        knowledge/
            general_guidelines.pdf
            gpt_luna_guidelines.pdf
            gemini_flash_guidelines.pdf
            coding_guidelines.pdf

Required packages: pypdf, requests
Required environment variable: LOCAL_LLM_MODEL

Optional environment variables:
    OLLAMA_TIMEOUT_SECONDS  (default: 180)
    OLLAMA_NUM_CTX          (default: 32768)
    OLLAMA_NUM_PREDICT      (default: 8192)
    OLLAMA_TEMPERATURE      (default: 0.2)
    PROMPTFORGE_KNOWLEDGE_DIR (override the default knowledge/ directory)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = Path(
    os.environ.get("PROMPTFORGE_KNOWLEDGE_DIR", str(BASE_DIR / "knowledge"))
).expanduser()
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"

OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "8192"))
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))

VALID_TARGET_MODELS = {"gpt_luna", "gemini_flash"}
TARGET_MODEL_NAMES = {
    "gpt_luna": "GPT-5.6 Luna",
    "gemini_flash": "Gemini 3.8 Flash",
}

SOURCE_CONFIG: dict[str, dict[str, Any]] = {
    "general": {
        "filename": "general_guidelines.pdf",
        "display_name": "General Gudelines.pdf",
        "expected_pages": 33,
        "prefix": "GEN",
        "applies_to": "both_models",
        "coding_only": False,
    },
    "gpt_luna": {
        "filename": "gpt_luna_guidelines.pdf",
        "display_name": "Prompt GPT 5.6 Luna.pdf",
        "expected_pages": 47,
        "prefix": "GPT",
        "applies_to": "gpt_luna",
        "coding_only": False,
    },
    "gemini_flash": {
        "filename": "gemini_flash_guidelines.pdf",
        "display_name": "Gemini Flash 3.8.pdf",
        "expected_pages": 43,
        "prefix": "GEM",
        "applies_to": "gemini_flash",
        "coding_only": False,
    },
    "coding": {
        "filename": "coding_guidelines.pdf",
        "display_name": "Coding Prompt.pdf",
        "expected_pages": 40,
        "prefix": "CODE",
        "applies_to": "coding_mode",
        "coding_only": True,
    },
}

# The knowledge base is cached once per Python process. This avoids re-reading
# and re-indexing the PDFs for every chat message.
_KB_CACHE: Optional["KnowledgeBase"] = None
_KB_ERROR: Optional[str] = None
_KB_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GuidelineChunk:
    """One complete, searchable PDF page with stable source metadata."""

    identifier: str
    source_key: str
    source_name: str
    page: int
    section: str
    heading: str
    text: str
    categories: frozenset[str]
    terms: tuple[str, ...]


@dataclass(frozen=True)
class SourceDocument:
    """Metadata and complete extracted text for one source PDF."""

    source_key: str
    source_name: str
    path: Path
    page_count: int
    full_text: str
    chunks: tuple[GuidelineChunk, ...]


@dataclass(frozen=True)
class KnowledgeBase:
    """Complete in-memory index for every supplied source document."""

    documents: dict[str, SourceDocument]
    chunks: tuple[GuidelineChunk, ...]
    total_pages: int


class KnowledgeBaseError(RuntimeError):
    """Raised when the authoritative local PDF knowledge base is incomplete."""


class OllamaServiceError(RuntimeError):
    """Raised for local Ollama connectivity, model, timeout, or response errors."""


class ModelResponseError(RuntimeError):
    """Raised when the local model returns malformed or incomplete JSON."""


class PromptIntegrityError(RuntimeError):
    """Raised when a generated prompt fails deterministic preservation checks."""


# ---------------------------------------------------------------------------
# Text extraction and indexing
# ---------------------------------------------------------------------------

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "i", "if", "in", "into", "is", "it", "its", "may",
    "me", "my", "of", "on", "or", "our", "should", "so", "that",
    "the", "their", "them", "there", "these", "this", "those", "to",
    "up", "us", "was", "we", "were", "what", "when", "where", "which",
    "who", "will", "with", "would", "you", "your",
}

CATEGORY_TERMS: dict[str, tuple[str, ...]] = {
    "coding": (
        "code", "coding", "program", "programming", "software", "developer",
        "debug", "bug", "function", "algorithm", "repository",
        "frontend", "backend", "database", "api", "flask", "python", "java",
        "javascript", "html", "css", "sql", "architecture", "implementation",
        "compile", "runtime", "framework", "library", "git", "github", "test",
    ),
    "research": (
        "research", "source", "sources", "citation", "citations", "reference",
        "references", "evidence", "study", "studies", "paper", "publication",
        "statistics", "official", "journal", "peer-reviewed", "market research",
    ),
    "learning": (
        "teach", "learn", "learning", "education", "academic", "student", "exam",
        "syllabus", "notes", "lesson", "course", "beginner", "explain", "tutorial",
    ),
    "mathematics": (
        "math", "mathematics", "mathematical", "equation", "algebra", "calculus",
        "geometry", "proof", "theorem", "probability", "statistics", "calculate",
        "calculation", "numerical", "formula", "derivation", "units",
    ),
    "data_analysis": (
        "data", "dataset", "analysis", "analytics", "analyse", "analyze", "chart",
        "graph", "visualization", "visualisation", "regression", "trend", "csv",
        "spreadsheet", "correlation", "forecast", "metrics",
    ),
    "business_finance": (
        "business", "finance", "financial", "budget", "revenue", "profit", "cost",
        "investment", "market", "customer", "pricing", "sales", "strategy", "roi",
    ),
    "writing": (
        "write", "writing", "rewrite", "draft", "email", "letter", "post", "article",
        "essay", "story", "description", "copy", "content", "tone", "proofread",
    ),
    "multimodal": (
        "image", "photo", "picture", "screenshot", "video", "audio", "pdf", "document",
        "attachment", "visual", "figure", "diagram", "scan", "transcribe", "transcription",
    ),
    "agent_tools": (
        "tool", "tools", "agent", "agentic", "browser", "search", "function calling",
        "automation", "execute", "execution", "permission", "approval", "workflow",
        "api", "endpoint", "external action",
    ),
    "tts": (
        "tts", "text to speech", "text-to-speech", "speech synthesis", "voiceover",
        "spoken output", "pronunciation", "voice generation",
    ),
    "verification": (
        "verify", "verification", "validate", "validation", "double check", "cross-check",
        "fact check", "fact-check", "accuracy", "correctness", "test", "testing", "audit",
    ),
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.+#_\-/][a-z0-9]+)*", re.IGNORECASE)
PART_PATTERN = re.compile(r"^\s*PART\s+\d+\s*[—–-]\s*(.*)$", re.IGNORECASE)
RULE_PATTERN = re.compile(r"^\s*(?:RULE|Rule)\s+(\d+)\s*[:—–.-]\s*(.*)$")


def _normalise_for_search(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return value.lower()


def _tokenise(value: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(_normalise_for_search(value))
        if token.lower() not in STOP_WORDS and len(token) > 1
    ]


def _detect_categories(text: str) -> frozenset[str]:
    lowered = _normalise_for_search(text)
    found: set[str] = set()
    for category, phrases in CATEGORY_TERMS.items():
        for phrase in phrases:
            phrase_norm = _normalise_for_search(phrase)
            # Match whole words/phrases. Plain substring matching can mistake
            # "photosynthesis" for "photo" or "capital" for "api".
            pattern = r"(?<![a-z0-9])" + re.escape(phrase_norm) + r"(?![a-z0-9])"
            if re.search(pattern, lowered):
                found.add(category)
                break
    if not found:
        found.add("general")
    return frozenset(found)


def _page_heading(lines: list[str], page_number: int) -> str:
    """Return the most useful heading visible on a page."""
    clean_lines = [re.sub(r"\s+", " ", line).strip(" \t\u2022-•") for line in lines]
    clean_lines = [line for line in clean_lines if line]

    for index, line in enumerate(clean_lines):
        if PART_PATTERN.match(line):
            title = PART_PATTERN.match(line).group(1).strip()  # type: ignore[union-attr]
            if index + 1 < len(clean_lines) and len(title) < 35:
                next_line = clean_lines[index + 1]
                if len(next_line) < 90 and not RULE_PATTERN.match(next_line):
                    title = f"{title} {next_line}".strip()
            return f"PART: {title or 'Untitled section'}"

    for line in clean_lines:
        match = RULE_PATTERN.match(line)
        if match:
            return f"RULE {match.group(1)}: {match.group(2).strip()}".rstrip(": ")

    for line in clean_lines[:8]:
        if len(line) >= 5 and len(line) <= 110:
            return line
    return f"Page {page_number}"


def _load_one_pdf(source_key: str, config: dict[str, Any]) -> SourceDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency installation issue
        raise KnowledgeBaseError(
            "The 'pypdf' package is not installed. Install the dependencies from requirements.txt."
        ) from exc

    pdf_path = KNOWLEDGE_DIR / str(config["filename"])
    if not pdf_path.is_file():
        raise KnowledgeBaseError(
            f"Required guideline PDF is missing: {pdf_path}. "
            f"Place the file at knowledge/{config['filename']}."
        )

    try:
        reader = PdfReader(str(pdf_path), strict=False)
        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
            except Exception as exc:
                raise KnowledgeBaseError(
                    f"The PDF '{config['display_name']}' is encrypted and cannot be read."
                ) from exc
            if not decrypt_result:
                raise KnowledgeBaseError(
                    f"The PDF '{config['display_name']}' is encrypted and cannot be read."
                )

        actual_pages = len(reader.pages)
        expected_pages = int(config["expected_pages"])
        if actual_pages != expected_pages:
            raise KnowledgeBaseError(
                f"The PDF '{config['display_name']}' has {actual_pages} pages; "
                f"the expected source contains {expected_pages} pages. "
                "Check that the complete, correct PDF was copied into knowledge/."
            )

        page_texts: list[str] = []
        extraction_failures: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                extracted = page.extract_text() or ""
            except Exception:
                extracted = ""
                extraction_failures.append(page_number)
            page_texts.append(extracted.strip())

        nonempty_pages = sum(bool(text.strip()) for text in page_texts)
        empty_pages = actual_pages - nonempty_pages
        if extraction_failures:
            raise KnowledgeBaseError(
                f"Text extraction failed on page(s) {extraction_failures} of "
                f"'{config['display_name']}'. The knowledge base is not considered complete."
            )
        if nonempty_pages == 0 or empty_pages > max(2, math.floor(actual_pages * 0.15)):
            raise KnowledgeBaseError(
                f"The PDF '{config['display_name']}' appears substantially incomplete: "
                f"only {nonempty_pages} of {actual_pages} pages contain extractable text."
            )

        chunks: list[GuidelineChunk] = []
        active_section = "Document introduction"
        complete_pages: list[str] = []
        prefix = str(config["prefix"])

        for page_number, page_text in enumerate(page_texts, start=1):
            lines = page_text.splitlines()
            for line in lines:
                part_match = PART_PATTERN.match(line.strip())
                if part_match:
                    active_section = _page_heading(lines, page_number).replace("PART: ", "")
                    break

            heading = _page_heading(lines, page_number)
            if not page_text:
                # Empty pages remain explicitly represented in the index, rather than
                # being discarded. Their text is empty because pypdf extracted no text.
                page_text = "[No extractable text on this page.]"

            # Stable internal page identifier. Rule headings and section metadata
            # accompany the ID so that the UI/API can cite the precise retrieved page.
            identifier = f"{prefix}-P{page_number:03d}"
            categories = _detect_categories(f"{active_section}\n{heading}\n{page_text}")
            terms = tuple(_tokenise(f"{active_section}\n{heading}\n{page_text}"))
            chunk = GuidelineChunk(
                identifier=identifier,
                source_key=source_key,
                source_name=str(config["display_name"]),
                page=page_number,
                section=active_section,
                heading=heading,
                text=page_text,
                categories=categories,
                terms=terms,
            )
            chunks.append(chunk)
            complete_pages.append(f"[PAGE {page_number}]\n{page_text}")

        full_text = "\n\n".join(complete_pages)
        if len(full_text.strip()) < 1000:
            raise KnowledgeBaseError(
                f"The PDF '{config['display_name']}' produced too little readable text."
            )

        return SourceDocument(
            source_key=source_key,
            source_name=str(config["display_name"]),
            path=pdf_path,
            page_count=actual_pages,
            full_text=full_text,
            chunks=tuple(chunks),
        )
    except KnowledgeBaseError:
        raise
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not read '{config['display_name']}': {type(exc).__name__}: {exc}"
        ) from exc


def _load_knowledge_base() -> KnowledgeBase:
    """Load all four PDFs and retain every extracted page in memory."""
    global _KB_CACHE, _KB_ERROR
    with _KB_LOCK:
        if _KB_CACHE is not None:
            return _KB_CACHE
        if _KB_ERROR is not None:
            raise KnowledgeBaseError(_KB_ERROR)

        documents: dict[str, SourceDocument] = {}
        try:
            for source_key, config in SOURCE_CONFIG.items():
                documents[source_key] = _load_one_pdf(source_key, config)
            all_chunks = tuple(
                chunk
                for document in documents.values()
                for chunk in document.chunks
            )
            total_pages = sum(document.page_count for document in documents.values())
            expected_total = sum(int(item["expected_pages"]) for item in SOURCE_CONFIG.values())
            if total_pages != expected_total:
                raise KnowledgeBaseError(
                    f"Knowledge-base page count mismatch: loaded {total_pages}, "
                    f"expected {expected_total}."
                )
            _KB_CACHE = KnowledgeBase(
                documents=documents,
                chunks=all_chunks,
                total_pages=total_pages,
            )
            return _KB_CACHE
        except KnowledgeBaseError as exc:
            _KB_ERROR = str(exc)
            raise


def reset_knowledge_base_cache() -> None:
    """Clear the in-memory PDF index (useful after replacing a PDF in development)."""
    global _KB_CACHE, _KB_ERROR
    with _KB_LOCK:
        _KB_CACHE = None
        _KB_ERROR = None


# ---------------------------------------------------------------------------
# Task classification and guideline retrieval
# ---------------------------------------------------------------------------


def _infer_task_categories(text: str) -> set[str]:
    """Infer broad task categories from the actual request and conversation."""
    categories = set(_detect_categories(text))
    # A generic request can still require universal prompt-design rules.
    if "general" in categories:
        categories.remove("general")
    return categories


def _bm25_scores(chunks: Iterable[GuidelineChunk], query: str) -> dict[str, float]:
    """Small, dependency-free BM25 implementation over indexed PDF pages."""
    chunk_list = list(chunks)
    query_terms = _tokenise(query)
    if not chunk_list or not query_terms:
        return {chunk.identifier: 0.0 for chunk in chunk_list}

    document_frequencies: Counter[str] = Counter()
    term_counts: dict[str, Counter[str]] = {}
    lengths: dict[str, int] = {}
    for chunk in chunk_list:
        counts = Counter(chunk.terms)
        term_counts[chunk.identifier] = counts
        lengths[chunk.identifier] = sum(counts.values())
        document_frequencies.update(set(counts))

    average_length = sum(lengths.values()) / max(1, len(lengths))
    total_docs = len(chunk_list)
    scores: dict[str, float] = {}
    k1, b = 1.5, 0.75
    query_counts = Counter(query_terms)

    for chunk in chunk_list:
        counts = term_counts[chunk.identifier]
        doc_length = lengths[chunk.identifier]
        score = 0.0
        for term, query_frequency in query_counts.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequencies.get(term, 0)
            inverse_frequency = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * doc_length / max(1.0, average_length))
            score += inverse_frequency * frequency * (k1 + 1) / denominator
            if query_frequency > 1:
                score *= 1 + min(0.25, (query_frequency - 1) * 0.05)
        # Page title/section matches are more important than incidental body matches.
        heading_terms = set(_tokenise(f"{chunk.section} {chunk.heading}"))
        score += 1.25 * len(set(query_terms) & heading_terms)
        scores[chunk.identifier] = score
    return scores


def _query_category_score(chunk: GuidelineChunk, categories: set[str]) -> int:
    return len(chunk.categories.intersection(categories))


def _is_mandatory_page(chunk: GuidelineChunk) -> bool:
    text = _normalise_for_search(f"{chunk.section} {chunk.heading} {chunk.text[:1600]}")
    markers = (
        "compulsory double-verification",
        "double-verification protocol",
        "double verification protocol",
        "golden rules",
        "final pre-submission checklist",
        "quality-control scorecard",
        "final quality-control scorecard",
    )
    return any(marker in text for marker in markers)


def _select_source_chunks(
    chunks: list[GuidelineChunk],
    scores: dict[str, float],
    categories: set[str],
    limit: int,
    mandatory_limit: int = 8,
) -> list[GuidelineChunk]:
    """Pick relevant pages, ensuring core protocol/checklist pages are considered."""
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            scores.get(chunk.identifier, 0.0)
            + 0.9 * _query_category_score(chunk, categories),
            -chunk.page,
        ),
        reverse=True,
    )
    selected: dict[str, GuidelineChunk] = {}

    mandatory = [chunk for chunk in chunks if _is_mandatory_page(chunk)]
    for chunk in mandatory[:mandatory_limit]:
        selected[chunk.identifier] = chunk

    for chunk in ranked:
        if len(selected) >= limit:
            break
        if scores.get(chunk.identifier, 0.0) <= 0 and not _query_category_score(chunk, categories):
            continue
        selected[chunk.identifier] = chunk

    # If a short or unusual prompt has no lexical matches, retrieve the most
    # broadly foundational pages instead of silently providing no guidelines.
    if not selected and ranked:
        for chunk in ranked[: min(5, limit)]:
            selected[chunk.identifier] = chunk

    return sorted(selected.values(), key=lambda chunk: (chunk.page, chunk.identifier))




def _filter_model_specific_pages(
    chunks: list[GuidelineChunk],
    target_model: str,
    categories: set[str],
) -> list[GuidelineChunk]:
    """Remove model-specific/TTS subsections that do not apply to this turn.

    Coding Prompt.pdf contains a short section comparing GPT-5.6 Luna and
    Gemini 3.8 Flash on pages 21-23. Those pages are filtered by selected
    profile so the two model-specific frameworks are never mixed. Gemini's
    dedicated TTS section is also excluded unless this is a TTS task.
    """
    filtered: list[GuidelineChunk] = []
    for chunk in chunks:
        if chunk.source_key == "coding" and chunk.page in {21, 22, 23}:
            if target_model == "gpt_luna":
                if chunk.page == 23:
                    continue  # This page contains the Gemini-specific continuation.
                if chunk.page == 22:
                    marker = re.search(r"(?im)^\s*B\.\s*GEMINI\s+3\.8\s+FLASH\s*$", chunk.text)
                    if marker:
                        trimmed = chunk.text[:marker.start()].strip()
                        if not trimmed:
                            continue
                        chunk = GuidelineChunk(
                            identifier=chunk.identifier, source_key=chunk.source_key,
                            source_name=chunk.source_name, page=chunk.page,
                            section=chunk.section, heading=chunk.heading, text=trimmed,
                            categories=_detect_categories(f"{chunk.section} {chunk.heading} {trimmed}"),
                            terms=tuple(_tokenise(f"{chunk.section} {chunk.heading} {trimmed}")),
                        )
            else:
                if chunk.page in {21, 22}:
                    continue  # Page 21-22 contain the GPT-specific section/transition.

        if chunk.source_key == "gemini_flash" and "tts" not in categories:
            if "tts" in chunk.section.lower() or "text-to-speech" in _normalise_for_search(chunk.text[:1200]):
                continue
        filtered.append(chunk)
    return filtered


def _retrieve_guidelines(
    kb: KnowledgeBase,
    message: str,
    history: list[dict[str, str]],
    target_model: str,
    coding_mode: bool,
) -> tuple[list[GuidelineChunk], list[str], set[str]]:
    """Retrieve applicable universal, selected-model and optional coding pages."""
    query_parts = [message]
    query_parts.extend(item["content"] for item in history if item["role"] == "user")
    query = "\n".join(query_parts)
    categories = _infer_task_categories(query)
    all_scores = _bm25_scores(kb.chunks, query)

    selected_chunks: list[GuidelineChunk] = []
    applicable_chunks = _filter_model_specific_pages(list(kb.chunks), target_model, categories)
    source_limits = {
        "general": 16,
        target_model: 16,
    }
    if coding_mode:
        source_limits["coding"] = 20

    # General and the selected model profile are always eligible. The other
    # model's document is never passed into this request.
    for source_key, limit in source_limits.items():
        source_chunks = [chunk for chunk in applicable_chunks if chunk.source_key == source_key]
        selected_chunks.extend(
            _select_source_chunks(source_chunks, all_scores, categories, limit)
        )

    # Task/category coverage: pull in high-scoring pages tagged for categories
    # explicitly detected in the task, within each selected source's limit.
    # The per-source ranking above already incorporates category matches; this
    # pass adds pages where the category match is strong but lexical overlap is low.
    for source_key, limit in source_limits.items():
        source_chunks = [chunk for chunk in applicable_chunks if chunk.source_key == source_key]
        for category in categories:
            matches = [chunk for chunk in source_chunks if category in chunk.categories]
            matches.sort(key=lambda c: all_scores.get(c.identifier, 0.0), reverse=True)
            for chunk in matches[:2]:
                if chunk.identifier not in {item.identifier for item in selected_chunks}:
                    current_count = sum(item.source_key == source_key for item in selected_chunks)
                    if current_count < limit + 4:
                        selected_chunks.append(chunk)

    # Keep source pages unique and order them predictably for the model.
    deduplicated = {chunk.identifier: chunk for chunk in selected_chunks}
    ordered = sorted(
        deduplicated.values(),
        key=lambda c: (list(SOURCE_CONFIG).index(c.source_key), c.page),
    )

    identifiers = [chunk.identifier for chunk in ordered]
    return ordered, identifiers, categories


def _format_guideline_context(chunks: list[GuidelineChunk]) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        categories = ", ".join(sorted(chunk.categories))
        blocks.append(
            f"[GUIDELINE_ID: {chunk.identifier}]\n"
            f"[SOURCE: {chunk.source_name} | PAGE: {chunk.page} | "
            f"SECTION: {chunk.section} | HEADING: {chunk.heading} | "
            f"CATEGORIES: {categories}]\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def _format_conversation_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "[No previous conversation messages.]"
    blocks = []
    for index, item in enumerate(history, start=1):
        blocks.append(f"[{index}] {item['role'].upper()}:\n{item['content']}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Conversation normalization
# ---------------------------------------------------------------------------


def _normalise_history(history: Any) -> list[dict[str, str]]:
    """Accept the documented role/content messages and common frontend variants."""
    if history is None:
        return []
    if not isinstance(history, list):
        raise ValueError("history must be a list of conversation messages.")

    normalized: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role in {"human", "user"}:
            role = "user"
        elif role in {"assistant", "ai", "bot"}:
            role = "assistant"
        else:
            continue
        content = item.get("content", item.get("message", item.get("text", "")))
        if not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


# ---------------------------------------------------------------------------
# Ollama communication and JSON validation
# ---------------------------------------------------------------------------


def _local_model_name() -> str:
    model_name = os.environ.get("LOCAL_LLM_MODEL", "").strip()
    if not model_name:
        raise OllamaServiceError(
            "LOCAL_LLM_MODEL is not set. Set it to the name of a model already installed in Ollama."
        )
    return model_name


def _ollama_chat(
    messages: list[dict[str, str]],
    *,
    temperature: Optional[float] = None,
) -> str:
    """Call only the local Ollama chat endpoint; no cloud fallback is permitted."""
    model_name = _local_model_name()
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": OLLAMA_TEMPERATURE if temperature is None else temperature,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }
    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaServiceError(
            "Cannot connect to the local Ollama service at 127.0.0.1:11434. "
            "Start Ollama and try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaServiceError(
            f"The local AI model did not respond within {OLLAMA_TIMEOUT_SECONDS:g} seconds. "
            "Try again or use a smaller local model."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaServiceError(f"Local Ollama request failed: {exc}") from exc

    if not response.ok:
        detail = _response_error_detail(response)
        if response.status_code == 404 or (
            "model" in detail.lower()
            and any(term in detail.lower() for term in ("not found", "does not exist", "pull"))
        ):
            raise OllamaServiceError(
                f"The local Ollama model '{model_name}' may not be installed or the chat endpoint "
                f"is unavailable. {detail}".strip()
            )
        raise OllamaServiceError(
            f"Ollama returned HTTP {response.status_code}. {detail}".strip()
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaServiceError("Ollama returned a non-JSON HTTP response.") from exc

    if not isinstance(data, dict):
        raise OllamaServiceError("Ollama returned an unexpected response structure.")
    if data.get("error"):
        error_text = str(data["error"])
        if "model" in error_text.lower() and any(
            term in error_text.lower() for term in ("not found", "does not exist", "pull")
        ):
            raise OllamaServiceError(
                f"The local Ollama model '{model_name}' is not installed. "
                "Install the model locally and set LOCAL_LLM_MODEL to its exact name."
            )
        raise OllamaServiceError(f"Ollama error: {error_text}")

    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaServiceError("The local AI returned an empty response.")
    if data.get("done") is False:
        raise OllamaServiceError("The local AI response did not finish generating.")
    return content.strip()


def _response_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
        return str(data.get("error", "")) if isinstance(data, dict) else ""
    except Exception:
        return response.text[:500].strip()


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Parse strict JSON, tolerating a surrounding Markdown code fence."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Find the first complete JSON object, accounting for quoted braces.
        start = cleaned.find("{")
        if start < 0:
            raise ModelResponseError("The local AI did not return a JSON object.")
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
        if end is None:
            raise ModelResponseError("The local AI returned incomplete JSON.")
        try:
            parsed = json.loads(cleaned[start:end])
        except json.JSONDecodeError as exc:
            raise ModelResponseError("The local AI returned malformed JSON.") from exc

    if not isinstance(parsed, dict):
        raise ModelResponseError("The local AI JSON response must be an object.")
    return parsed


def _as_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ModelResponseError(f"The local AI returned an invalid '{field_name}' field.")
    return [str(item).strip() for item in value if isinstance(item, (str, int, float)) and str(item).strip()]


def _validate_suggested_settings(value: Any, target_model: str) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return None
    if target_model == "gpt_luna":
        effort = value.get("reasoning_effort")
        allowed = {"none", "low", "medium", "high", "xhigh", "max"}
        if effort not in allowed:
            return None
        return {"reasoning_effort": effort}
    level = value.get("thinking_level")
    if level not in {"LOW", "MEDIUM", "HIGH"}:
        return None
    return {"thinking_level": level}


def _normalise_model_result(
    result: dict[str, Any],
    target_model: str,
    applied_guidelines: list[str],
) -> dict[str, Any]:
    status = result.get("status")
    assistant_message = result.get("assistant_message")
    if not isinstance(assistant_message, str) or not assistant_message.strip():
        assistant_message = ""

    assumptions = _as_string_list(result.get("assumptions", []), "assumptions")
    # Never trust rule IDs produced by the model. Only backend-selected IDs are returned.
    verified_ids = list(dict.fromkeys(applied_guidelines))
    suggested_settings = _validate_suggested_settings(
        result.get("suggested_settings"), target_model
    )

    if status == "needs_clarification":
        questions = _as_string_list(result.get("questions", []), "questions")
        if not questions:
            raise ModelResponseError(
                "The local AI marked the request as needing clarification but supplied no questions."
            )
        return {
            "status": "needs_clarification",
            "assistant_message": assistant_message or "I need one clarification before refining your prompt.",
            "questions": questions,
            "final_prompt": None,
            "assumptions": assumptions,
            "applied_guidelines": verified_ids,
            "suggested_settings": suggested_settings,
        }

    if status == "prompt_ready":
        final_prompt = result.get("final_prompt")
        if not isinstance(final_prompt, str) or not final_prompt.strip():
            raise ModelResponseError("The local AI marked the prompt ready but returned no final_prompt.")
        return {
            "status": "prompt_ready",
            "assistant_message": assistant_message or "Your refined prompt is ready.",
            "questions": [],
            "final_prompt": final_prompt.strip(),
            "assumptions": assumptions,
            "applied_guidelines": verified_ids,
            "suggested_settings": suggested_settings,
        }

    raise ModelResponseError(
        "The local AI returned an unsupported status. Expected 'needs_clarification' or 'prompt_ready'."
    )


# ---------------------------------------------------------------------------
# Prompt integrity checks
# ---------------------------------------------------------------------------


def _user_messages(history: list[dict[str, str]], current_message: str) -> list[str]:
    messages = [item["content"] for item in history if item["role"] == "user"]
    messages.append(current_message)
    return messages


def _protected_literals(texts: Iterable[str]) -> list[str]:
    """Extract literal values that should survive prompt rewriting unchanged."""
    protected: list[str] = []
    for text in texts:
        # Backtick-delimited code, paths and exact literals.
        protected.extend(match.group(1).strip() for match in re.finditer(r"`([^`\n]+)`", text) if match.group(1).strip())
        # Double-quoted phrases and typographic double quotes.
        protected.extend(
            match.group(1).strip()
            for match in re.finditer(r'["“]([^"”\n]{2,})["”]', text)
            if match.group(1).strip()
        )
        # URLs must not be replaced or fabricated.
        urls = re.findall(r"https?://[^\s<>\])}]+", text, flags=re.IGNORECASE)
        protected.extend(url.rstrip(".,;:!?") for url in urls)
        # Explicit filenames and common path-like values.
        protected.extend(
            re.findall(
                r"(?<![\w/])(?:[\w.-]+/)+[\w.-]+|\b[\w.-]+\.(?:py|js|html|css|json|pdf|txt|md|sql|csv|xlsx?)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        # Preserve numeric literals (including decimal/version/percentage values).
        protected.extend(
            re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:[.,]\d+)*(?:%|[A-Za-z]{1,5})?", text)
        )
    # Keep insertion order and avoid redundant checks.
    result: list[str] = []
    seen: set[str] = set()
    for literal in protected:
        literal = literal.strip()
        key = re.sub(r"\s+", " ", literal).casefold()
        if literal and key not in seen:
            seen.add(key)
            result.append(literal)
    return result


def _literal_is_preserved(literal: str, final_prompt: str) -> bool:
    def normalise(value: str) -> str:
        value = unicodedata.normalize("NFKC", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value.casefold()

    return normalise(literal) in normalise(final_prompt)


def _ensure_double_verification_instruction(final_prompt: str) -> str:
    """Ensure every delivered prompt contains the required two-review instruction."""
    lowered = _normalise_for_search(final_prompt)
    patterns = (
        r"two\s+(?:separate|independent|distinct)\s+(?:reviews|checks|verification passes)",
        r"review.{0,80}twice",
        r"first review.{0,500}second review",
        r"double[ -]verification",
        r"double[ -]check",
    )
    if any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in patterns):
        return final_prompt
    addition = (
        "\n\n## FINAL DOUBLE-VERIFICATION\n"
        "Before delivering the result, perform two distinct reviews. First, compare the result "
        "against the original request and every clarification to confirm that the objective, "
        "requirements, constraints, names, numbers, dates, and exclusions are preserved. "
        "Second, independently check for omissions, contradictions, ambiguity, unsupported "
        "additions, and errors. Correct any issues found and honestly disclose anything "
        "that could not be verified. Do not present unperformed tests or checks as completed."
    )
    return final_prompt.rstrip() + addition


def _check_prompt_integrity(original_texts: list[str], final_prompt: str) -> None:
    missing = [
        literal
        for literal in _protected_literals(original_texts)
        if not _literal_is_preserved(literal, final_prompt)
    ]
    if missing:
        sample = ", ".join(repr(item) for item in missing[:8])
        more = " ..." if len(missing) > 8 else ""
        raise PromptIntegrityError(
            "The generated prompt did not preserve all protected original literals: "
            f"{sample}{more}. No prompt was returned because preserving the user's details is mandatory."
        )


# ---------------------------------------------------------------------------
# Main prompt-generation and two-pass verification workflow
# ---------------------------------------------------------------------------


def _main_system_prompt(
    target_model: str,
    coding_mode: bool,
    guideline_context: str,
    guideline_ids: list[str],
    categories: set[str],
) -> str:
    model_name = TARGET_MODEL_NAMES[target_model]
    code_state = "ON: relevant coding rules are included" if coding_mode else "OFF: coding rulebook is excluded"
    category_text = ", ".join(sorted(categories)) if categories else "general task"
    ids_text = ", ".join(guideline_ids)

    if target_model == "gpt_luna":
        model_profile = """TARGET PROFILE: GPT-5.6 Luna.
Apply only GPT-5.6 Luna-specific guidance alongside the universal guidance. Use an outcome-first approach: establish the intended result, relevant context, hard constraints, preservation requirements, success criteria, permissions/approval boundaries, and stop conditions. Avoid unnecessary micromanagement and repetition. If a reasoning-effort suggestion is useful, return it separately as one of: none, low, medium, high, xhigh, max. Do not claim that prompt text changes an API setting."""
    else:
        model_profile = """TARGET PROFILE: Gemini 3.8 Flash.
Apply only Gemini 3.8 Flash-specific guidance alongside the universal guidance. Use direct precise instructions, one consistent structure, explicit definitions for material ambiguity, critical rules near the beginning, explicit verbosity/output expectations, appropriate context placement, clearly labelled multimodal inputs, and appropriate tool/recovery/stop policies. For large context, put supporting context before the final specific task where appropriate. If a thinking-level suggestion is useful, return it separately as LOW, MEDIUM, or HIGH. Never suggest MINIMAL. Do not claim that prompt text changes an API setting."""

    return f"""You are PromptForge Offline, a careful prompt-engineering assistant. Your job is to improve the user's prompt, not to answer the underlying task.

ABSOLUTE PRIORITY: PRESERVE THE USER'S ORIGINAL MEANING.
Never change, weaken, omit, invent, or silently reinterpret the user's objective, requirements, exclusions, names, numerical values, dates, technical specifications, output format, or constraints. Distinguish MUST/SHOULD/MAY/MUST NOT. Do not convert optional preferences into requirements or mandatory requirements into preferences. Preserve all important details from the original request and all user clarification answers in the conversation. Do not add new features or requirements. Optional suggestions, if truly useful, must be clearly separated and labelled OPTIONAL; do not insert them into the main prompt without user approval.

APPLICATION SETTINGS:
- Target model profile: {model_name} ({target_model}).
- Coding Guidelines: {code_state}.
- Task categories detected by the backend: {category_text}.
- Local inference is performed by Ollama; do not claim to be the actual cloud target model.

{model_profile}

KNOWLEDGE-BASE POLICY:
The backend has fully loaded and indexed all four authoritative PDFs, then retrieved the relevant pages for this specific task. The complete text of the retrieved pages appears below. Use all applicable information in these retrieved pages. Do not invent rules, source facts, or guideline IDs. The backend will return the applied guideline IDs; never create IDs yourself. Do not claim that an unprovided page was inspected. Coding Prompt.pdf pages are supplied only when Coding Guidelines is ON. The unselected model-specific PDF must not be used.

CLARIFICATION POLICY:
First reconstruct the user's original objective from the full conversation. Identify task, outcome, context, requirements, constraints, exclusions, audience, output expectations, missing information, ambiguity, and contradictions. Ask a focused clarification question only when missing or conflicting information materially affects the correctness or usefulness of the prompt. Do not ask for information already provided. Ask the smallest number of high-value questions, preferably one at a time. For minor gaps, proceed with a reasonable assumption and disclose it. If a material contradiction exists, ask before finalizing. If enough information is already available, generate the prompt without unnecessary questions.

PROMPT CONSTRUCTION:
Use the universal foundation ROLE + CONTEXT + OBJECTIVE + REQUIREMENTS + CONSTRAINTS + EXECUTION + VERIFICATION + OUTPUT, but include only sections useful for this task. Complex tasks may need headings such as ROLE, CONTEXT, OBJECTIVE, AVAILABLE MATERIALS, REQUIREMENTS, TECHNICAL ENVIRONMENT, CONSTRAINTS, SOURCES, EXECUTION, EDGE CASES, SUCCESS CRITERIA, OUTPUT FORMAT, VERIFICATION, and STOP CONDITIONS. Use one consistent structure. Prefer precise, measurable, direct instructions. Avoid unnecessary length, decorative role-play, repetitive instructions, and irrelevant rules. Preserve code details and programming terminology even when Coding Guidelines is OFF.

COMPULSORY DOUBLE-VERIFICATION:
Every generated prompt must contain an appropriate instruction for the target AI to perform two distinct reviews for important work: (1) verify original meaning, correctness, completeness, required details, exclusions, assumptions, and requirements; (2) independently check for contradictions, ambiguity, repetition, unsupported additions, omissions, and formatting/technical errors. Correct identified issues and report unverified areas honestly. Do not request private chain-of-thought; ask for concise findings or verification results. The prompt-engine itself will also run two separate verification passes before delivery.

RESPONSE RULES:
Return exactly one valid JSON object, with no Markdown fence and no prose outside JSON. Use this schema:
{{
  "status": "needs_clarification" OR "prompt_ready",
  "assistant_message": "Brief message to the user",
  "questions": ["Question 1"],
  "final_prompt": null OR "Complete refined prompt text",
  "assumptions": ["Assumption, if any"],
  "suggested_settings": null OR {{"reasoning_effort": "medium"}} OR {{"thinking_level": "MEDIUM"}}
}}
For needs_clarification: provide at least one material question and set final_prompt to null. For prompt_ready: questions must be an empty array and final_prompt must contain the complete ready-to-copy prompt. suggested_settings must follow the selected model's supported values or be null. Do not include applied_guidelines; the backend supplies that field from actual retrieval IDs.

RETRIEVED GUIDELINE IDENTIFIERS (backend controlled): {ids_text}

BEGIN RETRIEVED PDF GUIDELINES
{guideline_context}
END RETRIEVED PDF GUIDELINES
"""


def _conversation_messages(history: list[dict[str, str]], current_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": current_message})
    return messages


def _generate_initial_result(
    target_model: str,
    coding_mode: bool,
    history: list[dict[str, str]],
    message: str,
    guideline_context: str,
    guideline_ids: list[str],
    categories: set[str],
) -> dict[str, Any]:
    system = _main_system_prompt(
        target_model, coding_mode, guideline_context, guideline_ids, categories
    )
    conversation = _conversation_messages(history, message)
    messages = [{"role": "system", "content": system}] + conversation
    raw = _ollama_chat(messages)
    parsed = _extract_json_object(raw)
    return _normalise_model_result(parsed, target_model, guideline_ids)


def _verification_system_prompt(
    stage: int,
    target_model: str,
    guideline_context: str,
    guideline_ids: list[str],
) -> str:
    if stage == 1:
        task = "FIRST REVIEW — SEMANTIC FIDELITY AND COMPLETENESS. Compare the candidate with every user message and clarification answer. Check the objective, deliverable, requirements, priorities, constraints, prohibitions, names, numbers, dates, technologies, output format, assumptions, and success criteria. Correct omissions or distortions. Do not add new intent."
    else:
        task = "SECOND REVIEW — INDEPENDENT ERROR DETECTION. Start a fresh independent review; do not assume the first review was correct. Search for altered meaning, contradictions, ambiguous instructions, repeated or unnecessary content, unsupported additions, missing details, incorrect technical terms, mismatched output formats, and any protected literal that disappeared. Correct every issue that can be corrected without changing intent."

    return f"""You are performing verification stage {stage} for PromptForge Offline. You are reviewing a prompt, not answering the user's underlying task.

{task}

NON-NEGOTIABLE RULES:
- The original user's intent and all clarification answers are the source of truth.
- Never add requirements, features, facts, technologies, dates, or constraints not supplied or clearly implied by the user.
- Preserve all explicit prohibitions and mandatory/optional priority distinctions.
- Preserve numeric values, date/time values, names, URLs, quoted literals, code identifiers, file paths, and technical specifications.
- Maintain the selected target profile: {TARGET_MODEL_NAMES[target_model]}. Do not import guidance from the other model's PDF.
- Use only the retrieved guideline identifiers actually supplied: {', '.join(guideline_ids)}.
- Do not expose private chain-of-thought. Return only the corrected prompt and a concise list of issues found.
- The prompt must retain a suitable two-review verification instruction for the target AI when appropriate.

Retrieved guidance for reference:
{guideline_context}

Return exactly this JSON object and no other text:
{{
  "final_prompt": "Complete corrected prompt",
  "issues_found": ["Issue identified, or an empty list"],
  "preservation_check": {{
    "objective_preserved": true,
    "requirements_preserved": true,
    "constraints_preserved": true,
    "numbers_and_names_preserved": true,
    "no_unsupported_additions": true
  }}
}}
If the candidate cannot be verified without making an unsupported assumption, keep the closest faithful version and explicitly list the unresolved issue. Never silently guess.
"""


def _run_verification_pass(
    stage: int,
    target_model: str,
    history: list[dict[str, str]],
    current_message: str,
    candidate_prompt: str,
    guideline_context: str,
    guideline_ids: list[str],
) -> tuple[str, list[str], list[str]]:
    system = _verification_system_prompt(
        stage, target_model, guideline_context, guideline_ids
    )
    user_context = (
        "COMPLETE USER CONVERSATION (user messages contain the original request and clarifications):\n"
        f"{_format_conversation_history(history)}\n\n"
        f"LATEST USER MESSAGE:\n{current_message}\n\n"
        f"CANDIDATE PROMPT TO REVIEW:\n{candidate_prompt}"
    )
    raw = _ollama_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user_context},
        ],
        temperature=0.1,
    )
    parsed = _extract_json_object(raw)
    final_prompt = parsed.get("final_prompt")
    if not isinstance(final_prompt, str) or not final_prompt.strip():
        raise ModelResponseError(f"Verification pass {stage} returned no valid final_prompt.")
    issues = _as_string_list(parsed.get("issues_found", []), "issues_found")
    check = parsed.get("preservation_check")
    required_checks = {
        "objective_preserved",
        "requirements_preserved",
        "constraints_preserved",
        "numbers_and_names_preserved",
        "no_unsupported_additions",
    }
    if not isinstance(check, dict) or not required_checks.issubset(check):
        raise ModelResponseError(
            f"Verification pass {stage} did not return the required preservation checks."
        )
    failed_checks = [key for key in required_checks if check.get(key) is not True]
    return final_prompt.strip(), issues, failed_checks


def _error_response(message: str) -> dict[str, Any]:
    return {"status": "error", "assistant_message": message}


def process_message(
    message: str,
    history: Any,
    target_model: str,
    coding_mode: bool,
) -> dict[str, Any]:
    """Process one chatbot turn and return the documented PromptForge response.

    Args:
        message: Latest user message.
        history: Earlier messages as dictionaries with role/content fields.
        target_model: Exactly ``gpt_luna`` or ``gemini_flash``.
        coding_mode: True to retrieve coding-rulebook pages; otherwise False.

    Returns:
        One of the exact response dictionaries documented in the project spec:
        ``needs_clarification``, ``prompt_ready``, or ``error``.
    """
    try:
        if not isinstance(message, str) or not message.strip():
            return _error_response("Please enter a non-empty prompt or clarification answer.")
        if not isinstance(target_model, str) or target_model not in VALID_TARGET_MODELS:
            return _error_response(
                "Invalid target_model. Use exactly 'gpt_luna' or 'gemini_flash'."
            )
        if not isinstance(coding_mode, bool):
            return _error_response("coding_mode must be a Boolean value: true or false.")

        normalized_history = _normalise_history(history)
        history_characters = sum(len(item["content"]) for item in normalized_history)
        total_conversation_characters = history_characters + len(message.strip())
        if total_conversation_characters > 40000:
            return _error_response(
                "This refinement conversation is too large to process reliably in the configured local context. "
                "Start a new prompt session or shorten the conversation; no earlier messages have "
                "been silently discarded. You may also increase OLLAMA_NUM_CTX if your local model supports it."
            )
        kb = _load_knowledge_base()  # Always validate all four expected source PDFs.

        retrieved_chunks, guideline_ids, categories = _retrieve_guidelines(
            kb=kb,
            message=message.strip(),
            history=normalized_history,
            target_model=target_model,
            coding_mode=coding_mode,
        )
        if not retrieved_chunks:
            return _error_response(
                "No applicable guideline pages could be retrieved from the local PDF knowledge base. "
                "Check the source PDFs and try again."
            )

        guideline_context = _format_guideline_context(retrieved_chunks)
        initial_result = _generate_initial_result(
            target_model=target_model,
            coding_mode=coding_mode,
            history=normalized_history,
            message=message.strip(),
            guideline_context=guideline_context,
            guideline_ids=guideline_ids,
            categories=categories,
        )

        if initial_result["status"] == "needs_clarification":
            # No final prompt exists yet, so final-prompt verification is deferred
            # until the clarification answer makes the task sufficiently clear.
            initial_result["applied_guidelines"] = guideline_ids
            return initial_result

        candidate = initial_result["final_prompt"]
        user_texts = _user_messages(normalized_history, message.strip())

        # Pass 1: compare intent, requirements and details with the full transcript.
        candidate, first_issues, first_failed_checks = _run_verification_pass(
            1, target_model, normalized_history, message.strip(), candidate,
            guideline_context, guideline_ids,
        )

        # Pass 2: a separate independent review for contradictions and omissions.
        candidate, second_issues, second_failed_checks = _run_verification_pass(
            2, target_model, normalized_history, message.strip(), candidate,
            guideline_context, guideline_ids,
        )

        # A failed preservation checklist is a hard stop: never expose a prompt
        # that either independent reviewer could not affirm as faithful.
        failed_verification_checks = list(dict.fromkeys(first_failed_checks + second_failed_checks))
        if failed_verification_checks:
            raise PromptIntegrityError(
                "The two-pass verification did not confirm these requirements: "
                + ", ".join(failed_verification_checks)
                + ". No final prompt was returned."
            )

        # The project specification makes this instruction compulsory for every
        # delivered prompt, even when the first draft did not include it.
        candidate = _ensure_double_verification_instruction(candidate)

        # A deterministic final guard protects literal values and exact identifiers.
        _check_prompt_integrity(user_texts, candidate)

        # Add verification findings to assumptions only when they are unresolved.
        assumptions = list(initial_result.get("assumptions", []))
        unresolved_issues = [
            item for item in (first_issues + second_issues)
            if any(word in item.lower() for word in ("unresolved", "cannot verify", "unclear", "ambiguous"))
        ]
        for issue in unresolved_issues:
            if issue not in assumptions:
                assumptions.append(issue)

        return {
            "status": "prompt_ready",
            "assistant_message": initial_result.get("assistant_message") or "Your refined prompt is ready.",
            "questions": [],
            "final_prompt": candidate,
            "assumptions": assumptions,
            "applied_guidelines": guideline_ids,
            "suggested_settings": initial_result.get("suggested_settings"),
        }

    except KnowledgeBaseError as exc:
        LOGGER.exception("PromptForge knowledge-base error")
        return _error_response(f"Guideline knowledge base error: {exc}")
    except OllamaServiceError as exc:
        LOGGER.warning("PromptForge local AI service error: %s", exc)
        return _error_response(str(exc))
    except PromptIntegrityError as exc:
        LOGGER.error("PromptForge rejected a prompt that failed preservation checks: %s", exc)
        return _error_response(str(exc))
    except ModelResponseError as exc:
        LOGGER.exception("PromptForge received an invalid local AI response")
        return _error_response(f"The local AI returned an incomplete or invalid response: {exc}")
    except (TypeError, ValueError) as exc:
        return _error_response(f"Invalid request: {exc}")
    except Exception as exc:  # Defensive boundary for the Flask route.
        LOGGER.exception("Unexpected PromptForge prompt-engine error")
        return _error_response(
            f"An unexpected prompt-engine error occurred ({type(exc).__name__}). "
            "Check the application log and try again."
        )


__all__ = ["process_message", "reset_knowledge_base_cache"]
