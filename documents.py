"""Safely ingest UTF-8 text and text-based PDFs, retaining page provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

import pymupdf

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PAGES = 30
MAX_TEXT_CHARS = 100_000


@dataclass(frozen=True)
class DocumentPage:
    source: str
    page: int | None
    text: str


@dataclass(frozen=True)
class TextChunk:
    source: str
    page: int | None
    text: str
    index: int


class DocumentError(ValueError):
    """The provided document cannot be safely or usefully processed."""


def parse_document(filename: str, contents: bytes) -> list[DocumentPage]:
    """Parse in memory; no upload is written to disk by this module."""
    if not contents:
        raise DocumentError("الملف فارغ.")
    if len(contents) > MAX_FILE_BYTES:
        raise DocumentError("حجم الملف أكبر من 5 ميغابايت.")

    extension = PurePath(filename).suffix.lower()
    if extension == ".txt":
        try:
            text = contents.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise DocumentError("يجب أن يكون الملف النصي بترميز UTF-8.") from exc
        if not text:
            raise DocumentError("لم يُعثر على نص قابل للقراءة.")
        if len(text) > MAX_TEXT_CHARS:
            raise DocumentError("النص أطول من الحد المسموح (100 ألف حرف).")
        return [DocumentPage(filename, None, text)]

    if extension != ".pdf":
        raise DocumentError("الصيغ المدعومة هي PDF النصي وTXT فقط.")
    if not contents.startswith(b"%PDF-"):
        raise DocumentError("الملف المرفوع ليس PDF صالحًا.")

    pages: list[DocumentPage] = []
    char_count = 0
    try:
        with pymupdf.open(stream=contents, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise DocumentError("ملف PDF محمي بكلمة مرور.")
            if len(pdf) > MAX_PAGES:
                raise DocumentError("ملف PDF يتجاوز حد 30 صفحة.")
            for position, page in enumerate(pdf, start=1):
                extracted = page.get_text("text").strip()
                char_count += len(extracted)
                if char_count > MAX_TEXT_CHARS:
                    raise DocumentError("النص أطول من الحد المسموح (100 ألف حرف).")
                if extracted:
                    pages.append(DocumentPage(filename, position, extracted))
    except DocumentError:
        raise
    except (ValueError, RuntimeError, pymupdf.FileDataError) as exc:
        raise DocumentError("تعذّر قراءة PDF؛ تأكدي من سلامة الملف.") from exc

    if not pages:
        raise DocumentError("لا يوجد نص قابل للاستخراج؛ قد يكون PDF مصوّرًا ويحتاج OCR.")
    return pages


def make_chunks(
    pages: list[DocumentPage], window_words: int = 75, overlap_words: int = 15
) -> list[TextChunk]:
    if window_words <= 0 or not (0 <= overlap_words < window_words):
        raise ValueError("Invalid chunk window/overlap parameters")
    results: list[TextChunk] = []
    stride = window_words - overlap_words
    for document_page in pages:
        words = document_page.text.split()
        for start in range(0, len(words), stride):
            excerpt = " ".join(words[start : start + window_words])
            if excerpt:
                results.append(
                    TextChunk(
                        source=document_page.source,
                        page=document_page.page,
                        text=excerpt,
                        index=len(results),
                    )
                )
            if start + window_words >= len(words):
                break
    return results
