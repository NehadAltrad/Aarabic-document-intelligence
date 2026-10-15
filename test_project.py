from pathlib import Path

import pymupdf
import pytest

from docintel.arabic import normalize_arabic
from docintel.classification import PROJECT_ROOT, predict_category, train_classifier
from docintel.documents import DocumentError, make_chunks, parse_document
from docintel.indicators import extract_indicators
from docintel.retrieval import DocumentRetriever
from scripts.evaluate import evaluate


def test_arabic_normalization():
    assert normalize_arabic("إدَارَة المَعْلُومات  عَلَى") == "ادارة المعلومات علي"


def test_txt_parsing_and_chunk_provenance():
    pages = parse_document("example.txt", "مرحبًا بكم في المستند التجريبي.".encode())
    assert pages[0].source == "example.txt"
    assert pages[0].page is None
    chunks = make_chunks(pages, window_words=5, overlap_words=1)
    assert chunks and chunks[0].source == "example.txt"


def test_reject_unsupported_and_empty():
    with pytest.raises(DocumentError):
        parse_document("file.exe", b"hello")
    with pytest.raises(DocumentError):
        parse_document("file.txt", b"")


def test_reject_scanned_like_pdf_without_text():
    pdf = pymupdf.open()
    pdf.new_page()
    with pytest.raises(DocumentError, match="OCR"):
        parse_document("scan.pdf", pdf.tobytes())
    pdf.close()


def test_pdf_page_reference():
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((60, 70), "Test PDF content")
    pages = parse_document("paper.pdf", pdf.tobytes())
    assert pages[0].page == 1
    assert "Test PDF content" in pages[0].text
    pdf.close()


def test_classification_basic():
    model = train_classifier()
    prediction = predict_category(model, "تقرير الأداء يعرض قياس المؤشرات ونسب الإنجاز خلال الربع")
    assert prediction.category == "تقارير"
    assert 0 <= prediction.score <= 1


def test_retrieval_with_exact_source_evidence():
    file = PROJECT_ROOT / "data" / "samples" / "policy_data.txt"
    retriever = DocumentRetriever(make_chunks(parse_document(file.name, file.read_bytes())))
    response = retriever.ask("من صاحب الموافقة على مشاركة السجلات الحساسة؟")
    assert response.found
    assert "مالك البيانات" in response.evidence[0].excerpt
    assert response.evidence[0].source == "policy_data.txt"


def test_retrieval_can_abstain():
    retriever = DocumentRetriever(make_chunks(parse_document("x.txt", "سياسة مشاركة البيانات".encode())))
    result = retriever.ask("ZQXJVV 998877 randomquestion")
    assert not result.found


def test_simple_indicators():
    result = extract_indicators("بتاريخ 15/11/2026 وصلت النسبة إلى 92% وأرسل info@example.com")
    assert "15/11/2026" in result["التواريخ الرقمية"]
    assert "92%" in result["النسب المئوية"]
    assert "info@example.com" in result["البريد الإلكتروني"]


def test_evaluation_has_reproducible_case_counts():
    result = evaluate()
    assert result["classification"]["samples"] == 20
    assert result["retrieval"]["questions"] == 8
    assert 0 <= result["retrieval"]["hit_at_1"] <= 1
