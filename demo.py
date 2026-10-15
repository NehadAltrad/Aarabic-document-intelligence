"""Console demo without starting the web interface."""

from pathlib import Path

from docintel.classification import PROJECT_ROOT, predict_category, train_classifier
from docintel.documents import make_chunks, parse_document
from docintel.retrieval import DocumentRetriever


def main() -> None:
    sample = PROJECT_ROOT / "data" / "samples" / "policy_data.txt"
    pages = parse_document(sample.name, sample.read_bytes())
    category = predict_category(train_classifier(), "\n".join(page.text for page in pages))
    print(f"Predicted category: {category.category} (uncalibrated probability: {category.score:.3f})")
    result = DocumentRetriever(make_chunks(pages)).ask("ما المطلوب قبل إرسال السجلات الحساسة؟")
    print(result.message)
    for item in result.evidence:
        print(f"[{item.source} | page {item.page or '-'} | similarity {item.similarity:.3f}] {item.excerpt}")


if __name__ == "__main__":
    main()
