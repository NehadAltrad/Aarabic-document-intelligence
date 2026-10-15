"""Held-out category classification and evidence retrieval evaluation.

Run from project root: python -m scripts.evaluate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, f1_score

from docintel.classification import PROJECT_ROOT, read_labeled_csv, train_classifier
from docintel.documents import make_chunks, parse_document
from docintel.retrieval import DocumentRetriever


def evaluate() -> dict:
    model = train_classifier()
    x_test, y_test = read_labeled_csv(PROJECT_ROOT / "data" / "test" / "labels.csv")
    predictions = [str(label) for label in model.predict(x_test)]
    category_results = {
        "samples": len(y_test),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4),
        "by_category": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "errors": [
            {"text": text, "expected": expected, "predicted": actual}
            for text, expected, actual in zip(x_test, y_test, predictions)
            if expected != actual
        ],
    }

    sample_dir = PROJECT_ROOT / "data" / "samples"
    pages = []
    for filename in sorted(sample_dir.glob("*.txt")):
        pages.extend(parse_document(filename.name, filename.read_bytes()))
    retriever = DocumentRetriever(make_chunks(pages))
    qa_cases = json.loads((PROJECT_ROOT / "data" / "test" / "qa_questions.json").read_text(encoding="utf-8"))
    details = []
    for case in qa_cases:
        answer = retriever.ask(case["question"], top_k=3)
        correct = [
            item.source == case["source"] and case["quote"] in item.excerpt
            for item in answer.evidence
        ]
        details.append(
            {
                "question": case["question"],
                "expected_source": case["source"],
                "expected_quote": case["quote"],
                "top_1_hit": bool(correct[0]) if correct else False,
                "top_3_hit": any(correct),
                "returned_results": len(answer.evidence),
                "top_result": answer.evidence[0].source if answer.evidence else None,
            }
        )
    retrieval_results = {
        "questions": len(details),
        "hit_at_1": round(sum(row["top_1_hit"] for row in details) / len(details), 4),
        "hit_at_3": round(sum(row["top_3_hit"] for row in details) / len(details), 4),
        "details": details,
    }
    return {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_note": "All training/test/retrieval documents are small and synthetic. Results do not estimate real-world performance.",
        "classification": category_results,
        "retrieval": retrieval_results,
    }


def main() -> None:
    results = evaluate()
    report = PROJECT_ROOT / "reports" / "evaluation.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cls = results["classification"]
    ret = results["retrieval"]
    print(f"Classification | n={cls['samples']}, accuracy={cls['accuracy']:.1%}, macro-F1={cls['macro_f1']:.3f}")
    print(f"Retrieval      | n={ret['questions']}, Hit@1={ret['hit_at_1']:.1%}, Hit@3={ret['hit_at_3']:.1%}")
    print(f"Report saved: {report}")


if __name__ == "__main__":
    main()
