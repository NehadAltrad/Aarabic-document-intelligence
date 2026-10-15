# Evaluation Protocol | بروتوكول تقييم النموذج

## 1. Questions we're measuring

**Classification:** Does a supervised char n-gram TF-IDF + Logistic Regression pipeline choose the correct document category on labeled, held-out Arabic examples?

**Evidence retrieval:** Does TF-IDF cosine retrieval place the expected exact supporting phrase from the expected source file in the first 1 / first 3 returned chunks?

This evaluation **does not** assess natural language generation, correctness of a synthesized answer, OCR quality, fairness, or actual enterprise deployment suitability.

## 2. Dataset boundaries

- **Training classification:** 56 synthetic short texts, 14 for each of four categories.
- **Held-out classification:** 20 different synthetic short texts, 5 for each category.
- **Retrieval:** 4 synthetic `.txt` documents and 8 question/source/quote references.
- Texts are educational, manually authored, domain-simple, and not representative of natural noisy Arabic or real institutional documents.
- This is an independent text holdout but shares writing style/vocabulary with training, so strong performance is expected.

## 3. Measurements

- **Accuracy:** `correct classifications / all evaluated classifications`.
- **Macro-F1:** arithmetic mean of F1 across categories, so each category receives equal weight.
- **Hit@k:** fraction of queries where at least one of the first `k` results contains the expected quote **and** comes from the expected file. It is an evidence retrieval metric, not answer correctness.
- Source page numbers are displayed for PDF, but current evaluation cases use TXT files.

## 4. Reproduction

Run on Python 3.11+ from repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.evaluate
```

Open `reports/evaluation.json` for detailed per-question hits and classification errors.

## 5. Current demo results

| Metric | Score | Number of cases |
|---|---:|---:|
| Accuracy | 1.000 | 20 |
| Macro-F1 | 1.000 | 20 |
| Hit@1 | 1.000 | 8 |
| Hit@3 | 1.000 | 8 |

**Do not interpret 1.000 as robust real-world performance.** With such a tiny and simple dataset it can reflect lexical shortcuts and sampling variability. No confidence intervals are claimed.

## 6. Realistic next benchmark

1. Define approved document categories and clear labeling rules, then gather legally usable, anonymized documents.
2. Split by source document and (where practical) by institution/time to avoid near-duplicate leakage.
3. Include long PDFs, noisy extraction, near-category cases, paraphrases, questions without answers, and documents with overlapping terminology.
4. Have reviewers annotate reference passages and agree on resolution of ambiguous examples.
5. Report per-category precision, recall and F1, confusion matrix, Hit@k, MRR, and abstention false-positive rate on unanswerable questions.
6. Repeat evaluation after model revisions and use a blind final holdout.

## 7. Known limitations

- Model scores are **uncalibrated probabilities**, not validated measures of correctness.
- Similarity threshold `0.17` is a heuristic. Lexical overlap can cause unsupported matches.
- Retrieval snippets may be incomplete due to fixed word chunk boundaries; source review is essential.
- The demonstration has **no human evaluation of answer utility** and no statistical generalization claims.
