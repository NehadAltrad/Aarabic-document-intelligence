"""Extractive Q&A: retrieve exact excerpts rather than inventing generated answers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .arabic import normalize_arabic
from .documents import TextChunk


@dataclass(frozen=True)
class Evidence:
    source: str
    page: int | None
    excerpt: str
    similarity: float


@dataclass(frozen=True)
class Answer:
    found: bool
    evidence: tuple[Evidence, ...]
    message: str


class DocumentRetriever:
    def __init__(self, chunks: list[TextChunk]):
        if not chunks:
            raise ValueError("At least one text chunk is needed")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize_arabic,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            max_features=40_000,
        )
        self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in chunks])

    def ask(self, question: str, top_k: int = 3, min_similarity: float = 0.17) -> Answer:
        if not question.strip():
            return Answer(False, (), "اكتبي سؤالًا أولًا.")
        question_vector = self.vectorizer.transform([question])
        if question_vector.nnz == 0:
            return Answer(False, (), "لا توجد أدلة كافية للإجابة من المستندات.")
        similarities = cosine_similarity(question_vector, self.matrix).ravel()
        candidates = np.argsort(-similarities, kind="stable")[: max(1, top_k)]
        results = tuple(
            Evidence(
                source=self.chunks[int(index)].source,
                page=self.chunks[int(index)].page,
                excerpt=self.chunks[int(index)].text,
                similarity=float(similarities[index]),
            )
            for index in candidates
            if similarities[index] >= min_similarity
        )
        if not results:
            return Answer(False, (), "لا توجد أدلة كافية للإجابة من المستندات.")
        return Answer(
            True,
            results,
            "هذه أقرب المقاطع ذات الصلة بالسؤال؛ راجعي الدليل، فهي ليست إجابة مولَّدة.",
        )
