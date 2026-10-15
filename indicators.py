"""Simple, explainable patterns extracted from Arabic source text."""

import re


def extract_indicators(text: str) -> dict[str, list[str]]:
    patterns = {
        "التواريخ الرقمية": r"(?<!\d)\d{1,4}[/-]\d{1,2}[/-]\d{1,4}(?!\d)",
        "النسب المئوية": r"(?<!\d)\d+(?:[.,]\d+)?\s*[%٪]",
        "البريد الإلكتروني": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    }
    return {label: list(dict.fromkeys(re.findall(pattern, text))) for label, pattern in patterns.items()}
