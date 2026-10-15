"""Small transparent Arabic normalization helpers."""

import re
import unicodedata

_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TRANSLATION = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ؤ": "و", "ئ": "ي"})


def normalize_arabic(text: str) -> str:
    """Normalize spelling variants for modeling; preserve original source for citations."""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text).replace("ـ", "")
    text = text.translate(_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip().lower()
