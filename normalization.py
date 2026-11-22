import re
import unicodedata

ARABIC_BLOCKS = r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"

def normalize_ar(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u0640", "")  # tatweel
    # remove combining marks (diacritics)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    # Arabic-Indic numbers
    for i, d in enumerate("٠١٢٣٤٥٦٧٨٩"):
        text = text.replace(d, str(i))
    # Alef variants
    text = re.sub(r"[أإآ]", "ا", text)
    # remove punctuation/symbols (keep Arabic + ascii word chars + whitespace)
    text = re.sub(rf"[^{ARABIC_BLOCKS}\w\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
