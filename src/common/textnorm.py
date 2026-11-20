import re

_WS = re.compile(r"\s+")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s<>]", re.UNICODE)


def clean_text(text: str) -> str:
    return _WS.sub(" ", str(text)).strip()


def dedup_key(text: str) -> str:
    t = str(text).lower()
    t = _URL.sub("<url>", t)
    t = _DIGITS.sub("0", t)
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip() or "<empty>"
