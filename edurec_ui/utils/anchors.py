import re
import unicodedata
from typing import Any, Dict, List, Optional


def _strip_accents_lower(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()
    except Exception:
        return s.lower()


def _detect_big_label(line: str) -> Optional[str]:
    t = _strip_accents_lower(line or "").strip()
    # Bai 1, Bai toan 1, Bai:1
    m = re.match(r"^(bai|bai\s*toan|bai\s*\w*)\s*(\d+)([^\d]|$)", t)
    if m:
        return f"B{int(m.group(2))}"
    # Cau 1 as big if no explicit Bai present
    m = re.match(r"^(cau)\s*(\d+)([^\d]|$)", t)
    if m:
        return f"B{int(m.group(2))}"
    # Numeric heading like '1.' or '1)'
    m = re.match(r"^(\d+)\s*[\.)]", t)
    if m:
        return f"B{int(m.group(1))}"
    # 'B1', 'B2.' etc
    m = re.match(r"^b\s*(\d+)([^\d]|$)", t)
    if m:
        return f"B{int(m.group(1))}"
    return None


def _detect_subpart_label(line: str) -> Optional[str]:
    t = _strip_accents_lower(line or "").strip()
    # a), b., c)
    m = re.match(r"^([a-z])\s*[\)\.\-]", t)
    if m:
        ch = m.group(1)
        if "a" <= ch <= "z":
            return ch
    # 'cau a', 'cau b'
    m = re.match(r"^cau\s*([a-z])([^a-z]|$)", t)
    if m:
        return m.group(1)
    return None


def build_anchors_from_text(text: Optional[str]) -> List[Dict[str, Any]]:
    """Build anchor segments from OCR text. Each segment has an anchor_id and text.
    - Big anchors: B{n}
    - Subparts: B{n}.{letter}
    """
    if not text:
        return []
    lines = [ln for ln in str(text).splitlines()]
    anchors: List[Dict[str, Any]] = []
    current_big: Optional[str] = None
    current_id: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal buf, current_id
        if current_id is not None and buf:
            anchors.append({"anchor_id": current_id, "text": "\n".join(buf).strip()})
        buf = []

    for ln in lines:
        big = _detect_big_label(ln)
        sub = _detect_subpart_label(ln)
        if big:
            flush()
            current_big = big
            current_id = big
            buf.append(ln)
            continue
        if sub and current_big:
            flush()
            current_id = f"{current_big}.{sub}"
            buf.append(ln)
            continue
        if current_id is None and _strip_accents_lower(ln).strip():
            current_id = current_big or "B0"
        buf.append(ln)
    flush()
    return anchors

