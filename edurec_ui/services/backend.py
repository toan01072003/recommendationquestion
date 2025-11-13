import json
from typing import Any, Dict, Optional

import requests


def call_deepseek_ocr_api(api_base: str, img_name: str, img_bytes: bytes, language: str = "vi") -> Optional[str]:
    """Call FastAPI endpoint /ocr/deepseek-extract and return joined text or None.

    Parameters
    - api_base: Base URL to your FastAPI server (e.g., http://localhost:8000)
    - img_name: A display name for the image
    - img_bytes: Raw image bytes
    - language: Language hint (default: "vi")
    """
    if not img_bytes:
        return None
    url = (api_base or "http://localhost:8000").rstrip("/") + "/ocr/deepseek-extract"
    files = {"submission_image": (img_name or "submission.png", img_bytes, "application/octet-stream")}
    data = {"language": language or "vi"}
    try:
        resp = requests.post(url, files=files, data=data, timeout=60)
        resp.raise_for_status()
        js = resp.json()
        if isinstance(js, dict) and isinstance(js.get("lines"), list):
            return "\n".join(str(x) for x in js["lines"])
    except Exception:
        return None
    return None


def call_evaluate_with_key_api(
    api_base: str,
    exam_file: Optional[bytes],
    exam_name: Optional[str],
    key_file: Optional[bytes],
    key_name: Optional[str],
    sub_file: bytes,
    sub_name: Optional[str],
    language: str = "vi",
) -> Optional[Dict[str, Any]]:
    """Call FastAPI endpoint /assessments/evaluate-with-key. Returns dict or None.

    Parameters
    - api_base: Base URL to your FastAPI server (e.g., http://localhost:8000)
    - exam_file/key_file/sub_file: Raw bytes for exam, answer key (optional), and submission (required)
    - exam_name/key_name/sub_name: Display names
    - language: Language hint (default: "vi")
    """
    if not sub_file:
        return None
    url = (api_base or "http://localhost:8000").rstrip("/") + "/assessments/evaluate-with-key"
    files = {}
    if exam_file:
        files["exam_image"] = (exam_name or "exam.png", exam_file, "application/octet-stream")
    if key_file:
        files["answer_key_image"] = (key_name or "answer.png", key_file, "application/octet-stream")
    files["submission_image"] = (sub_name or "submission.png", sub_file, "application/octet-stream")
    data = {"language": language or "vi"}
    try:
        resp = requests.post(url, files=files, data=data, timeout=90)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None

