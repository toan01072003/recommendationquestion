"""Backend API client functions for EduRec FastAPI server."""

from typing import Any, Dict, Optional
import logging

import requests

logger = logging.getLogger(__name__)

# Default timeout for API calls (seconds)
DEFAULT_TIMEOUT = 60
EVALUATE_TIMEOUT = 90


def _build_url(api_base: str, endpoint: str) -> str:
    """Build full URL from base and endpoint."""
    base = (api_base or "http://localhost:8000").rstrip("/")
    return f"{base}{endpoint}"


def call_deepseek_ocr_api(
    api_base: str,
    img_name: str,
    img_bytes: bytes,
    language: str = "vi",
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[str]:
    """Call FastAPI endpoint /ocr/deepseek-extract and return joined text or None.

    Parameters:
        api_base: Base URL to your FastAPI server (e.g., http://localhost:8000)
        img_name: A display name for the image
        img_bytes: Raw image bytes
        language: Language hint (default: "vi")
        timeout: Request timeout in seconds

    Returns:
        Joined OCR text lines or None if failed
    """
    if not img_bytes:
        return None

    url = _build_url(api_base, "/ocr/deepseek-extract")
    files = {"submission_image": (img_name or "submission.png", img_bytes, "application/octet-stream")}
    data = {"language": language or "vi"}

    try:
        resp = requests.post(url, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        js = resp.json()
        if isinstance(js, dict) and isinstance(js.get("lines"), list):
            return "\n".join(str(x) for x in js["lines"])
    except requests.exceptions.Timeout:
        logger.warning("OCR API timeout after %ds", timeout)
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to OCR API at %s", url)
    except Exception as e:
        logger.warning("OCR API error: %s", e)

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
    timeout: int = EVALUATE_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Call FastAPI endpoint /assessments/evaluate-with-key.

    Parameters:
        api_base: Base URL to your FastAPI server (e.g., http://localhost:8000)
        exam_file: Raw bytes for exam image (optional)
        exam_name: Display name for exam
        key_file: Raw bytes for answer key image (optional)
        key_name: Display name for answer key
        sub_file: Raw bytes for submission image (required)
        sub_name: Display name for submission
        language: Language hint (default: "vi")
        timeout: Request timeout in seconds

    Returns:
        Evaluation result dict or None if failed
    """
    if not sub_file:
        return None

    url = _build_url(api_base, "/assessments/evaluate-with-key")
    files = {}

    if exam_file:
        files["exam_image"] = (exam_name or "exam.png", exam_file, "application/octet-stream")
    if key_file:
        files["answer_key_image"] = (key_name or "answer.png", key_file, "application/octet-stream")
    files["submission_image"] = (sub_name or "submission.png", sub_file, "application/octet-stream")

    data = {"language": language or "vi"}

    try:
        resp = requests.post(url, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.warning("Evaluate API timeout after %ds", timeout)
    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Evaluate API at %s", url)
    except Exception as e:
        logger.warning("Evaluate API error: %s", e)

    return None

