import os
import tempfile
import time
from typing import Iterable, Optional


def get_model():
    """Return a configured Gemini model using GOOGLE_API_KEY/GEMINI_API_KEY.

    Raises a RuntimeError if API key is missing or SDK is not installed.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY/GEMINI_API_KEY for Gemini")
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(f"Gemini SDK not installed: {e}")
    genai.configure(api_key=api_key)
    generation_config = {"temperature": 0.2, "response_mime_type": "application/json"}
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name=model_name, generation_config=generation_config)


def upload_bytes_to_gemini(name: str, data: bytes):
    """Upload bytes as a temporary file to Gemini and return the file ref."""
    import google.generativeai as genai
    suffix = os.path.splitext(name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(data)
        path = tf.name
    try:
        return genai.upload_file(path=path, display_name=name)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def wait_until_files_active(file_refs: Iterable, timeout_sec: int = 90):
    """Wait for uploaded Gemini files to become ACTIVE (best-effort)."""
    if not file_refs:
        return
    import google.generativeai as genai
    start = time.time()
    pending = list(file_refs)
    while pending and (time.time() - start) < timeout_sec:
        still = []
        for f in pending:
            try:
                fid = getattr(f, "name", None) or getattr(f, "uri", None) or getattr(f, "id", None)
                if not fid:
                    continue
                fresh = genai.get_file(fid)
                state = getattr(fresh, "state", None)
                sname = getattr(state, "name", None) or str(state)
                if sname and "ACTIVE" in sname:
                    continue
                if sname and "FAILED" in sname:
                    raise RuntimeError(f"Gemini file processing FAILED: {fid}")
                still.append(f)
            except Exception:
                still.append(f)
        if not still:
            return
        time.sleep(1.0)
        pending = still

