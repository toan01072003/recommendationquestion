import io
import json
import os
import tempfile
from typing import Optional

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    _TRANSFORMERS_OK = True
except Exception:
    torch = None
    AutoModel = AutoTokenizer = None
    _TRANSFORMERS_OK = False


_MODEL = None
_TOKENIZER = None
_DEVICE = "cpu"


def _load_local_model():
    """Lazy-load the local DeepSeek-OCR model from Hugging Face.
    No API key is required. If loading fails, keep globals as None.
    You may need to install packages noted in docs/DEEPSEEK.md.
    """
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None and _TOKENIZER is not None:
        return
    if not _TRANSFORMERS_OK:
        return
    name = os.environ.get("DEEPSEEK_OCR_MODEL", "deepseek-ai/DeepSeek-OCR")
    attn_impl = os.environ.get("DS_ATTN_IMPL", None)
    kwargs = {"trust_remote_code": True, "use_safetensors": True}
    if attn_impl:
        kwargs["_attn_implementation"] = attn_impl
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(name, **kwargs)
        device = "cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "-1") != "-1" else "cpu"
        mdl = mdl.eval().to(device)
        # Prefer bfloat16 on CUDA when available
        if device == "cuda":
            try:
                mdl = mdl.to(torch.bfloat16)
            except Exception:
                pass
        _MODEL, _TOKENIZER, _DEVICE = mdl, tok, device
    except Exception:
        _MODEL = None
        _TOKENIZER = None
        _DEVICE = "cpu"


def ocr_bytes(img_bytes: bytes, language: str = "vi") -> Optional[str]:
    """Run OCR locally using deepseek-ai/DeepSeek-OCR via transformers.
    Returns a plain text transcription (joined lines) or None on failure.
    """
    if not img_bytes:
        return None
    _load_local_model()
    if _MODEL is None or _TOKENIZER is None:
        return None
    # Persist to a temp image path for the model's .infer interface
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
        tf.write(img_bytes)
        img_path = tf.name
    try:
        # Prompt encourages clean OCR or markdown; choose markdown then post-process to plain lines
        # Default prompt: preserve math and visible anchors (Bài 1, a), ...)
        prompt = os.environ.get(
            "DEEPSEEK_OCR_PROMPT",
            "<image>\n<|grounding|>Convert the document to markdown; preserve math formulas using inline $...$ and block $$...$$; keep visible labels like 'Bài 1', 'Câu 1', 'a)' on their own lines."
        )
        # Fallback sizes can be set via env vars
        base_size = int(os.environ.get("DEEPSEEK_OCR_BASE_SIZE", "1024"))
        image_size = int(os.environ.get("DEEPSEEK_OCR_IMAGE_SIZE", "640"))
        crop_mode = os.environ.get("DEEPSEEK_OCR_CROP", "true").lower() == "true"
        # Many community checkpoints expose .infer(tokenizer, ...)
        res = _MODEL.infer(
            _TOKENIZER,
            prompt=prompt,
            image_file=img_path,
            output_path="",
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            test_compress=False,
            save_results=False,
        )
        text = res if isinstance(res, str) else str(res)
        # Normalize to plain lines text
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("lines"), list):
                return "\n".join(str(x) for x in data["lines"])
        except Exception:
            pass
        # Remove markdown bullets when present
        lines = [ln.strip(" -*\t") for ln in text.splitlines()]
        return "\n".join(l for l in lines if l)
    except Exception:
        return None
    finally:
        try:
            os.remove(img_path)
        except Exception:
            pass
