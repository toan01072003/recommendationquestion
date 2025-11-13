import json
import re
from typing import Any


def safe_json_loads(text: str, default: Any):
    """Load JSON with a fallback that escapes stray backslashes.

    Useful for occasionally malformed LLM outputs.
    """
    try:
        return json.loads(text)
    except Exception:
        try:
            fixed = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)
            return json.loads(fixed)
        except Exception:
            return default

