import os
import io
import json
import time
import tempfile
import re
import unicodedata
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
import streamlit.components.v1 as components
import requests


# Load API key
load_dotenv(override=False)

st.set_page_config(page_title="EduRec – Phân tích đề & gợi ý luyện tập", page_icon="📐", layout="wide")
st.title("EduRec – Phân tích đề + bài làm và gợi ý luyện tập")
st.caption("Chạy thuần Streamlit; cần GOOGLE_API_KEY/GEMINI_API_KEY trong môi trường.")


# ---------------- Gemini helpers ----------------
def get_model():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Thiếu GOOGLE_API_KEY/GEMINI_API_KEY trong môi trường hoặc Streamlit secrets.")
        st.stop()
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = {"temperature": 0.2, "response_mime_type": "application/json"}
    return genai.GenerativeModel(model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"), generation_config=generation_config)


def is_likely_image_bytes(b: bytes) -> bool:
    if not b or len(b) < 4:
        return False
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if b.startswith(b"\xff\xd8"):
        return True
    if b.startswith(b"GIF8"):
        return True
    if b.startswith(b"RIFF") and b[8:12] == b"WEBP":
        return True
    return False


def upload_bytes_to_gemini(name: str, data: bytes):
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


def wait_until_files_active(file_refs, timeout_sec: int = 90):
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


# --------------- DeepSeek OCR via backend ---------------
def call_deepseek_ocr_api(api_base: str, img_name: str, img_bytes: bytes, language: str = "vi") -> Optional[str]:
    """Call FastAPI endpoint /ocr/deepseek-extract and return joined text or None.
    Expects FastAPI app from app.py running at api_base (e.g., http://localhost:8000).
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


def call_evaluate_with_key_api(api_base: str,
                               exam_file: Optional[bytes], exam_name: Optional[str],
                               key_file: Optional[bytes], key_name: Optional[str],
                               sub_file: bytes, sub_name: Optional[str],
                               language: str = "vi") -> Optional[Dict[str, Any]]:
    """Call FastAPI endpoint /assessments/evaluate-with-key. Returns dict or None."""
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


# --------------- JSON parsing helpers ---------------
def safe_json_loads(text: str, default: Any):
    try:
        return json.loads(text)
    except Exception:
        try:
            fixed = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)
            return json.loads(fixed)
        except Exception:
            return default


# --------------- Anchor linking helpers ---------------
def _strip_accents_lower(s: str) -> str:
    try:
        return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()
    except Exception:
        return s.lower()


def _detect_big_label(line: str) -> Optional[str]:
    t = _strip_accents_lower(line or "").strip()
    # Bài 1, Bai 1, Bai:1
    m = re.match(r"^(bai|bai\s*toan|bai\s*\w*)\s*(\d+)([^\d]|$)", t)
    if m:
        return f"B{int(m.group(2))}"
    # Câu 1 as big if no explicit Bài present
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
            # starting a new big question
            flush()
            current_big = big
            current_id = big
            # keep the heading line in buffer, helps context
            buf.append(ln)
            continue
        if sub and current_big:
            flush()
            current_id = f"{current_big}.{sub}"
            buf.append(ln)
            continue
        # default: continue current segment
        if current_id is None and _strip_accents_lower(ln).strip():
            # Start a default segment if any text appears
            current_id = current_big or "B0"
        buf.append(ln)
    flush()
    return anchors


# --------------- Scoring helpers ---------------
def _normalize_num_token(tok: str) -> Optional[float]:
    if tok is None:
        return None
    s = str(tok).strip()
    if not s:
        return None
    s = s.replace(" ", "").replace(",", ".")
    # fraction a/b
    if "/" in s and all(part.strip() for part in s.split("/", 1)):
        try:
            a, b = s.split("/", 1)
            a = float(a); b = float(b)
            if b != 0:
                return a / b
        except Exception:
            pass
    try:
        return float(s)
    except Exception:
        return None


def _split_candidates(txt: str) -> List[str]:
    if not txt:
        return []
    # split by common separators
    parts = re.split(r"[;\n,]+", str(txt))
    return [p.strip() for p in parts if p.strip()]


def rule_check_correct(student: Optional[str], gold: Optional[str]) -> Optional[bool]:
    """Conservative rule-based check.
    - If both parse to numeric (or lists of numerics), compare with tolerance.
    - Else if normalized exact string match, consider correct.
    Returns True/False, or None if uncertain.
    """
    if not student or not gold:
        return None
    # List case
    s_parts = _split_candidates(student)
    g_parts = _split_candidates(gold)
    if s_parts and g_parts:
        s_nums = [_normalize_num_token(x) for x in s_parts]
        g_nums = [_normalize_num_token(x) for x in g_parts]
        if all(v is not None for v in s_nums) and all(v is not None for v in g_nums):
            if len(s_nums) == len(g_nums):
                s_sorted = sorted(s_nums)
                g_sorted = sorted(g_nums)
                return all(abs(a - b) <= 1e-6 for a, b in zip(s_sorted, g_sorted))
    # Single numeric
    sn = _normalize_num_token(student)
    gn = _normalize_num_token(gold)
    if sn is not None and gn is not None:
        return abs(sn - gn) <= 1e-6
    # Fallback to normalized exact text
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        return s
    if norm(student) == norm(gold):
        return True
    return None


def extract_final_answer_from_text(text: Optional[str]) -> str:
    """Heuristic to extract final answer from an answer-key text block.
    - Prefer lines containing 'Đáp án', 'Kết quả' (case-insensitive, accent-insensitive).
    - Else take the last non-empty line; if it contains '=', take RHS.
    """
    if not text:
        return ""
    try:
        lines = [ln.strip() for ln in str(text).splitlines() if str(ln).strip()]
        if not lines:
            return ""
        for ln in reversed(lines):
            t = _strip_accents_lower(ln)
            if ("dap an" in t) or ("ket qua" in t) or t.startswith("kq"):
                if ":" in ln:
                    return ln.split(":", 1)[1].strip()
                return ln.strip()
        last = lines[-1]
        if "=" in last:
            rhs = last.split("=")[-1].strip()
            return rhs or last
        return last
    except Exception:
        return ""
def parse_goal(goal_text: Optional[str]) -> Optional[float]:
    if not goal_text:
        return None
    t = goal_text.strip().replace(" ", "")
    try:
        if t.endswith("%"):
            return max(0.0, min(1.0, float(t[:-1]) / 100.0))
        if "/" in t:
            a, b = t.split("/", 1)
            a = float(a); b = float(b)
            if b != 0:
                return max(0.0, min(1.0, a / b))
        v = float(t)
        if 0 <= v <= 1:
            return v
        if 1 < v <= 100:
            return v / 100.0
    except Exception:
        return None
    return None


def derive_gradebook(evaluation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    earned = 0.0
    total = 0.0
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("items"), list):
        return {"entries": [], "totals": {"earned": 0.0, "total": 0.0}}
    for it in evaluation["items"]:
        if not isinstance(it, dict):
            continue
        label = str(it.get("label") or "").strip()
        if not label:
            continue
        pts = it.get("points")
        try:
            pts = float(pts) if pts is not None else 1.0
        except Exception:
            pts = 1.0
        pe = it.get("points_earned")
        if pe is None:
            ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
            pe = pts if ok else 0.0
        else:
            try:
                pe = float(pe)
            except Exception:
                pe = 0.0
        total += pts
        earned += pe
        entries.append({
            "label": label,
            "skill_tag": it.get("skill_tag") or it.get("skillId"),
            "points": pts,
            "points_earned": pe,
            "rationale": it.get("rationale"),
        })
    return {"entries": entries, "totals": {"earned": earned, "total": total}}


def detect_geometry_context(evaluation: Dict[str, Any]) -> bool:
    text = " ".join([str(it.get("question") or "") for it in evaluation.get("items", []) if isinstance(it, dict)])
    text = text.lower()
    cues = [
        "tam giac", "vuong", "duong cao", "hinh chieu", "goc", "song song", "trung diem", "duong tron", "tiep tuyen",
        "prove", "similar", "altitude", "right triangle", "projection", "circle", "tangent"
    ]
    return any(c in text for c in cues)


def build_support_plan(weak: List[Dict[str, Any]], goal_frac: Optional[float], user_frac: Optional[float], total_q: int) -> List[Dict[str, Any]]:
    if total_q <= 0 or not weak:
        return []
    weights = [(w.get("skillId"), float(w.get("severity", 0.3))) for w in weak if w.get("skillId")]
    weights = [(sid, max(0.01, sev)) for sid, sev in weights]
    if not weights:
        return []
    weights = sorted(weights, key=lambda x: -x[1])[:3]
    sw = sum(w for _, w in weights) or 1.0
    alloc = [(sid, max(1, round(total_q * w / sw))) for sid, w in weights]
    diff = total_q - sum(c for _, c in alloc)
    i = 0
    while diff != 0 and alloc:
        sid, c = alloc[i % len(alloc)]
        if diff > 0:
            alloc[i % len(alloc)] = (sid, c + 1); diff -= 1
        else:
            if c > 1:
                alloc[i % len(alloc)] = (sid, c - 1); diff += 1
        i += 1
    plan = []
    for sid, count in alloc:
        # favor medium/hard for exam alignment
        e = max(0, round(count * 0.2))
        m = max(0, round(count * 0.5))
        h = max(0, count - e - m)
        plan.append({"skillId": sid, "counts": {"easy": int(e), "medium": int(m), "hard": int(h)}})
    return plan


# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.subheader("Thiết lập")
    lang = st.selectbox("Ngôn ngữ", options=["vi", "en"], index=0)
    goal_text = st.text_input("Mục tiêu điểm (ví dụ 8/10 hoặc 80%)", value="")
    user_text = st.text_input("Điểm của bạn (ví dụ 6/10)", value="")
    max_q = st.number_input("Số câu luyện gợi ý", min_value=1, max_value=20, value=6)
    exams = st.file_uploader("Ảnh đề (1 hoặc nhiều trang)", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=True)
    keys = st.file_uploader("Ảnh đáp án/chấm điểm (1 hoặc nhiều trang)", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=True)
    subs = st.file_uploader("Ảnh bài làm (1 hoặc nhiều trang)", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=True)


if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
    st.session_state.gradebook = None
    st.session_state.exam_anchors = []
    st.session_state.sub_anchors = []
    st.session_state.key_anchors = []
    st.session_state.ds_ocr_text = None
    st.session_state.ans_ocr_text = None
    st.session_state.gen_questions = []
    st.session_state.hint_questions = []

colA, colB = st.columns(2)
with colA:
    analyze_clicked = st.button("Phân tích & Chấm điểm")
with colB:
    suggest_clicked = st.button("Gợi ý luyện tập")

if analyze_clicked:
    model = get_model()

    # Upload files
    exam_refs = []
    key_refs = []
    sub_refs = []
    for f in (exams or []):
        data = f.getvalue()
        if is_likely_image_bytes(data):
            exam_refs.append(upload_bytes_to_gemini(f.name, data))
    for f in (keys or []):
        data = f.getvalue()
        if is_likely_image_bytes(data):
            key_refs.append(upload_bytes_to_gemini(f.name, data))
    for f in (subs or []):
        data = f.getvalue()
        if is_likely_image_bytes(data):
            sub_refs.append(upload_bytes_to_gemini(f.name, data))

    if not exam_refs and not sub_refs and not key_refs:
        st.warning("Cần ít nhất 1 ảnh đề, đáp án hoặc bài làm.")
        st.stop()

    try:
        wait_until_files_active(exam_refs + key_refs + sub_refs)
    except Exception:
        pass

    # Optional: DeepSeek OCR via backend for submission images
    api_base = os.environ.get("EDUREC_API_BASE", "http://localhost:8000")
    ds_ocr_text = None
    ans_ocr_text = None
    exam_ocr_text = None
    try:
        texts = []
        for f in (subs or []):
            data = f.getvalue()
            if is_likely_image_bytes(data):
                t = call_deepseek_ocr_api(api_base, f.name, data, language=lang)
                if t:
                    texts.append(t)
        if texts:
            ds_ocr_text = "\n\n---\n\n".join(texts)
        # OCR answer key as well
        atexts = []
        for f in (keys or []):
            data = f.getvalue()
            if is_likely_image_bytes(data):
                t = call_deepseek_ocr_api(api_base, f.name, data, language=lang)
                if t:
                    atexts.append(t)
        if atexts:
            ans_ocr_text = "\n\n---\n\n".join(atexts)
        # OCR exam as well (for Anchor IDs on exam side)
        etexts = []
        for f in (exams or []):
            data = f.getvalue()
            if is_likely_image_bytes(data):
                t = call_deepseek_ocr_api(api_base, f.name, data, language=lang)
                if t:
                    etexts.append(t)
        if etexts:
            exam_ocr_text = "\n\n---\n\n".join(etexts)
    except Exception:
        ds_ocr_text = None
        ans_ocr_text = None
        exam_ocr_text = None

    # Evaluate: parse exam into Bài/ý and map answers
    evaluation: Dict[str, Any] = {}
    # Build anchors from OCR (if available)
    exam_anchors = build_anchors_from_text(exam_ocr_text) if exam_ocr_text else []
    sub_anchors = build_anchors_from_text(ds_ocr_text) if ds_ocr_text else []
    key_anchors = build_anchors_from_text(ans_ocr_text) if ans_ocr_text else []

    # Prefer server API evaluation (handles step-level + cascade rule)
    ex_bytes = None; ex_name = None
    key_bytes0 = None; key_name = None
    sub_bytes0 = None; sub_name = None
    try:
        if exams:
            ex_bytes = exams[0].getvalue(); ex_name = exams[0].name
        if keys:
            key_bytes0 = keys[0].getvalue(); key_name = keys[0].name
        if subs:
            sub_bytes0 = subs[0].getvalue(); sub_name = subs[0].name
    except Exception:
        pass
    evaluation = call_evaluate_with_key_api(api_base, ex_bytes, ex_name, key_bytes0, key_name, sub_bytes0 or b"", sub_name, language=lang) or {}
    # Fallback to in-app LLM evaluation when API unavailable
    if not evaluation:
        eval_prompt = {
            "task": "evaluate_submission_items",
            "instructions": [
                "Parse the exam into big questions 'Bài x' and subparts 'a,b,c'.",
                "Normalize labels as 'B1.a', 'B1.b', ... (ASCII only).",
                "Extract short 'question' texts for each leaf.",
                "Map student's answers from the submission to these leaf labels; set mapping_confidence.",
                "Use the provided answer key (answer_key_ocr and anchors_hint.answer_key) to determine correctness and partial credit where applicable.",
                "Judge correctness; if no printed points, default 1 point per leaf.",
                "Add a free-form 'skill_tag' for the math topic.",
                "Step-level grading: extract solution_steps_expected and student_steps, and provide step_evaluation.",
                "Output all texts (solution_steps_expected, student_steps, rationale) in Vietnamese (vi). If needed, rewrite into Vietnamese.",
                "Return a concise 'correct_answer' (final answer only) in Vietnamese with units if shown in the key.",
                "Return JSON only.",
            ],
            "output_schema": {"type": "object", "properties": {"items": {"type": "array"}}, "required": ["items"], "additionalProperties": False},
            "locale": lang,
        }
        if ds_ocr_text:
            eval_prompt["ocr_hint"] = ds_ocr_text
        if ans_ocr_text:
            eval_prompt["answer_key_ocr"] = ans_ocr_text
        if exam_anchors or sub_anchors or key_anchors:
            eval_prompt["anchors_hint"] = {"exam": exam_anchors, "answer_key": key_anchors, "submission": sub_anchors}
        parts_ev: List[Any] = [json.dumps(eval_prompt)] + exam_refs + key_refs + sub_refs
        try:
            evresp = model.generate_content(parts_ev)
            raw = getattr(evresp, "text", "{}")
            evaluation = safe_json_loads(raw, {})
            if not isinstance(evaluation, dict):
                evaluation = {"raw": raw}
        except Exception as e:
            st.error(f"Lỗi đánh giá: {e}")
            evaluation = {}

    # Post-process: fill missing student_answer via Anchor IDs when possible
    try:
        if isinstance(evaluation, dict) and isinstance(evaluation.get("items"), list) and sub_anchors:
            sub_map = {a.get("anchor_id"): a.get("text") for a in sub_anchors if a.get("anchor_id")}
            for it in evaluation["items"]:
                if not isinstance(it, dict):
                    continue
                lbl = it.get("label")
                if not lbl:
                    continue
                if (not it.get("student_answer")) and lbl in sub_map:
                    it["student_answer"] = sub_map[lbl]
                    # boost mapping confidence if we mapped by exact anchor id
                    try:
                        mc = float(it.get("mapping_confidence") or 0.0)
                    except Exception:
                        mc = 0.0
                    it["mapping_confidence"] = max(0.9, mc)
                    if not it.get("rationale"):
                        it["rationale"] = "Mapped via Anchor ID"
        # Fill correct_answer from answer key anchors when missing
        if isinstance(evaluation, dict) and isinstance(evaluation.get("items"), list) and key_anchors:
            key_map = {a.get("anchor_id"): a.get("text") for a in key_anchors if a.get("anchor_id")}
            for it in evaluation["items"]:
                if not isinstance(it, dict):
                    continue
                lbl = it.get("label")
                if not lbl:
                    continue
                if (not it.get("correct_answer")) and lbl in key_map:
                    it["correct_answer"] = key_map[lbl]
        # Rule-based correctness override when clear
        if isinstance(evaluation, dict) and isinstance(evaluation.get("items"), list):
            for it in evaluation["items"]:
                if not isinstance(it, dict):
                    continue
                st_ans = it.get("student_answer")
                gold = it.get("correct_answer")
                rc = rule_check_correct(st_ans, gold)
                if rc is True:
                    it["llm_judgement_correct"] = True
                    it.setdefault("points", it.get("points") or 1.0)
                    try:
                        it["points_earned"] = float(it.get("points"))
                    except Exception:
                        it["points_earned"] = 1.0
                    if it.get("rationale"):
                        it["rationale"] = str(it["rationale"]) + " | Rule-check: exact match"
                    else:
                        it["rationale"] = "Rule-check: exact match"
                elif rc is False:
                    if it.get("llm_judgement_correct") is None and it.get("is_marked_correct") is None:
                        it["llm_judgement_correct"] = False
                    if it.get("points_earned") is None:
                        it["points_earned"] = 0.0
                    if it.get("rationale"):
                        it["rationale"] = str(it["rationale"]) + " | Rule-check: mismatch"
                    else:
                        it["rationale"] = "Rule-check: mismatch"
    except Exception:
        pass

    # If step-level details missing but both sides have steps, perform simple rule-based step matching
    try:
        if isinstance(evaluation, dict) and isinstance(evaluation.get("items"), list):
            for it in evaluation["items"]:
                if not isinstance(it, dict):
                    continue
                if it.get("step_evaluation"):
                    continue
                exp_steps = it.get("solution_steps_expected") or []
                stu_steps = it.get("student_steps") or []
                if not isinstance(exp_steps, list) or not isinstance(stu_steps, list):
                    continue
                n = max(len(exp_steps), len(stu_steps))
                if n == 0:
                    continue
                ev = []
                correct_count = 0
                wrong_seen = False
                for i in range(n):
                    e = (exp_steps[i] if i < len(exp_steps) else None) or ""
                    s = (stu_steps[i] if i < len(stu_steps) else None) or ""
                    ok = (re.sub(r"\s+", " ", str(e)).strip().lower() == re.sub(r"\s+", " ", str(s)).strip().lower()) if (e and s) else False
                    if wrong_seen:
                        ok = False
                    row = {
                        "step_index": i+1,
                        "expected_step": e,
                        "student_step": s,
                        "matches_expected": bool(ok),
                        "error_type": None if ok else ("CASCADE_AFTER_WRONG" if wrong_seen else "MISMATCH"),
                        "notes": None,
                    }
                    if not wrong_seen and ok:
                        correct_count += 1
                    if not wrong_seen and not ok:
                        wrong_seen = True
                    ev.append(row)
                it["step_evaluation"] = ev
                # derive partial credit when total points known
                try:
                    pts = float(it.get("points") or 1.0)
                except Exception:
                    pts = 1.0
                earned = pts * (correct_count / n)
                it["points"] = pts
                if it.get("points_earned") is None:
                    it["points_earned"] = round(earned, 2)
                # add rationale note
                note = f"Rule-step-check: {correct_count}/{n} steps matched"
                if it.get("rationale"):
                    it["rationale"] = str(it["rationale"]) + " | " + note
                else:
                    it["rationale"] = note
    except Exception:
        pass

    # Persist results for the suggestion step
    st.session_state.evaluation = evaluation
    st.session_state.gradebook = derive_gradebook(evaluation)
    st.session_state.exam_anchors = exam_anchors
    st.session_state.sub_anchors = sub_anchors
    st.session_state.key_anchors = key_anchors
    st.session_state.ds_ocr_text = ds_ocr_text
    st.session_state.ans_ocr_text = ans_ocr_text
    st.session_state.exam_ocr_text = exam_ocr_text

# -------- Render results if available --------
evaluation = st.session_state.get('evaluation')
gradebook = st.session_state.get('gradebook') or {"entries": [], "totals": {"earned": 0, "total": 0}}
exam_anchors = st.session_state.get('exam_anchors', [])
sub_anchors = st.session_state.get('sub_anchors', [])
key_anchors = st.session_state.get('key_anchors', [])
ds_ocr_text = st.session_state.get('ds_ocr_text')
ans_ocr_text = st.session_state.get('ans_ocr_text')
exam_ocr_text = st.session_state.get('exam_ocr_text')

if isinstance(evaluation, dict) and evaluation:
    # Overview
    st.subheader("Tổng quan")
    gf = parse_goal(goal_text)
    uf = parse_goal(user_text)
    is_geom = detect_geometry_context(evaluation)
    ca, cb, cc = st.columns(3)
    with ca: st.metric("Mục tiêu", f"{round(gf*100):d}%" if isinstance(gf,(int,float)) else "?")
    with cb: st.metric("Điểm của bạn", f"{round(uf*100):d}%" if isinstance(uf,(int,float)) else "?")
    with cc: st.metric("Chủ đề", "Hình học" if is_geom else "Tổng hợp")

    # Gradebook table
    st.subheader("Bảng điểm theo mục")
    ent = gradebook.get("entries", [])
    if ent:
        st.dataframe(
            [{"Mục": e.get("label"), "Điểm": e.get("points_earned"), "/": e.get("points"), "Kỹ năng": e.get("skill_tag"), "Ghi chú": e.get("rationale")} for e in ent],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Chưa có mục nào được trích.")

    # Extracted per-item content (exam vs student vs answer key)
    st.subheader("Bảng trích xuất: Đề – Bài làm – Đáp án")
    items = evaluation.get("items") if isinstance(evaluation, dict) else None
    key_map = {a.get("anchor_id"): a.get("text") for a in (key_anchors or []) if a.get("anchor_id")}
    exam_map = {a.get("anchor_id"): a.get("text") for a in (exam_anchors or []) if a.get("anchor_id")}
    if isinstance(items, list) and items:
        rows = []
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = it.get("label")
            ques = (it.get("question") or ((exam_map.get(lbl, "")) if lbl else ""))
            ans = it.get("student_answer") or ""
            gold = it.get("correct_answer")
            if not gold and lbl:
                gold = extract_final_answer_from_text(key_map.get(lbl, ""))
            try:
                conf = float(it.get("mapping_confidence")) if it.get("mapping_confidence") is not None else None
            except Exception:
                conf = None
            ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
            pts = it.get("points")
            pe = it.get("points_earned")
            # Steps summary
            exp_steps = it.get("solution_steps_expected") or []
            stu_steps = it.get("student_steps") or []
            step_eval = it.get("step_evaluation") or []
            # Count correct steps and first wrong index if available
            step_ok = 0
            first_wrong = None
            if isinstance(step_eval, list) and step_eval:
                for idx, ev in enumerate(step_eval, start=1):
                    if isinstance(ev, dict) and ev.get("matches_expected") is True:
                        step_ok += 1
                    else:
                        if first_wrong is None:
                            first_wrong = idx
            total_steps = max(len(exp_steps) if isinstance(exp_steps, list) else 0,
                              len(stu_steps) if isinstance(stu_steps, list) else 0,
                              len(step_eval) if isinstance(step_eval, list) else 0)
            if total_steps == 0 and isinstance(exp_steps, list):
                total_steps = len(exp_steps)
            # Join first few steps for display
            def _join_steps(lst):
                try:
                    parts = [str(x).strip() for x in (lst or []) if str(x).strip()]
                    s = " | ".join(parts[:3])
                    return s[:90] + ("…" if len(s) > 90 else "")
                except Exception:
                    return ""
            exp_join = _join_steps(exp_steps)
            stu_join = _join_steps(stu_steps)
            comp = f"{step_ok}/{total_steps}"
            if first_wrong is not None:
                comp += f" (sai từ bước {first_wrong})"
            rows.append({
                "Mục": lbl,
                "Đúng?": "✓" if ok else ("✗" if ok is False else "?"),
                "Tin cậy": (f"{round(conf*100):d}%" if isinstance(conf, (int, float)) else "—"),
                "Điểm": (f"{pe}/{pts}" if (pe is not None or pts is not None) else "—"),
                "Đề bài": (str(ques)[:90] + ("…" if len(str(ques)) > 90 else "")),
                "Bài làm": (str(ans)[:90] + ("…" if len(str(ans)) > 90 else "")),
                "Đáp án chuẩn": (str(gold)[:90] + ("…" if len(str(gold)) > 90 else "")),
                "Bước (chuẩn)": exp_join,
                "Bước (HS)": stu_join,
                "Đối chiếu bước": comp if total_steps > 0 else "—",
                "Nhận xét": (st.session_state.get('comments_map', {}).get(lbl, ""))
            })
        # Editable table for per-item comments
        edited = st.data_editor(
            rows,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Nhận xét": st.column_config.TextColumn("Nhận xét", width="medium", help="Nhập nhận xét/nhầm lẫn cho mục này"),
                "Đề bài": st.column_config.TextColumn("Đề bài", width="large"),
                "Bài làm": st.column_config.TextColumn("Bài làm", width="large"),
                "Đáp án chuẩn": st.column_config.TextColumn("Đáp án chuẩn", width="large"),
                "Bước (chuẩn)": st.column_config.TextColumn("Bước (chuẩn)", width="large"),
                "Bước (HS)": st.column_config.TextColumn("Bước (HS)", width="large"),
                "Đối chiếu bước": st.column_config.TextColumn("Đối chiếu bước", width="medium"),
            },
            disabled=["Mục","Đúng?","Tin cậy","Điểm","Đề bài","Bài làm","Đáp án chuẩn","Bước (chuẩn)","Bước (HS)","Đối chiếu bước"],
        )
        # Persist comments map to session for suggestion step
        try:
            st.session_state.comments_map = {row.get("Mục"): row.get("Nhận xét") for row in (edited or []) if row.get("Mục")}
        except Exception:
            st.session_state.comments_map = st.session_state.get('comments_map', {})
        with st.expander("Chi tiết (đầy đủ) theo từng mục"):
            for it in items:
                if not isinstance(it, dict):
                    continue
                lbl = it.get("label") or "—"
                ques = (it.get("question") or ((exam_map.get(lbl, "")) if lbl else "")) or ""
                ans = it.get("student_answer") or ""
                gold = it.get("correct_answer") or extract_final_answer_from_text(key_map.get(lbl, ""))
                ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
                st.markdown(f"### `{lbl}` — {'Đúng' if ok else 'Sai' if ok is False else '?'}")
                if ques:
                    st.caption("Đề bài:")
                    st.code(ques)
                if ans:
                    st.caption("Bài làm:")
                    st.code(ans)
                if gold:
                    st.caption("Đáp án/Chấm điểm:")
                    st.code(gold)
                # Step-by-step comparison when available
                exp_steps = it.get("solution_steps_expected") or []
                stu_steps = it.get("student_steps") or []
                step_eval = it.get("step_evaluation") or []
                if exp_steps or stu_steps or step_eval:
                    st.caption("Đối chiếu từng bước:")
                    rows_steps = []
                    n = max(len(exp_steps), len(stu_steps), len(step_eval))
                    for i in range(n):
                        es = exp_steps[i] if i < len(exp_steps) else ""
                        ss = stu_steps[i] if i < len(stu_steps) else ""
                        ev = step_eval[i] if i < len(step_eval) and isinstance(step_eval[i], dict) else {}
                        m = ev.get("matches_expected")
                        mark = "✓" if m is True else ("✗" if m is False else "?")
                        rows_steps.append({"Bước": i+1, "Đáp án chuẩn": es, "Bước của HS": ss, "Đúng?": mark, "Lỗi": ev.get("error_type") or ""})
                    st.dataframe(rows_steps, use_container_width=True, hide_index=True)
                if it.get("rationale"):
                    st.caption("Nhận xét của hệ thống:")
                    st.write(it.get("rationale"))
                if it.get("rubric"):
                    st.caption("Rubric:")
                    try:
                        st.json(it.get("rubric"))
                    except Exception:
                        st.write(it.get("rubric"))
    else:
        st.info("Chưa có dữ liệu trích xuất từ bài làm.")

    # Nhắc người dùng nhập nhận xét ngay ở cột Nhận xét của bảng trên.
    st.caption("Mẹo: điền nhận xét vào cột 'Nhận xét' của bảng trên cho từng mục.")

    # Debug: OCR and anchors
    if ds_ocr_text:
        with st.expander("DeepSeek OCR (submission)"):
            st.code(ds_ocr_text)
    if exam_ocr_text:
        with st.expander("DeepSeek OCR (exam)"):
            st.code(exam_ocr_text)
    if ans_ocr_text:
        with st.expander("DeepSeek OCR (answer key)"):
            st.code(ans_ocr_text)
    if (exam_anchors or sub_anchors or key_anchors):
        with st.expander("Anchors (OCR-based)"):
            if exam_anchors:
                st.markdown("- Exam anchors:")
                for a in exam_anchors[:20]:
                    st.markdown(f"  - `{a.get('anchor_id')}`: {a.get('text')[:120]}...")
            if key_anchors:
                st.markdown("- Answer key anchors:")
                for a in key_anchors[:20]:
                    st.markdown(f"  - `{a.get('anchor_id')}`: {a.get('text')[:120]}...")
            if sub_anchors:
                st.markdown("- Submission anchors:")
                for a in sub_anchors[:20]:
                    st.markdown(f"  - `{a.get('anchor_id')}`: {a.get('text')[:120]}...")

    with st.expander("JSON evaluation"):
        st.code(json.dumps(evaluation, ensure_ascii=False, indent=2))

# -------- Suggestion step --------
if suggest_clicked:
    if not isinstance(st.session_state.get('evaluation'), dict):
        st.warning("Chưa có kết quả phân tích. Hãy bấm 'Phân tích & Chấm điểm' trước.")
    else:
        model = get_model()
        evaluation = st.session_state.get('evaluation')
        # Build weak skills
        wrong_count: Dict[str, int] = {}
        for it in evaluation.get("items", []) if isinstance(evaluation, dict) else []:
            ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
            if ok:
                continue
            sid = it.get("skill_tag") or it.get("skillId")
            if not sid:
                continue
            wrong_count[sid] = wrong_count.get(sid, 0) + 1
        weak = []
        total_wrong = sum(wrong_count.values())
        if total_wrong > 0:
            for sid, cnt in wrong_count.items():
                weak.append({"skillId": sid, "severity": round(cnt/total_wrong, 2)})
        if not weak:
            weak = [{"skillId": "GENERAL.REVIEW", "severity": 0.5}]

        goal_frac = parse_goal(goal_text)
        user_frac = parse_goal(user_text)
        support_plan = build_support_plan(weak, goal_frac, user_frac, int(max_q))

        # Geometry context
        is_geom = detect_geometry_context(evaluation)
        geom_templates = [
            "a) Chứng minh hai tam giác đồng dạng (ví dụ ΔAHB ~ ΔCAB) bằng góc bằng nhau.",
            "b) Suy ra hệ thức độ dài từ tam giác vuông có đường cao: AH^2 = AM.AB; AB^2 = BH.BC; AC^2 = CH.BC; BH.CH = AH^2.",
            "c) Cho số liệu (vd AB=6cm, BC=10cm), tính AC, AH, BH, CH.",
            "d) Kẻ đường vuông góc/phân giác qua A để tạo giao điểm và chứng minh các hệ thức tỉ lệ đoạn thẳng.",
        ]

        # Collect observed wrong items and user comments
        wrong_items = []
        for it in evaluation.get("items", []) if isinstance(evaluation, dict) else []:
            ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
            if not ok:
                wi = {k: it.get(k) for k in ("label","question","skill_tag","rationale")}
                # include step-level errors if present
                se = []
                for ev in (it.get("step_evaluation") or []):
                    if isinstance(ev, dict) and ev.get("matches_expected") is False:
                        se.append({"step_index": ev.get("step_index"), "error_type": ev.get("error_type"), "student_step": ev.get("student_step"), "expected_step": ev.get("expected_step")})
                if se:
                    wi["step_errors"] = se[:5]
                wrong_items.append(wi)
        # Build user comments per-item map
        comments_map = st.session_state.get('comments_map') or {}

        gen_questions: List[Dict[str, Any]] = []
        if support_plan:
            gen_prompt = {
                "task": "generate_support_practice",
                "instructions": [
                    "Generate short, clear math questions suited for middle school.",
                    "Follow the plan: for each skillId, produce the requested counts per difficulty (easy/medium/hard).",
                    "Calibrate difficulty relative to the exam style; prefer geometry-style problems if topic_hint=geometry.",
                    "You MAY use inline LaTeX ($...$) for formulas so it is readable; keep text concise.",
                    "If geometry, you may include a small inline SVG diagram in 'diagram_svg' (viewBox '0 0 300 200').",
                    "Incorporate the user's error comments per item to target misconceptions.",
                    "Provide final answers and a concise solution_outline; no external images or links.",
                    "Return JSON array only.",
                ],
                "plan": support_plan,
                "topic_hint": "geometry" if is_geom else None,
                "geometry_templates": geom_templates if is_geom else None,
                "observed_errors": wrong_items,
                "user_error_comments_map": comments_map,
                "output_schema": {"type": "array"},
                "locale": lang,
            }
            parts = [json.dumps(gen_prompt)]
            try:
                gresp = model.generate_content(parts)
                raw = getattr(gresp, "text", "[]")
                gen_questions = safe_json_loads(raw, [])
                if not isinstance(gen_questions, list):
                    gen_questions = []
            except Exception as e:
                st.warning(f"Lỗi sinh câu hỏi: {e}")
                gen_questions = []

        # Hints per wrong item
        hint_questions: List[Dict[str, Any]] = []
        if wrong_items:
            hints_prompt = {
                "task": "generate_guiding_questions",
                "instructions": [
                    "For each wrong item, write 1-2 short guiding questions (Socratic hints) that nudge the student to the next step, without giving the final answer.",
                    "Keep language concise for grade 8 Vietnamese math.",
                    "Refer to items by their labels like 'B1.a'.",
                    "Use user's comments per item to tailor hints when relevant.",
                    "Return JSON array only.",
                ],
                "wrong_items": wrong_items,
                "user_error_comments_map": comments_map,
                "max_hints": min(8, max(3, len(wrong_items))),
                "goal_fraction": goal_frac,
                "output_schema": {"type": "array"},
                "locale": lang,
            }
            try:
                hresp = model.generate_content([json.dumps(hints_prompt)])
                raw = getattr(hresp, "text", "[]")
                hint_questions = safe_json_loads(raw, [])
                if not isinstance(hint_questions, list):
                    hint_questions = []
            except Exception:
                hint_questions = []

        # Persist and render
        st.session_state.gen_questions = gen_questions
        st.session_state.hint_questions = hint_questions

# Render suggestions if any
gen_questions = st.session_state.get('gen_questions') or []
hint_questions = st.session_state.get('hint_questions') or []
if gen_questions or hint_questions:
    st.subheader("Gợi ý theo từng mục sai")
    if hint_questions:
        for h in hint_questions:
            st.markdown(f"- `{h.get('label')}`: " + "; ".join([str(x) for x in (h.get("hints") or [])]))
    else:
        st.info("Chưa có gợi ý.")

    st.subheader("Câu hỏi luyện tập được gợi ý")
    if gen_questions:
        for i, q in enumerate(gen_questions, 1):
            skill = q.get('skillId') or ''
            question = q.get('question') or ''
            st.markdown(f"{i}. [{skill}] {question}")
            # Optional diagram
            svg = q.get('diagram_svg')
            if isinstance(svg, str) and svg.strip():
                components.html(svg, height=240)
            elif q.get('diagram_description'):
                st.caption("Sơ đồ: " + str(q.get('diagram_description')))
            ans = q.get('answer')
            if ans is not None:
                st.markdown(f"   Đáp án: {ans}")
            if q.get("solution_outline"):
                st.caption("Gợi ý lời giải: " + str(q.get("solution_outline")))
    else:
        st.info("Chưa sinh được câu hỏi. Hãy nhập nhận xét và ấn Gợi ý luyện tập.")
