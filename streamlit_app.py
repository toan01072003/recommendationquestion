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
    subs = st.file_uploader("Ảnh bài làm (1 hoặc nhiều trang)", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=True)


if st.button("Phân tích & Gợi ý"):
    model = get_model()

    # Upload files
    exam_refs = []
    sub_refs = []
    for f in (exams or []):
        data = f.getvalue()
        if is_likely_image_bytes(data):
            exam_refs.append(upload_bytes_to_gemini(f.name, data))
    for f in (subs or []):
        data = f.getvalue()
        if is_likely_image_bytes(data):
            sub_refs.append(upload_bytes_to_gemini(f.name, data))

    if not exam_refs and not sub_refs:
        st.warning("Cần ít nhất 1 ảnh đề hoặc bài làm.")
        st.stop()

    try:
        wait_until_files_active(exam_refs + sub_refs)
    except Exception:
        pass

    # Optional: DeepSeek OCR via backend for submission images
    api_base = os.environ.get("EDUREC_API_BASE", "http://localhost:8000")
    ds_ocr_text = None
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
        exam_ocr_text = None

    # Evaluate: parse exam into Bài/ý and map answers
    evaluation: Dict[str, Any] = {}
    # Build anchors from OCR (if available)
    exam_anchors = build_anchors_from_text(exam_ocr_text) if exam_ocr_text else []
    sub_anchors = build_anchors_from_text(ds_ocr_text) if ds_ocr_text else []

    eval_prompt = {
        "task": "evaluate_submission_items",
        "instructions": [
            "Parse the exam into big questions 'Bài x' and subparts 'a,b,c'.",
            "Normalize labels as 'B1.a', 'B1.b', ... (ASCII only).",
            "Extract short 'question' texts for each leaf.",
            "Map student's answers from the submission to these leaf labels; set mapping_confidence.",
            "Judge correctness; if no printed points, default 1 point per leaf.",
            "Add a free-form 'skill_tag' for the math topic (e.g., GEOM.SIMILARITY, GEOM.ALTITUDE, FRAC.SIMPLIFY, EQ.SOLVE_1VAR).",
            "If anchors_hint is provided, prefer mapping by exact anchor_id equality (e.g., anchor 'B1.a' -> label 'B1.a'); when used, set mapping_confidence >= 0.9 and copy the corresponding anchor text into student_answer.",
            "Return JSON only.",
        ],
        "output_schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array"}
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "locale": lang,
    }
    # Provide OCR hint to the LLM if available
    if 'ds_ocr_text' in locals() and ds_ocr_text:
        try:
            eval_prompt["ocr_hint"] = ds_ocr_text
            if isinstance(eval_prompt.get("instructions"), list):
                eval_prompt["instructions"].append("If ocr_hint is provided, use it to improve mapping and cleaner text extraction.")
        except Exception:
            pass
    # Provide anchors hint when available
    try:
        if exam_anchors or sub_anchors:
            eval_prompt["anchors_hint"] = {"exam": exam_anchors, "submission": sub_anchors}
    except Exception:
        pass
    parts_ev: List[Any] = [json.dumps(eval_prompt)] + exam_refs + sub_refs
    try:
        evresp = model.generate_content(parts_ev)
        evaluation = json.loads(getattr(evresp, "text", "{}"))
        if not isinstance(evaluation, dict):
            evaluation = {"raw": evaluation}
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
    except Exception:
        pass

    # Gradebook & weak skills
    gradebook = derive_gradebook(evaluation)
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

    # Geometry booster: if exam context suggests geometry, instruct generator accordingly
    is_geom = detect_geometry_context(evaluation)
    geom_templates = [
        "a) Chứng minh hai tam giác đồng dạng (ví dụ ΔAHB ~ ΔCAB) bằng góc bằng nhau.",
        "b) Suy ra hệ thức độ dài từ tam giác vuông có đường cao: AH^2 = AM.AB; AB^2 = BH.BC; AC^2 = CH.BC; BH.CH = AH^2.",
        "c) Cho số liệu (vd AB=6cm, BC=10cm), tính AC, AH, BH, CH.",
        "d) Kẻ đường vuông góc/phân giác qua A để tạo giao điểm và chứng minh các hệ thức tỉ lệ đoạn thẳng.",
    ]

    gen_questions: List[Dict[str, Any]] = []
    if support_plan:
        gen_prompt = {
            "task": "generate_support_practice",
            "instructions": [
                "Generate short, clear math questions suited for middle school.",
                "Follow the plan: for each skillId, produce the requested counts per difficulty (easy/medium/hard).",
                "Calibrate difficulty relative to the exam style; prefer geometry-style problems if topic_hint=geometry.",
                "For geometry, DO NOT draw diagrams; write text-only problems similar to school exams with subparts a), b), c).",
                "Use right-triangle altitude and similarity facts when appropriate.",
                "Provide final answers and a concise solution_outline; avoid LaTeX and images.",
                "Return JSON array only.",
            ],
            "plan": support_plan,
            "topic_hint": "geometry" if is_geom else None,
            "geometry_templates": geom_templates if is_geom else None,
            "output_schema": {"type": "array"},
            "locale": lang,
        }
        parts = [json.dumps(gen_prompt)] + exam_refs + sub_refs
        try:
            gresp = model.generate_content(parts)
            gen_questions = json.loads(getattr(gresp, "text", "[]"))
            if not isinstance(gen_questions, list):
                gen_questions = []
        except Exception as e:
            st.warning(f"Lỗi sinh câu hỏi: {e}")
            gen_questions = []

    # Hints per wrong item
    hint_questions: List[Dict[str, Any]] = []
    wrong_items = []
    if isinstance(evaluation, dict) and isinstance(evaluation.get("items"), list):
        for it in evaluation["items"]:
            ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
            if not ok:
                wrong_items.append({k: it.get(k) for k in ("label","question","skill_tag","rationale")})
    if wrong_items:
        hints_prompt = {
            "task": "generate_guiding_questions",
            "instructions": [
                "For each wrong item, write 1-2 short guiding questions (Socratic hints) that nudge the student to the next step, without giving the final answer.",
                "Keep language concise for grade 8 Vietnamese math; avoid LaTeX.",
                "Refer to items by their labels like 'B1.a'.",
                "Return JSON array only.",
            ],
            "wrong_items": wrong_items,
            "max_hints": min(8, max(3, len(wrong_items))),
            "goal_fraction": goal_frac,
            "output_schema": {"type": "array"},
            "locale": lang,
        }
        try:
            hresp = model.generate_content([json.dumps(hints_prompt)])
            hint_questions = json.loads(getattr(hresp, "text", "[]"))
            if not isinstance(hint_questions, list):
                hint_questions = []
        except Exception:
            hint_questions = []

    # -------- Render --------
    st.subheader("Tổng quan")
    gf = parse_goal(goal_text)
    uf = parse_goal(user_text)
    ca, cb, cc = st.columns(3)
    with ca: st.metric("Mục tiêu", f"{round(gf*100):d}%" if isinstance(gf,(int,float)) else "?")
    with cb: st.metric("Điểm của bạn", f"{round(uf*100):d}%" if isinstance(uf,(int,float)) else "?")
    with cc: st.metric("Chủ đề", "Hình học" if is_geom else "Tổng hợp")

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

    # Show per-item extracted answer content
    st.subheader("Phần bài làm đã trích theo từng mục")
    items = evaluation.get("items") if isinstance(evaluation, dict) else None
    if isinstance(items, list) and items:
        rows = []
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = it.get("label")
            ans = it.get("student_answer") or ""
            try:
                conf = float(it.get("mapping_confidence")) if it.get("mapping_confidence") is not None else None
            except Exception:
                conf = None
            ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
            rows.append({
                "Mục": lbl,
                "Đúng?": "✓" if ok else ("✗" if ok is False else "?"),
                "Tin cậy": (f"{round(conf*100):d}%" if isinstance(conf, (int, float)) else "—"),
                "Trích xuất": (ans[:180] + ("…" if len(ans) > 180 else ""))
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        with st.expander("Chi tiết trích xuất (đầy đủ)"):
            for it in items:
                if not isinstance(it, dict):
                    continue
                lbl = it.get("label") or "—"
                ans = it.get("student_answer") or ""
                if not ans:
                    continue
                st.markdown(f"`{lbl}`")
                st.code(ans)
    else:
        st.info("Chưa có dữ liệu trích xuất từ bài làm.")

    st.subheader("Gợi ý theo từng mục sai")
    if hint_questions:
        for h in hint_questions:
            st.markdown(f"- `{h.get('label')}`: " + "; ".join([str(x) for x in (h.get("hints") or [])]))
    else:
        st.info("Chưa có gợi ý.")

    st.subheader("Câu hỏi luyện tập được gợi ý")
    if gen_questions:
        for i, q in enumerate(gen_questions, 1):
            st.markdown(f"{i}. [{q.get('skillId')}] {q.get('question')}")
            st.markdown(f"   Đáp án: {q.get('answer')}")
            if q.get("solution_outline"):
                st.caption("Gợi ý lời giải: " + str(q.get("solution_outline")))
    else:
        st.info("Chưa sinh được câu hỏi. Hãy thử tăng số câu hoặc tải ảnh rõ hơn.")

    if 'ds_ocr_text' in locals() and ds_ocr_text:
        with st.expander("DeepSeek OCR (submission)"):
            st.code(ds_ocr_text)
    # Show anchors for debugging
    if (exam_anchors or sub_anchors):
        with st.expander("Anchors (OCR-based)"):
            if exam_anchors:
                st.markdown("- Exam anchors:")
                for a in exam_anchors[:20]:
                    st.markdown(f"  - `{a.get('anchor_id')}`: {a.get('text')[:120]}...")
            if sub_anchors:
                st.markdown("- Submission anchors:")
                for a in sub_anchors[:20]:
                    st.markdown(f"  - `{a.get('anchor_id')}`: {a.get('text')[:120]}...")

    with st.expander("JSON evaluation"):
        st.code(json.dumps(evaluation, ensure_ascii=False, indent=2))
