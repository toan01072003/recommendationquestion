import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Import từ edurec_ui modules thay vì duplicate code
from edurec_ui.services.gemini import (
    get_model as _get_model_base,
    upload_bytes_to_gemini,
    wait_until_files_active,
)
from edurec_ui.services.backend import call_deepseek_ocr_api
from edurec_ui.utils.anchors import build_anchors_from_text

# Load .env for GOOGLE_API_KEY when running locally
load_dotenv(override=False)

st.set_page_config(page_title="EduRec - Phân tích bài làm AI", page_icon="📚", layout="wide")
st.title("📚 EduRec - Hệ thống phân tích bài làm thông minh")
st.caption("Pipeline AI tự động: Phân đề → Chấm điểm → Phân tích điểm yếu → Gợi ý luyện tập")


# -------------------- Gemini helpers --------------------
@st.cache_resource
def get_model():
    """Cached Gemini model để tránh re-init mỗi lần chạy."""
    try:
        return _get_model_base()
    except RuntimeError as e:
        st.error(f"Lỗi khởi tạo Gemini: {e}")
        st.stop()


def is_likely_image_bytes(b: bytes) -> bool:
    """Kiểm tra bytes có phải là ảnh hợp lệ không."""
    if not b or len(b) < 4:
        return False
    # PNG
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    # JPEG
    if b.startswith(b"\xff\xd8"):
        return True
    # GIF
    if b.startswith(b"GIF8"):
        return True
    # WebP
    if b.startswith(b"RIFF") and len(b) > 12 and b[8:12] == b"WEBP":
        return True
    return False


# -------------------- Scoring helpers --------------------
def parse_goal(goal_text: Optional[str]) -> Optional[float]:
    if not goal_text:
        return None
    t = goal_text.strip().replace(" ", "")
    try:
        if t.endswith("%"):
            return max(0.0, min(1.0, float(t[:-1]) / 100.0))
        if "/" in t:
            a, b = t.split("/", 1)
            a = float(a)
            b = float(b)
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
        label = str(it.get("label") or "")
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
            "parent_label": it.get("parent_label"),
            "skill_tag": it.get("skill_tag") or it.get("skillId"),
            "points": pts,
            "points_earned": pe,
            "rationale": it.get("rationale"),
            "rubric": it.get("rubric"),
        })
    return {"entries": entries, "totals": {"earned": earned, "total": total}}


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
    delta = None
    if goal_frac is not None and user_frac is not None:
        delta = max(-1.0, min(1.0, float(goal_frac) - float(user_frac)))
    plan = []
    sev_map = {w.get("skillId"): float(w.get("severity", 0.3)) for w in weak if w.get("skillId")}
    for sid, count in alloc:
        sev = sev_map.get(sid, 0.3)
        if sev >= 0.7:
            mix = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
        elif sev >= 0.4:
            mix = {"easy": 0.1, "medium": 0.5, "hard": 0.4}
        else:
            mix = {"easy": 0.05, "medium": 0.45, "hard": 0.5}
        if delta is not None:
            if delta >= 0.2:
                mix["easy"] = max(0.0, mix["easy"] - 0.1)
                mix["hard"] = min(0.7, mix["hard"] + 0.1)
            elif delta <= -0.1:
                mix["easy"] = min(0.5, mix["easy"] + 0.1)
                mix["hard"] = max(0.1, mix["hard"] - 0.1)
        e = max(0, round(count * mix["easy"]))
        m = max(0, round(count * mix["medium"]))
        h = max(0, round(count * mix["hard"]))
        tot = e + m + h
        while tot < count:
            if m <= max(e, h): m += 1
            elif e <= h: e += 1
            else: h += 1
            tot = e + m + h
        while tot > count:
            if m >= max(e, h) and m > 0: m -= 1
            elif e >= h and e > 0: e -= 1
            elif h > 0: h -= 1
            tot = e + m + h
        plan.append({"skillId": sid, "counts": {"easy": int(e), "medium": int(m), "hard": int(h)}})
    return plan


# -------------------- Sidebar: settings and uploads --------------------
with st.sidebar:
    st.subheader("Thiết lập")
    lang = st.selectbox("Ngôn ngữ", options=["vi", "en"], index=0)
    goal_text = st.text_input("Mục tiêu điểm (vd 8/10 hoặc 80%)", value="")
    user_text = st.text_input("Điểm của bạn (vd 6/10 hoặc 60%)", value="")
    max_q = st.number_input("Số câu luyện gợi ý", min_value=1, max_value=20, value=6)
    st.markdown("<hr>", unsafe_allow_html=True)
    exams = st.file_uploader("Ảnh đề (1 hoặc nhiều trang)", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=True)
    subs = st.file_uploader("Ảnh bài làm (1 hoặc nhiều trang)", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=True)
    clear = st.button("Bắt đầu phiên mới")


if clear:
    st.session_state.clear()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Chào bạn! Tải ảnh đề và bài làm vào sidebar bên trái, nhập mục tiêu điểm, rồi gõ bất kỳ để bắt đầu pipeline phân tích:\n\n1️⃣ **Phân đề** - Tách câu hỏi thành các mục B1.a, B1.b...\n2️⃣ **Chấm điểm** - Đánh giá từng câu với AI\n3️⃣ **Phân tích** - Xác định kỹ năng yếu\n4️⃣ **Gợi ý** - Tạo bài luyện tập ZPD phù hợp"}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"]) 


def run_pipeline(user_prompt: str, progress_bar=None):
    """Pipeline chính để phân tích và chấm điểm bài làm."""
    model = get_model()

    def update_progress(value: float, text: str):
        if progress_bar:
            progress_bar.progress(value, text)

    # Step 1: Upload files to Gemini
    update_progress(0.1, "Đang tải ảnh lên Gemini...")
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
        return {"assistant": "Mình cần ít nhất một ảnh đề hoặc bài làm để phân tích."}

    # Step 2: Wait for files to be active
    update_progress(0.2, "Đang chờ Gemini xử lý ảnh...")
    try:
        wait_until_files_active(exam_refs + sub_refs)
    except RuntimeError as e:
        return {"assistant": f"Lỗi khi xử lý ảnh: {e}"}

    # Step 3: OCR via DeepSeek backend (optional)
    update_progress(0.3, "Đang trích xuất văn bản (OCR)...")
    api_base = os.environ.get("EDUREC_API_BASE", "http://localhost:8000")
    ds_ocr_text = None
    exam_ocr_text = None
    ocr_warning = None

    texts = []
    for f in (subs or []):
        data = f.getvalue()
        if is_likely_image_bytes(data):
            t = call_deepseek_ocr_api(api_base, f.name, data, language=lang)
            if t:
                texts.append(t)
    if texts:
        ds_ocr_text = "\n\n---\n\n".join(texts)

    # OCR exam as well to build anchors on exam side
    etexts = []
    for f in (exams or []):
        data = f.getvalue()
        if is_likely_image_bytes(data):
            t = call_deepseek_ocr_api(api_base, f.name, data, language=lang)
            if t:
                etexts.append(t)
    if etexts:
        exam_ocr_text = "\n\n---\n\n".join(etexts)

    # Warn if OCR failed but continue with Gemini vision
    if not ds_ocr_text and not exam_ocr_text:
        ocr_warning = "OCR backend không khả dụng, sử dụng Gemini vision trực tiếp."

    # Step 4: Build anchors and evaluate
    update_progress(0.4, "Đang phân tích cấu trúc đề...")
    exam_anchors = build_anchors_from_text(exam_ocr_text) if exam_ocr_text else []
    sub_anchors = build_anchors_from_text(ds_ocr_text) if ds_ocr_text else []

    # Step 5: Evaluate with Gemini
    update_progress(0.5, "Đang chấm điểm với Gemini...")
    evaluation = {}
    eval_prompt = {
        "task": "evaluate_submission_items",
        "instructions": [
            "Parse the exam into big questions 'Bài x' and subparts 'a,b,c'.",
            "Normalize labels as 'B1.a', 'B1.b', ... (ASCII only).",
            "Extract short 'question' texts for each leaf.",
            "Map student's answers from the submission to these leaf labels; set mapping_confidence.",
            "Judge correctness; if no printed points, default 1 point per leaf.",
            "Add a free-form 'skill_tag' for the math topic (e.g., FRAC.SIMPLIFY, EQ.SOLVE_1VAR).",
            "When wrong or partial, produce a brief rationale and a rubric with 2–4 criteria.",
            "If anchors_hint is provided, prefer mapping by exact anchor_id equality (e.g., 'B1.a'), set mapping_confidence >= 0.9 and copy that anchor text into student_answer.",
            "Return JSON only.",
        ],
        "output_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "parent_label": {"type": ["string","null"]},
                            "question": {"type": ["string","null"]},
                            "skill_tag": {"type": ["string","null"]},
                            "student_answer": {"type": ["string","null"]},
                            "mapping_confidence": {"type": ["number","null"]},
                            "is_marked_correct": {"type": ["boolean","null"]},
                            "llm_judgement_correct": {"type": ["boolean","null"]},
                            "points": {"type": ["number","null"]},
                            "points_earned": {"type": ["number","null"]},
                            "rationale": {"type": ["string","null"]},
                            "rubric": {"type": ["array","null"], "items": {"type": "object"}},
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "locale": lang,
    }
    # Provide OCR hint to the LLM if available
    if ds_ocr_text:
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
        evaluation = {"error": str(e)}

    # Post-process: fill missing student_answer via anchors when possible
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
                    try:
                        mc = float(it.get("mapping_confidence") or 0.0)
                    except Exception:
                        mc = 0.0
                    it["mapping_confidence"] = max(0.9, mc)
                    if not it.get("rationale"):
                        it["rationale"] = "Mapped via Anchor ID"
    except Exception:
        pass

    # Gradebook + weak skills
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
    tot_wrong = sum(wrong_count.values())
    if tot_wrong > 0:
        for sid, cnt in wrong_count.items():
            weak.append({"skillId": sid, "severity": round(cnt / tot_wrong, 2)})
    if not weak:
        weak = [{"skillId": "GENERAL.REVIEW", "severity": 0.5}]

    # Step 6: Build support plan and generate practice
    update_progress(0.7, "Đang tạo bài luyện tập...")
    goal_frac = parse_goal(goal_text)
    user_frac = parse_goal(user_text)
    support_plan = build_support_plan(weak, goal_frac, user_frac, int(max_q))

    gen_questions: List[Dict[str, Any]] = []
    if support_plan:
        gen_prompt = {
            "task": "generate_support_practice",
            "instructions": [
                "Generate short, clear math questions suited for middle school.",
                "Follow the plan: for each skillId, produce the requested counts per difficulty (easy/medium/hard).",
                "Calibrate difficulty relative to the exam style.",
                "If geometry (hình học), include a small inline SVG diagram in 'diagram_svg' (viewBox '0 0 300 200'), using simple shapes and text labels A,B,C,...",
                "If SVG is not feasible, include 'diagram_description'.",
                "Provide final answers and a concise solution_outline; avoid LaTeX and images.",
                "Return JSON array only.",
            ],
            "plan": support_plan,
            "goal_fraction": goal_frac,
            "user_score_fraction": user_frac,
            "observed_errors": [
                {k: it.get(k) for k in ("label","question","skill_tag","student_answer","rationale")}
                for it in (evaluation.get("items") if isinstance(evaluation, dict) else []) if isinstance(it, dict)
            ],
            "output_schema": {"type": "array"},
            "locale": lang,
        }
        parts = [json.dumps(gen_prompt)] + exam_refs + sub_refs
        try:
            gresp = model.generate_content(parts)
            gen_questions = json.loads(getattr(gresp, "text", "[]"))
            if not isinstance(gen_questions, list):
                gen_questions = []
        except Exception:
            gen_questions = []

    # Step 7: Generate hints for wrong items
    update_progress(0.85, "Đang tạo gợi ý Socratic...")
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

    # Step 8: Compose final result
    update_progress(1.0, "Hoàn tất!")
    earned = gradebook.get("totals", {}).get("earned")
    total = gradebook.get("totals", {}).get("total") or 0
    acc = f"{(earned/total*100):.0f}%" if total > 0 else "?"
    gf = f"{round((goal_frac or 0)*100):d}%" if goal_frac is not None else "?"
    msg = f"Mình đã phân tích đề theo từng Bài/ý, ghép bài làm và chấm điểm. Kết quả tạm: {earned}/{total} (~{acc}). Mục tiêu: {gf}. Dưới đây là gợi ý theo từng mục sai và bộ câu luyện tập."

    return {
        "assistant": msg,
        "evaluation": evaluation,
        "gradebook": gradebook,
        "hints": hint_questions,
        "practice": gen_questions,
        "ocr_text": ds_ocr_text,
        "ocr_warning": ocr_warning,
    }


# -------------------- Chat input --------------------
if user := st.chat_input("Nhập tin nhắn để bắt đầu phân tích…"):
    st.session_state.messages.append({"role": "user", "content": user})
    with st.chat_message("user"):
        st.write(user)

    # Sử dụng progress bar thay vì spinner
    progress_bar = st.progress(0, "Đang khởi tạo...")
    result = run_pipeline(user, progress_bar=progress_bar)
    progress_bar.empty()  # Xóa progress bar sau khi hoàn tất

    with st.chat_message("assistant"):
        # Hiển thị warning nếu OCR không khả dụng
        if result.get("ocr_warning"):
            st.warning(result["ocr_warning"])

        st.write(result.get("assistant"))

        # Gradebook
        gb = result.get("gradebook", {})
        ent = gb.get("entries", [])
        if ent:
            st.subheader("Bảng điểm theo mục")
            st.dataframe(
                [
                    {
                        "Mục": e.get("label"),
                        "Điểm": e.get("points_earned"),
                        "/": e.get("points"),
                        "Kỹ năng": e.get("skill_tag"),
                        "Ghi chú": e.get("rationale"),
                    }
                    for e in ent
                ],
                use_container_width=True,
                hide_index=True,
            )

        # Hints
        hints = result.get("hints", [])
        if hints:
            st.subheader("Gợi ý theo từng mục sai")
            for h in hints:
                hint_list = h.get("hints") or []
                hint_text = "; ".join([str(x) for x in hint_list]) if hint_list else "Không có gợi ý"
                st.markdown(f"- **{h.get('label')}**: {hint_text}")

        # Practice
        qs = result.get("practice", [])
        if qs:
            st.subheader("Câu luyện tập gợi ý")
            for i, q in enumerate(qs, 1):
                with st.container():
                    st.markdown(f"**{i}. [{q.get('skillId')}]** {q.get('question')}")
                    st.markdown(f"*Đáp án:* {q.get('answer')}")
                    if q.get("solution_outline"):
                        st.markdown(f"*Gợi ý lời giải:* {q.get('solution_outline')}")
                    svg = q.get("diagram_svg")
                    if isinstance(svg, str) and svg.strip():
                        components.html(svg, height=240)
                    elif q.get("diagram_description"):
                        st.caption("Sơ đồ: " + str(q.get("diagram_description")))

        # Expandable sections
        with st.expander("JSON chi tiết (evaluation)"):
            st.code(json.dumps(result.get("evaluation"), ensure_ascii=False, indent=2))

        if result.get("ocr_text"):
            with st.expander("DeepSeek OCR (submission)"):
                st.code(result.get("ocr_text"))

    st.session_state.messages.append({"role": "assistant", "content": result.get("assistant")})
