"""EduRec Demo v2 - Single Page App (Một trang duy nhất)"""

import os
import json
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Import từ edurec_ui modules
from edurec_ui.services.gemini import (
    get_model as _get_model_base,
    upload_bytes_to_gemini,
    wait_until_files_active,
)
from edurec_ui.services.backend import call_deepseek_ocr_api, call_evaluate_with_key_api
from edurec_ui.utils.anchors import build_anchors_from_text

# Load environment variables
load_dotenv(override=False)

# Page configuration
st.set_page_config(
    page_title="EduRec Demo v2",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------- Helpers --------------------
@st.cache_resource
def get_model():
    """Cached Gemini model."""
    try:
        return _get_model_base()
    except RuntimeError as e:
        st.error(f"❌ Lỗi khởi tạo Gemini: {e}")
        st.stop()


def is_likely_image_bytes(b: bytes) -> bool:
    if not b or len(b) < 4:
        return False
    return (
        b.startswith(b"\x89PNG\r\n\x1a\n")
        or b.startswith(b"\xff\xd8")
        or b.startswith(b"GIF8")
        or (b.startswith(b"RIFF") and len(b) > 12 and b[8:12] == b"WEBP")
    )


def parse_goal(goal_text: Optional[str]) -> Optional[float]:
    if not goal_text:
        return None
    t = goal_text.strip().replace(" ", "")
    try:
        if t.endswith("%"):
            return max(0.0, min(1.0, float(t[:-1]) / 100.0))
        if "/" in t:
            a, b = t.split("/", 1)
            a, b = float(a), float(b)
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

        pts = float(it.get("points") or 1.0)
        pe = it.get("points_earned")
        if pe is None:
            ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
            pe = pts if ok else 0.0
        else:
            pe = float(pe)

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


# -------------------- Header --------------------
st.title("📚 EduRec Demo v2")
st.caption("Hệ thống đánh giá và đề xuất học tập thông minh cho Toán THCS")

# Settings expander
with st.expander("⚙️ Cài đặt", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        api_base = st.text_input(
            "API base URL",
            value=os.environ.get("EDUREC_API_BASE", "http://localhost:8000"),
        )
    with col2:
        lang = st.selectbox("Ngôn ngữ", options=["vi", "en"], index=0)
    with col3:
        gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            st.success("✅ Gemini API")
        else:
            st.error("❌ Gemini API")

st.markdown("---")

# ==================== SECTION 1: CHATBOT AI ====================
st.header("🤖 Chatbot AI - Chấm điểm & Gợi ý luyện tập")

col_left, col_right = st.columns([3, 2])

with col_left:
    exams = st.file_uploader(
        "📄 Ảnh đề thi (có thể nhiều trang)",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=True,
        key="exam",
        help="Tải lên 1 hoặc nhiều ảnh đề bài",
    )

    subs = st.file_uploader(
        "✏️ Ảnh bài làm (có thể nhiều trang)",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=True,
        key="sub",
        help="Tải lên 1 hoặc nhiều ảnh bài làm của học sinh",
    )

with col_right:
    st.markdown("**🎯 Thông tin điểm**")
    goal_text = st.text_input("Mục tiêu", value="8/10", help="VD: 8/10, 80%")
    user_text = st.text_input("Điểm hiện tại", value="", help="VD: 6/10, 60% (tùy chọn)")
    max_q = st.number_input("Số câu luyện", min_value=1, max_value=20, value=6)

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("🚀 Bắt đầu phân tích", type="primary", use_container_width=True):
    if not exams and not subs:
        st.warning("⚠️ Vui lòng tải lên ít nhất 1 ảnh đề hoặc bài làm!")
    else:
        with st.spinner("⏳ Đang phân tích..."):
            model = get_model()

            # Upload images to Gemini
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

            # Wait for processing
            try:
                wait_until_files_active(exam_refs + sub_refs)
            except RuntimeError as e:
                st.error(f"❌ Lỗi: {e}")
                st.stop()

            # Evaluate with Gemini
            eval_prompt = {
                "task": "evaluate_submission_items",
                "instructions": [
                    "Parse exam into questions and subparts (B1.a, B1.b, ...).",
                    "Evaluate student answers and assign points.",
                    "Identify weak skills.",
                    "Return JSON with items array.",
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
                                    "question": {"type": "string"},
                                    "skill_tag": {"type": "string"},
                                    "student_answer": {"type": "string"},
                                    "is_marked_correct": {"type": "boolean"},
                                    "llm_judgement_correct": {"type": "boolean"},
                                    "points": {"type": "number"},
                                    "points_earned": {"type": "number"},
                                    "rationale": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "locale": lang,
            }

            parts = [json.dumps(eval_prompt)] + exam_refs + sub_refs
            evresp = model.generate_content(parts)

            try:
                evaluation = json.loads(getattr(evresp, "text", "{}"))
            except:
                evaluation = {"items": []}

            # Derive gradebook
            gradebook = derive_gradebook(evaluation)
            earned = gradebook.get("totals", {}).get("earned", 0)
            total = gradebook.get("totals", {}).get("total", 0)
            percent = (earned / total * 100) if total > 0 else 0

        # Display results
        st.success(f"### ✅ Hoàn tất!")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Điểm đạt được", f"{earned:.1f}/{total:.1f}")
        with col2:
            st.metric("Phần trăm", f"{percent:.0f}%")
        with col3:
            goal_frac = parse_goal(goal_text)
            goal_percent = (goal_frac * 100) if goal_frac else 0
            st.metric("Mục tiêu", f"{goal_percent:.0f}%")

        # Gradebook table
        entries = gradebook.get("entries", [])
        if entries:
            st.subheader("📊 Bảng điểm chi tiết")
            st.dataframe(
                [
                    {
                        "Câu": e.get("label"),
                        "Kỹ năng": e.get("skill_tag", ""),
                        "Điểm": f"{e.get('points_earned')}/{e.get('points')}",
                        "Nhận xét": e.get("rationale", ""),
                    }
                    for e in entries
                ],
                use_container_width=True,
                hide_index=True,
            )

        # JSON details
        with st.expander("🔍 Xem JSON chi tiết"):
            st.json(evaluation)

st.markdown("---")

# ==================== SECTION 2: OCR & ANCHORS ====================
st.header("📝 OCR & Phân tích cấu trúc")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_ocr = st.file_uploader(
        "Chọn ảnh",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="ocr",
    )

    if uploaded_ocr:
        st.image(uploaded_ocr, caption="Preview", use_container_width=True)

with col2:
    if uploaded_ocr and st.button("🔍 Trích xuất văn bản", key="ocr_btn"):
        with st.spinner("Đang OCR..."):
            img_bytes = uploaded_ocr.read()
            text = call_deepseek_ocr_api(api_base, uploaded_ocr.name, img_bytes, language=lang)

        if not text:
            st.error("❌ OCR thất bại. Kiểm tra backend.")
        else:
            st.success(f"✅ Trích xuất thành công! ({len(text)} ký tự)")

            # Display OCR
            st.text_area("Văn bản trích xuất", value=text, height=200, disabled=True)

            # Build anchors
            anchors = build_anchors_from_text(text)
            if anchors:
                st.info(f"🎯 Phát hiện {len(anchors)} anchor(s)")
                for seg in anchors:
                    with st.expander(f"**{seg.get('anchor_id', '?')}**"):
                        st.text(seg.get("text", "(trống)"))

st.markdown("---")

# ==================== SECTION 3: ĐÁNH GIÁ BÀI LÀM ====================
st.header("✅ Đánh giá bài làm chi tiết")

col1, col2, col3 = st.columns(3)

with col1:
    exam_eval = st.file_uploader("📄 Ảnh đề", type=["png", "jpg", "jpeg", "webp", "gif"], key="e_exam")
    if exam_eval:
        st.image(exam_eval, use_container_width=True)

with col2:
    key_eval = st.file_uploader("📋 Ảnh đáp án", type=["png", "jpg", "jpeg", "webp", "gif"], key="e_key")
    if key_eval:
        st.image(key_eval, use_container_width=True)

with col3:
    sub_eval = st.file_uploader("✏️ Ảnh bài nộp *", type=["png", "jpg", "jpeg", "webp", "gif"], key="e_sub")
    if sub_eval:
        st.image(sub_eval, use_container_width=True)

if st.button("🎯 Đánh giá ngay", type="primary", disabled=not sub_eval, use_container_width=True):
    with st.spinner("Đang đánh giá..."):
        exam_b = exam_eval.read() if exam_eval else None
        key_b = key_eval.read() if key_eval else None
        sub_b = sub_eval.read()

        result = call_evaluate_with_key_api(
            api_base,
            exam_b,
            exam_eval.name if exam_eval else None,
            key_b,
            key_eval.name if key_eval else None,
            sub_b,
            sub_eval.name if sub_eval else None,
            language=lang,
        )

    if not result:
        st.error("❌ Không nhận được kết quả.")
    else:
        # Calculate score
        if isinstance(result, dict) and "items" in result:
            items = result.get("items", [])
            total_pts = sum(float(it.get("points") or 1) for it in items if isinstance(it, dict))
            earned_pts = sum(float(it.get("points_earned") or 0) for it in items if isinstance(it, dict))
            pct = (earned_pts / total_pts * 100) if total_pts > 0 else 0

            st.success(f"### 🎉 Kết quả: {earned_pts:.1f}/{total_pts:.1f} điểm ({pct:.0f}%)")

        with st.expander("📊 Xem kết quả JSON", expanded=True):
            st.json(result)

st.markdown("---")
st.caption("💡 Tip: Scroll lên xuống để sử dụng các tính năng khác nhau")
