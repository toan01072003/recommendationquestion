"""EduRec Demo v2 - Unified Streamlit App (Giao diện hợp nhất)"""

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
    initial_sidebar_state="expanded",
)

# Title
st.title("EduRec Demo v2 — Hệ thống đánh giá thông minh")
st.caption("Tất cả tính năng trong một giao diện")


# -------------------- Helpers --------------------
@st.cache_resource
def get_model():
    """Cached Gemini model."""
    try:
        return _get_model_base()
    except RuntimeError as e:
        st.error(f"Lỗi khởi tạo Gemini: {e}")
        st.stop()


def is_likely_image_bytes(b: bytes) -> bool:
    """Kiểm tra bytes có phải là ảnh không."""
    if not b or len(b) < 4:
        return False
    return (
        b.startswith(b"\x89PNG\r\n\x1a\n")
        or b.startswith(b"\xff\xd8")
        or b.startswith(b"GIF8")
        or (b.startswith(b"RIFF") and len(b) > 12 and b[8:12] == b"WEBP")
    )


def parse_goal(goal_text: Optional[str]) -> Optional[float]:
    """Parse goal text to fraction."""
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
    """Derive gradebook from evaluation."""
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
            "parent_label": it.get("parent_label"),
            "skill_tag": it.get("skill_tag") or it.get("skillId"),
            "points": pts,
            "points_earned": pe,
            "rationale": it.get("rationale"),
            "rubric": it.get("rubric"),
        })

    return {"entries": entries, "totals": {"earned": earned, "total": total}}


def build_support_plan(weak: List[Dict[str, Any]], goal_frac: Optional[float], user_frac: Optional[float], total_q: int) -> List[Dict[str, Any]]:
    """Build support plan based on weak skills."""
    if total_q <= 0 or not weak:
        return []

    weights = [(w.get("skillId"), max(0.01, float(w.get("severity", 0.3)))) for w in weak if w.get("skillId")]
    if not weights:
        return []

    weights = sorted(weights, key=lambda x: -x[1])[:3]
    sw = sum(w for _, w in weights) or 1.0
    alloc = [(sid, max(1, round(total_q * w / sw))) for sid, w in weights]

    # Adjust to match total
    diff = total_q - sum(c for _, c in alloc)
    idx = 0
    while diff != 0 and alloc:
        sid, c = alloc[idx % len(alloc)]
        if diff > 0:
            alloc[idx % len(alloc)] = (sid, c + 1)
            diff -= 1
        else:
            if c > 1:
                alloc[idx % len(alloc)] = (sid, c - 1)
                diff += 1
        idx += 1

    # Difficulty mix
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
            if m <= max(e, h):
                m += 1
            elif e <= h:
                e += 1
            else:
                h += 1
            tot = e + m + h

        while tot > count:
            if m >= max(e, h) and m > 0:
                m -= 1
            elif e >= h and e > 0:
                e -= 1
            elif h > 0:
                h -= 1
            tot = e + m + h

        plan.append({"skillId": sid, "counts": {"easy": int(e), "medium": int(m), "hard": int(h)}})

    return plan


# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("⚙️ Cài đặt chung")

    api_base = st.text_input(
        "API base URL",
        value=os.environ.get("EDUREC_API_BASE", "http://localhost:8000"),
        help="URL của FastAPI backend",
    )

    lang = st.selectbox(
        "Ngôn ngữ",
        options=["vi", "en"],
        index=0,
        help="Ngôn ngữ chính",
    )

    st.divider()

    # Status check
    st.subheader("📊 Trạng thái")
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        st.success("✅ Gemini API Key")
    else:
        st.error("❌ Gemini API Key chưa cấu hình")

    st.divider()
    st.caption("💡 Tip: Sử dụng tabs bên trên để chuyển đổi giữa các tính năng")


# -------------------- Main Tabs --------------------
tab1, tab2, tab3 = st.tabs(["🤖 Chatbot AI", "📝 OCR & Anchors", "✅ Đánh giá bài làm"])


# ==================== TAB 1: CHATBOT ====================
with tab1:
    st.header("🤖 Chatbot - Chấm điểm & Gợi ý luyện tập")

    col1, col2 = st.columns([2, 1])

    with col1:
        exams = st.file_uploader(
            "Ảnh đề thi",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=True,
            key="chatbot_exam",
        )
        subs = st.file_uploader(
            "Ảnh bài làm",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=True,
            key="chatbot_sub",
        )

    with col2:
        goal_text = st.text_input("Mục tiêu điểm", value="8/10", help="VD: 8/10, 80%")
        user_text = st.text_input("Điểm hiện tại", value="", help="VD: 6/10, 60%")
        max_q = st.number_input("Số câu luyện", min_value=1, max_value=20, value=6)

    # Chat interface
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Chào bạn! Tải ảnh đề và bài làm, sau đó gửi tin nhắn để bắt đầu phân tích."}
        ]

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if user_input := st.chat_input("Nhập tin nhắn để bắt đầu phân tích..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        # Process
        with st.spinner("Đang phân tích..."):
            model = get_model()

            # Upload images
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
                response = "Vui lòng tải lên ảnh đề hoặc bài làm."
            else:
                try:
                    wait_until_files_active(exam_refs + sub_refs)
                except RuntimeError as e:
                    response = f"Lỗi: {e}"
                else:
                    # Simple evaluation
                    eval_prompt = {
                        "task": "evaluate_submission_items",
                        "instructions": [
                            "Parse exam, evaluate submission, return JSON with items array.",
                            "Each item has: label, question, skill_tag, student_answer, points, points_earned.",
                        ],
                        "output_schema": {
                            "type": "object",
                            "properties": {"items": {"type": "array"}},
                        },
                        "locale": lang,
                    }

                    parts = [json.dumps(eval_prompt)] + exam_refs + sub_refs
                    evresp = model.generate_content(parts)

                    try:
                        evaluation = json.loads(getattr(evresp, "text", "{}"))
                    except:
                        evaluation = {}

                    gradebook = derive_gradebook(evaluation)
                    earned = gradebook.get("totals", {}).get("earned", 0)
                    total = gradebook.get("totals", {}).get("total", 0)

                    response = f"✅ Đã phân tích xong! Kết quả: **{earned:.1f}/{total:.1f}** điểm"

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})


# ==================== TAB 2: OCR & ANCHORS ====================
with tab2:
    st.header("📝 OCR & Anchors")
    st.caption("Trích xuất văn bản và phân tích cấu trúc bài")

    uploaded_ocr = st.file_uploader(
        "Chọn ảnh bài nộp",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="ocr_upload",
    )

    if uploaded_ocr:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_ocr, caption="Preview", use_container_width=True)

        with col2:
            if st.button("🔍 Chạy OCR", type="primary"):
                with st.spinner("Đang trích xuất văn bản..."):
                    img_bytes = uploaded_ocr.read()
                    text = call_deepseek_ocr_api(api_base, uploaded_ocr.name, img_bytes, language=lang)

                if not text:
                    st.error("❌ OCR thất bại. Kiểm tra backend FastAPI.")
                else:
                    st.success(f"✅ Trích xuất thành công ({len(text)} ký tự)")

                    # Display OCR result
                    with st.expander("📄 Kết quả OCR", expanded=True):
                        st.text_area("Raw text", value=text, height=200, disabled=True)

                    # Build anchors
                    anchors = build_anchors_from_text(text)

                    if anchors:
                        st.subheader(f"🎯 Phát hiện {len(anchors)} anchor(s)")
                        for seg in anchors:
                            with st.expander(f"**{seg.get('anchor_id', '?')}**"):
                                st.text(seg.get("text") or "(trống)")
                    else:
                        st.info("Không phát hiện anchor (Bài/Câu)")


# ==================== TAB 3: EVALUATE ====================
with tab3:
    st.header("✅ Đánh giá bài làm với đáp án")
    st.caption("Tải ảnh đề, đáp án và bài nộp để chấm điểm tự động")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📋 Tài liệu tham khảo**")
        exam_eval = st.file_uploader("Ảnh đề", type=["png", "jpg", "jpeg", "webp", "gif"], key="eval_exam")
        key_eval = st.file_uploader("Ảnh đáp án", type=["png", "jpg", "jpeg", "webp", "gif"], key="eval_key")

    with col2:
        st.markdown("**📝 Bài làm học sinh**")
        sub_eval = st.file_uploader("Ảnh bài nộp (bắt buộc)", type=["png", "jpg", "jpeg", "webp", "gif"], key="eval_sub")

    # Preview images
    if exam_eval or key_eval or sub_eval:
        preview_cols = st.columns(3)
        if exam_eval:
            with preview_cols[0]:
                st.image(exam_eval, caption="Đề", use_container_width=True)
        if key_eval:
            with preview_cols[1]:
                st.image(key_eval, caption="Đáp án", use_container_width=True)
        if sub_eval:
            with preview_cols[2]:
                st.image(sub_eval, caption="Bài nộp", use_container_width=True)

    # Evaluate button
    if st.button("🚀 Đánh giá ngay", type="primary", disabled=not sub_eval):
        with st.spinner("Đang đánh giá bài làm..."):
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
            st.error("❌ Không nhận được kết quả. Kiểm tra server FastAPI.")
        else:
            # Display summary
            if isinstance(result, dict) and "items" in result:
                items = result.get("items", [])
                total_points = sum(float(it.get("points") or 1) for it in items if isinstance(it, dict))
                earned_points = sum(float(it.get("points_earned") or 0) for it in items if isinstance(it, dict))

                st.success(f"### 🎉 Kết quả: **{earned_points:.1f}/{total_points:.1f}** điểm ({earned_points/total_points*100:.0f}%)")

            # Display detailed result
            with st.expander("📊 Kết quả chi tiết (JSON)", expanded=True):
                st.json(result)
