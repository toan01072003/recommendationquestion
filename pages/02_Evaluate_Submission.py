"""Evaluate Submission page - Đánh giá bài làm với đề và đáp án."""

import os
from typing import Optional

import streamlit as st

from edurec_ui.services.backend import call_evaluate_with_key_api


def _get_api_base() -> str:
    """Lấy API base URL từ session state hoặc env."""
    return st.session_state.get("api_base_url") or os.environ.get("EDUREC_API_BASE", "http://localhost:8000")


# Page config
st.title("Đánh giá bài làm với đáp án")
st.caption("Gửi ảnh đề, đáp án (tùy chọn) và bài nộp lên FastAPI để đánh giá.")

# Sidebar settings
with st.sidebar:
    st.subheader("Cài đặt")
    st.text_input(
        "API base URL",
        key="api_base_url",
        value=_get_api_base(),
        help="URL của FastAPI backend",
    )
    lang = st.selectbox(
        "Ngôn ngữ",
        options=["vi", "en"],
        index=0,
        key="eval_lang",
        help="Ngôn ngữ chính trong tài liệu",
    )

# File uploaders in columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tài liệu tham khảo**")
    exam = st.file_uploader(
        "Ảnh đề (tuỳ chọn)",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="exam_img",
        help="Ảnh đề bài kiểm tra",
    )
    key = st.file_uploader(
        "Ảnh đáp án (tuỳ chọn)",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="key_img",
        help="Ảnh đáp án mẫu",
    )

with col2:
    st.markdown("**Bài làm học sinh**")
    sub = st.file_uploader(
        "Ảnh bài nộp (bắt buộc)",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        key="sub_img",
        help="Ảnh bài làm của học sinh",
    )

# Preview uploaded images
preview_cols = st.columns(3)
if exam:
    with preview_cols[0]:
        with st.expander("Xem đề", expanded=False):
            st.image(exam, use_container_width=True)
if key:
    with preview_cols[1]:
        with st.expander("Xem đáp án", expanded=False):
            st.image(key, use_container_width=True)
if sub:
    with preview_cols[2]:
        with st.expander("Xem bài nộp", expanded=False):
            st.image(sub, use_container_width=True)

# Evaluate button
if st.button("Đánh giá", type="primary", disabled=not sub):
    if not sub:
        st.warning("Cần tải lên ảnh bài nộp.")
        st.stop()

    with st.spinner("Đang đánh giá bài làm..."):
        exam_b = exam.read() if exam else None
        key_b = key.read() if key else None
        sub_b = sub.read()

        result: Optional[dict] = call_evaluate_with_key_api(
            _get_api_base(),
            exam_b,
            exam.name if exam else None,
            key_b,
            key.name if key else None,
            sub_b,
            sub.name if sub else None,
            language=lang,
        )

    if not result:
        st.error("Không nhận được kết quả. Kiểm tra server FastAPI.")
        st.stop()

    # Display summary if available
    if isinstance(result, dict) and "items" in result:
        items = result.get("items", [])
        total_points = sum(float(it.get("points") or 1) for it in items if isinstance(it, dict))
        earned_points = sum(float(it.get("points_earned") or 0) for it in items if isinstance(it, dict))
        st.success(f"Kết quả: **{earned_points:.1f}/{total_points:.1f}** điểm")

    # Display JSON result
    st.subheader("Kết quả chi tiết")
    st.json(result)
