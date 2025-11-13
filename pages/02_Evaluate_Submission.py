import os
from typing import Optional

import streamlit as st

from edurec_ui.services.backend import call_evaluate_with_key_api


def _get_api_base() -> str:
    return st.session_state.get("api_base_url") or os.environ.get("EDUREC_API_BASE", "http://localhost:8000")


st.title("Đánh giá bài làm với đáp án")
st.caption("Gửi ảnh đề, đáp án (tùy chọn) và bài nộp lên FastAPI để đánh giá.")

with st.sidebar:
    st.subheader("Cài đặt")
    st.text_input("API base URL", key="api_base_url", value=_get_api_base())
    lang = st.selectbox("Ngôn ngữ", ["vi", "en"], index=0, key="eval_lang")

col1, col2 = st.columns(2)
with col1:
    exam = st.file_uploader("Ảnh đề (tuỳ chọn)", type=["png", "jpg", "jpeg", "webp", "gif"], key="exam_img")
    key = st.file_uploader("Ảnh đáp án (tuỳ chọn)", type=["png", "jpg", "jpeg", "webp", "gif"], key="key_img")
with col2:
    sub = st.file_uploader("Ảnh bài nộp (bắt buộc)", type=["png", "jpg", "jpeg", "webp", "gif"], key="sub_img")

if st.button("Đánh giá"):
    if not sub:
        st.warning("Cần tải lên ảnh bài nộp.")
        st.stop()
    exam_b = exam.read() if exam else None
    key_b = key.read() if key else None
    sub_b = sub.read()
    result: Optional[dict] = call_evaluate_with_key_api(
        _get_api_base(),
        exam_b, exam.name if exam else None,
        key_b, key.name if key else None,
        sub_b, sub.name if sub else None,
        language=lang,
    )
    if not result:
        st.error("Không nhận được kết quả. Kiểm tra server FastAPI.")
        st.stop()

    st.subheader("Kết quả JSON")
    st.json(result)
