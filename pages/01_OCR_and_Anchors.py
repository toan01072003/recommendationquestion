import os
from typing import Optional

import streamlit as st

from edurec_ui.services.backend import call_deepseek_ocr_api
from edurec_ui.utils.anchors import build_anchors_from_text


def _get_api_base() -> str:
    return st.session_state.get("api_base_url") or os.environ.get("EDUREC_API_BASE", "http://localhost:8000")


st.title("OCR & Anchors")
st.caption("Trích xuất văn bản bằng DeepSeek OCR (qua FastAPI) và nhóm anchor đoạn bài.")

with st.sidebar:
    st.subheader("Cài đặt")
    st.text_input("API base URL", key="api_base_url", value=_get_api_base())
    lang = st.selectbox("Ngôn ngữ", ["vi", "en"], index=0, key="ocr_lang")

uploaded = st.file_uploader("Ảnh bài nộp (PNG/JPG/WebP)", type=["png", "jpg", "jpeg", "webp", "gif"])

if st.button("Chạy DeepSeek OCR"):
    if not uploaded:
        st.warning("Vui lòng tải ảnh.")
        st.stop()
    img_bytes = uploaded.read()
    text: Optional[str] = call_deepseek_ocr_api(_get_api_base(), uploaded.name, img_bytes, language=lang)
    if not text:
        st.error("OCR không khả dụng hoặc thất bại. Kiểm tra server FastAPI và DEEPSEEK_API_KEY.")
        st.stop()
    st.subheader("Kết quả OCR")
    st.text_area("Raw OCR", value=text, height=240)

    st.subheader("Anchors")
    anchors = build_anchors_from_text(text)
    if not anchors:
        st.info("Không phát hiện anchor.")
    else:
        for seg in anchors:
            with st.expander(f"{seg['anchor_id']}"):
                st.write(seg.get("text") or "")
