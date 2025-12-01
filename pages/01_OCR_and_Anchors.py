"""OCR & Anchors page - Trích xuất văn bản và phân tích cấu trúc bài."""

import os
from typing import Optional

import streamlit as st

from edurec_ui.services.backend import call_deepseek_ocr_api
from edurec_ui.utils.anchors import build_anchors_from_text


def _get_api_base() -> str:
    """Lấy API base URL từ session state hoặc env."""
    return st.session_state.get("api_base_url") or os.environ.get("EDUREC_API_BASE", "http://localhost:8000")


# Page config
st.title("OCR & Anchors")
st.caption("Trích xuất văn bản bằng DeepSeek OCR (qua FastAPI) và nhóm anchor đoạn bài.")

# Sidebar settings
with st.sidebar:
    st.subheader("Cài đặt")
    api_base = st.text_input(
        "API base URL",
        key="api_base_url",
        value=_get_api_base(),
        help="URL của FastAPI backend (vd: http://localhost:8000)",
    )
    lang = st.selectbox(
        "Ngôn ngữ",
        options=["vi", "en"],
        index=0,
        key="ocr_lang",
        help="Ngôn ngữ chính trong tài liệu",
    )

# File uploader
uploaded = st.file_uploader(
    "Ảnh bài nộp",
    type=["png", "jpg", "jpeg", "webp", "gif"],
    help="Hỗ trợ PNG, JPG, WebP, GIF",
)

# Preview uploaded image
if uploaded:
    with st.expander("Xem trước ảnh", expanded=False):
        st.image(uploaded, use_container_width=True)

# OCR button
if st.button("Chạy DeepSeek OCR", type="primary", disabled=not uploaded):
    if not uploaded:
        st.warning("Vui lòng tải ảnh.")
        st.stop()

    with st.spinner("Đang trích xuất văn bản..."):
        img_bytes = uploaded.read()
        text: Optional[str] = call_deepseek_ocr_api(
            _get_api_base(), uploaded.name, img_bytes, language=lang
        )

    if not text:
        st.error("OCR không khả dụng hoặc thất bại. Kiểm tra server FastAPI và DEEPSEEK_API_KEY.")
        st.stop()

    # Display OCR result
    st.subheader("Kết quả OCR")
    st.text_area("Raw OCR", value=text, height=240, disabled=True)

    # Build and display anchors
    st.subheader("Anchors (Cấu trúc bài)")
    anchors = build_anchors_from_text(text)

    if not anchors:
        st.info("Không phát hiện anchor (Bài/Câu).")
    else:
        st.success(f"Phát hiện {len(anchors)} anchor(s)")
        for seg in anchors:
            anchor_id = seg.get("anchor_id", "?")
            anchor_text = seg.get("text") or "(trống)"
            with st.expander(f"**{anchor_id}**"):
                st.text(anchor_text)
