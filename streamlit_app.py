"""EduRec Demo v2 - Main Streamlit App."""

import os

from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv(override=False)

# Page configuration
st.set_page_config(
    page_title="EduRec Demo v2",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("EduRec Demo v2 — Streamlit")
st.caption("Hệ thống đánh giá và đề xuất học tập thông minh cho Toán THCS")

# Sidebar settings
with st.sidebar:
    st.subheader("Cài đặt chung")
    default_api = os.environ.get("EDUREC_API_BASE", "http://localhost:8000")
    st.text_input(
        "API base URL",
        key="api_base_url",
        value=st.session_state.get("api_base_url", default_api),
        help="URL của FastAPI backend",
    )

    st.divider()

    st.markdown("""
    **Yêu cầu:**
    - FastAPI backend đang chạy
    - `GOOGLE_API_KEY` hoặc `GEMINI_API_KEY`
    - (Tùy chọn) DeepSeek OCR endpoint
    """)

    # Status check
    st.divider()
    st.subheader("Trạng thái")
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        st.success("Gemini API Key: Đã cấu hình")
    else:
        st.warning("Gemini API Key: Chưa cấu hình")

# Main content
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Trang tính năng")
    st.markdown("""
    1. **OCR & Anchors**
       - Trích xuất văn bản từ ảnh bài làm
       - Phân tích cấu trúc theo Bài/Câu

    2. **Đánh giá bài làm**
       - Tải ảnh đề, đáp án, bài nộp
       - Chấm điểm tự động với AI
    """)

with col2:
    st.subheader("🚀 Bắt đầu nhanh")
    st.markdown("""
    1. Chạy FastAPI backend:
    ```bash
    uvicorn app:app --port 8000
    ```

    2. Hoặc dùng **Chatbot độc lập**:
    ```bash
    streamlit run streamlit_chat.py
    ```
    """)

st.info("👈 Xem menu trái (Pages) để chuyển trang tính năng.")

