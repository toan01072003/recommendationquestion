import os
from dotenv import load_dotenv
import streamlit as st


load_dotenv(override=False)

st.set_page_config(page_title="EduRec Demo v2", page_icon="📚", layout="wide")

st.title("EduRec Demo v2 — Streamlit")
st.caption("Giao diện nhiều trang. Các chức năng chính được tách module để dễ bảo trì.")

with st.sidebar:
    st.subheader("Cài đặt chung")
    default_api = os.environ.get("EDUREC_API_BASE", "http://localhost:8000")
    st.text_input("API base URL", key="api_base_url", value=st.session_state.get("api_base_url", default_api))
    st.markdown("""
    - Đảm bảo backend FastAPI đang chạy (uvicorn app:app --port 8000).
    - Cấu hình khóa Gemini: `GOOGLE_API_KEY` hoặc `GEMINI_API_KEY`.
    - DeepSeek OCR dùng endpoint backend `/ocr/deepseek-extract`.
    """)

st.markdown("""
### Trang tính năng
- OCR & Anchors: Trích xuất OCR và nhóm đoạn bài theo anchor.
- Đánh giá bài làm: Gửi ảnh đề/đáp án/bài nộp để backend chấm điểm.

Xem menu trái (Pages) để chuyển trang.
""")

