import os
import io
import json
import requests
import streamlit as st


API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8080")

st.set_page_config(page_title="EduRec – Luyện đề bằng AI", page_icon="🧠", layout="centered")
st.title("🧠 EduRec – Gợi ý bài luyện từ ảnh đề + bài làm")
st.caption("Nhập ảnh đề thi, ảnh bài làm, mục tiêu điểm; AI phân tích lỗi và sinh bài luyện phù hợp")

with st.sidebar:
    st.subheader("Cấu hình")
    api_url = st.text_input("API base URL", API_BASE_URL, help="Địa chỉ FastAPI (ví dụ http://localhost:8080)")
    lang = st.selectbox("Ngôn ngữ", options=["vi", "en"], index=0)
    max_q = st.number_input("Số câu gợi ý", min_value=1, max_value=20, value=6)
    st.markdown("""
    Lưu ý: Streamlit chỉ là UI. Hãy chạy server FastAPI song song:
    - `uvicorn app:app --reload --host 0.0.0.0 --port 8080`
    - Đặt `GOOGLE_API_KEY` trong `.env` hoặc biến môi trường trên máy chạy FastAPI.
    """)

exam = st.file_uploader("Ảnh đề thi", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=False)
subm = st.file_uploader("Ảnh bài làm", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=False)

col1, col2 = st.columns(2)
with col1:
    goal_text = st.text_input("Mục tiêu điểm (vd 8/10 hoặc 80%)", value="")
with col2:
    user_text = st.text_input("Điểm của bạn (vd 6/10 hoặc 60%)", value="")

run = st.button("Phân tích & Gợi ý luyện tập", type="primary")

def _files_payload():
    files = {}
    if exam is not None:
        files["exam_image"] = (exam.name, exam.getvalue(), exam.type or "application/octet-stream")
    if subm is not None:
        files["submission_image"] = (subm.name, subm.getvalue(), subm.type or "application/octet-stream")
    return files

def _post_suggest():
    url = api_url.rstrip("/") + "/agent/suggest-questions"
    data = {
        "goal_score_text": goal_text,
        "user_score_text": user_text,
        "max_questions": str(max_q),
        "language": lang,
        "source": "llm",
    }
    resp = requests.post(url, data=data, files=_files_payload(), timeout=120)
    resp.raise_for_status()
    return resp.json()

def _chip(text, color="#e3e3e3"):
    return f"<span style='background:{color};padding:2px 6px;border-radius:8px;font-size:12px'>{text}</span>"

if run:
    if not api_url:
        st.error("Chưa cấu hình API base URL ở sidebar.")
        st.stop()
    try:
        with st.spinner("Đang phân tích ảnh và sinh câu hỏi..."):
            res = _post_suggest()
    except requests.RequestException as e:
        st.error(f"Lỗi gọi API: {e}")
        st.stop()

    st.subheader("Tóm tắt")
    col_a, col_b, col_c = st.columns(3)
    goal_frac = res.get("goal_fraction")
    user_frac = res.get("user_score_fraction")
    extracted = res.get("extracted_submission_score") or {}
    with col_a:
        st.metric("Mục tiêu", f"{round(goal_frac*100):d}%" if isinstance(goal_frac, (int,float)) else "–")
    with col_b:
        st.metric("Điểm của bạn", f"{round(user_frac*100):d}%" if isinstance(user_frac, (int,float)) else "–")
    with col_c:
        st.metric("Điểm đọc từ ảnh", extracted.get("text") or extracted.get("extracted_score_text") or "–")
    if msg := res.get("assistant_message"):
        st.write(msg)

    st.subheader("Đánh giá bài làm (theo câu)")
    ev = res.get("evaluation") or {}
    items = ev.get("items") or []
    if items:
        rows = []
        for it in items:
            rows.append({
                "Mục": it.get("label"),
                "Câu hỏi": it.get("question"),
                "Kỹ năng": it.get("skill_tag") or it.get("skillId"),
                "Đúng?": True if it.get("is_marked_correct") or it.get("llm_judgement_correct") else False,
                "Lỗi": it.get("error_type"),
                "Bước sai": it.get("error_step_index"),
                "Bài làm": it.get("student_answer"),
                "Điểm": it.get("points_earned"),
                "Điểm tối đa": it.get("points"),
                "Giải thích": it.get("rationale"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa trích xuất được mục nào từ ảnh (kiểm tra ảnh rõ nét, đầy đủ).")

    if res.get("error_summary"):
        st.subheader("Tổng hợp lỗi thủ tục")
        st.table(res["error_summary"])

    st.subheader("Câu hỏi LLM sinh thêm")
    gen = res.get("llm_generated_questions") or []
    if gen:
        rows = []
        for i, q in enumerate(gen, 1):
            rows.append({
                "#": i,
                "Skill": q.get("skillId"),
                "Độ khó": q.get("difficulty_tag"),
                "Câu hỏi": q.get("question"),
                "Đáp án": q.get("answer"),
                "Gợi ý lời giải": q.get("solution_outline"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có câu hỏi sinh thêm. Hãy thử ảnh rõ nét hơn hoặc tăng số câu.")

    with st.expander("JSON đầy đủ"):
        st.code(json.dumps(res, ensure_ascii=False, indent=2))

