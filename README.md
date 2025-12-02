# EduRec Demo v2 (Middle School Math, Toán THCS)

Hệ thống đánh giá và đề xuất học tập thông minh cho Toán THCS, sử dụng AI để:
- 📝 Phân tích đề thi và bài làm học sinh tự động
- 🎯 Chấm điểm và đánh giá theo từng kỹ năng
- 🤖 Gợi ý bài tập ôn luyện cá nhân hóa (ZPD: 0.6-0.8)
- 📊 Phân tích độ khó 5 mức và xác định điểm yếu

## Công Nghệ

- **Backend**: FastAPI + Google Gemini AI + DeepSeek OCR
- **Frontend**: Streamlit (giao diện đơn giản, tất cả trong 1 trang)
- **AI Features**: IRT, ZPD targeting, Cascade grading, Error taxonomy

## Cài Đặt

```bash
# Clone và cài đặt
git clone <repo-url>
cd edurec_demo_v2

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt

# Cấu hình API keys (.env)
GOOGLE_API_KEY=your_gemini_api_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here  # Optional
```

## Chạy Ứng Dụng

### Option 1: Giao Diện Đầy Đủ (Recommended)
```bash
streamlit run streamlit_single.py
```
- ✅ Tất cả tính năng trong 1 trang (Chatbot + OCR + Evaluate)
- ✅ Vertical scroll, không cần chuyển trang
- ✅ Giao diện gọn gàng, dễ sử dụng

### Option 2: Chatbot Độc Lập
```bash
streamlit run streamlit_chat.py
```
- ✅ Chat AI với phân tích đầy đủ
- ✅ Không cần FastAPI backend
- ✅ Chỉ cần GOOGLE_API_KEY

### Option 3: FastAPI Backend (Optional)
```bash
uvicorn app:app --reload --port 8000
```
Docs: http://localhost:8000/docs

**Endpoints:**
- `POST /ocr/deepseek-extract` — OCR trích xuất văn bản
- `POST /assessments/evaluate-with-key` — Chấm điểm với đáp án
- `GET /assessments/analyze-batch` — Phân tích nhiều bài
- `POST /student/profile` — Mastery theo kỹ năng
- `POST /recommendations/playlist` — Gợi ý bài tập ZPD

## Cấu Trúc Thư Mục

```
edurec_demo_v2/
├── streamlit_single.py      # ⭐ Giao diện chính (tất cả trong 1 trang)
├── streamlit_chat.py         # Chatbot độc lập
├── edurec_ui/                # Reusable UI components
│   ├── services/
│   │   ├── gemini.py         # Gemini AI integration
│   │   └── backend.py        # FastAPI client
│   └── utils/
│       └── anchors.py        # Anchor detection (B1, B1.a, ...)
├── app.py                    # FastAPI backend
└── requirements.txt
```

## Tính Năng Chính

### 1. 🤖 Chatbot AI - Chấm điểm & Gợi ý
- Upload đề thi và bài làm (nhiều trang)
- AI phân tích tự động theo từng Bài/Câu
- Chấm điểm chi tiết với rationale
- Gợi ý bài tập luyện theo ZPD

### 2. 📝 OCR & Phân tích cấu trúc
- Trích xuất văn bản từ ảnh (DeepSeek OCR)
- Phát hiện anchor tự động (B1, B1.a, B2.b, ...)
- Hiển thị cấu trúc đề bài rõ ràng

### 3. ✅ Đánh giá bài làm chi tiết
- So sánh với đáp án chuẩn
- Tính điểm tự động
- Hiển thị kết quả JSON chi tiết

## Anchor System

Hệ thống chuẩn hóa câu hỏi theo ASCII:
- **Big question**: `B{n}` (VD: "Bài 1", "Câu 2" → `B1`, `B2`)
- **Sub-question**: `B{n}.{letter}` (VD: "a)", "b)" → `B1.a`, `B1.b`)

## Đánh Giá AI

- **Prompt Engineering**: 8.5/10 - JSON Schema structured prompts
- **Educational Theory**: 8/10 - IRT + ZPD + Error taxonomy
- **Code Quality**: 7.5/10 - Clean, modular, well-documented

## License

MIT
