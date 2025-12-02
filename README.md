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

### Option 1: Pipeline Phân Tích (Recommended)
```bash
streamlit run streamlit_app.py
```
- ✅ Pipeline AI tự động: Phân đề → Chấm điểm → Phân tích → Gợi ý
- ✅ Giao diện chat trực quan với progress bar 8 bước
- ✅ Không cần FastAPI backend, chỉ cần GOOGLE_API_KEY

### Option 2: File Backup
```bash
streamlit run streamlit_chat.py
```
- ✅ Tương tự streamlit_app.py (backup file)
- ✅ Cùng pipeline phân tích

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
├── streamlit_app.py          # ⭐ Pipeline phân tích chính
├── streamlit_chat.py         # Backup file (tương tự app)
├── edurec_ui/                # Reusable UI components
│   ├── services/
│   │   ├── gemini.py         # Gemini AI integration
│   │   └── backend.py        # FastAPI client
│   └── utils/
│       └── anchors.py        # Anchor detection (B1, B1.a, ...)
├── app.py                    # FastAPI backend (optional)
└── requirements.txt
```

## Tính Năng Chính

### 1. 🎯 Pipeline Phân Tích Bài Làm (8 bước)
1. **Upload ảnh** - Tải đề thi và bài làm lên Gemini
2. **Đợi xử lý** - Gemini vision xử lý ảnh
3. **OCR** - DeepSeek trích xuất văn bản (optional)
4. **Phân đề** - Tách thành B1.a, B1.b với anchor detection
5. **Chấm điểm** - AI đánh giá từng câu với JSON Schema
6. **Phân tích điểm yếu** - Xác định kỹ năng cần cải thiện
7. **Tạo bài luyện** - Generate questions theo ZPD (0.6-0.8)
8. **Gợi ý Socratic** - Hints cho câu sai

**Kết quả:**
- Bảng điểm chi tiết với rationale
- Câu hỏi luyện tập cá nhân hóa
- Hints theo từng mục sai

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
