# EduRec Demo v2 (Middle School Math, Toán THCS)

- Phân tích bài làm & đề, điểm số, mục tiêu của học sinh.
- Phân tách đề & đánh giá độ khó **5 mức** (trên từng bài và nhiều bài).
- Tìm điểm cần cải thiện & hướng dẫn.
- Gợi ý bài tập ôn luyện ZPD (0.6–0.8) + spaced.

## Run
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```
Docs: http://localhost:8080/docs

## Endpoints
- `GET /assessments/analyze-batch` — tổng hợp nhiều bài.
- `GET /assessments/score-chart/{examId}` — biểu đồ histogram điểm (PNG).
- `POST /student/profile` — mastery per-skill (nhiều bài) + gaps + guidance.
- `POST /agent/diagnose-hint` — agent heuristic (tiered hints).
- `POST /recommendations/playlist` — playlist ZPD dựa trên mastery tổng hợp.
# recommendationquestion

## Trạng Thái Hiện Tại (OCR + AnchorId)

- Hiện mới hoàn thiện phần OCR và kết nối bài làm theo chỉ mục anchor (anchorId).
- OCR chạy qua endpoint backend: `/ocr/deepseek-extract` (DeepSeek OCR). Ảnh bài nộp được trích xuất text, sau đó chia đoạn theo “anchor”.
- Quy tắc anchor (chuẩn hóa ASCII):
  - Big question: `B{n}` (ví dụ: “Bài 1”, “Câu 2”, “1.” → `B1`, `B2`)
  - Tiểu mục: `B{n}.{letter}` (ví dụ: “a)”, “b.” → `B1.a`, `B1.b`)
- Mapping hiện tại dựa trên `anchorId`: gắn đoạn OCR của bài nộp với từng mục/tiểu mục. Phần chấm bước/LLM sẽ phát triển tiếp.

### Cách Dùng Nhanh

- Mở trang `OCR & Anchors` (Streamlit) và tải ảnh bài nộp.
- Hệ thống gọi `/ocr/deepseek-extract`, hiển thị “Raw OCR”, và tự nhóm thành các “Anchors” (`B1`, `B1.a`, ...).
- Sử dụng `anchorId` để tham chiếu các đoạn bài tương ứng khi tích hợp với đề/đáp án.

### Giới Hạn Hiện Tại

- Chưa chấm điểm cuối cùng hoặc suy luận LLM đầy đủ; trọng tâm là pipeline OCR → phân đoạn → liên kết qua anchorId.
- Heuristic phát hiện anchor có thể sai với một số bố cục đề; sẽ tiếp tục tinh chỉnh.

## Streamlit (UI nhiều trang)

```bash
streamlit run streamlit_app.py
```

- Trang “OCR & Anchors”: dùng DeepSeek OCR qua endpoint `/ocr/deepseek-extract` để trích xuất và nhóm văn bản.
- Trang “Đánh giá bài làm”: gửi ảnh đề/đáp án/bài nộp tới `/assessments/evaluate-with-key` để chấm.

Các tiện ích UI được tách trong: `edurec_ui/` (services, utils) để dễ tái sử dụng.
