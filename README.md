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
