## Đánh giá bằng Gemini

Yêu cầu: thiết lập biến môi trường `GOOGLE_API_KEY` (hoặc `GEMINI_API_KEY`).

Ví dụ:

```powershell
# Windows PowerShell
$env:GOOGLE_API_KEY = ”AIzaSyCrYO5BTCECd0VT6JcTmtopqo2_CjVyMxQ”
```

Sau đó chạy server:

```powershell
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Các endpoint mới:

- POST `/assessments/extract-score`
  - Form-data: `submission_image` (file)
  - Kết quả: JSON với điểm đọc được từ ảnh bài làm và độ tự tin.

- POST `/assessments/grade-from-images`
  - Form-data: `exam_image` (file), `submission_image` (file)
  - Kết quả: JSON gồm điểm đọc được trên bài làm, danh sách câu hỏi/mục (khi bóc tách được), nhận định đúng/sai (dựa vào dấu tick/cross hoặc nội dung), và tổng hợp.

- POST `/agent/generate-questions-by-levels`
  - Form-data:
    - `base_question` (string, tùy chọn): câu gốc để sinh biến thể
    - `levels` (string, tùy chọn): đặc tả mức độ, ví dụ `easy:2,medium:2,hard:2` (mặc định)
    - `exam_image`/`submission_image` (file, tùy chọn): nếu gửi, agent sẽ suy luận kỹ năng từ ảnh rồi sinh câu hỏi theo cấp độ
    - `skills_hint` (string, tùy chọn): danh sách skillId gợi ý, phân tách bởi dấu phẩy
    - `language` (string, mặc định `vi`)
  - Kết quả: JSON gồm `questions` mỗi phần tử có `level`, `skillId` (nếu map được), `question`, `answer`, `solution_outline`.

Thử nhanh (PowerShell):

```powershell
curl -X POST "http://localhost:8080/assessments/extract-score" `
  -F "submission_image=@C:\path\to\submission.jpg"

curl -X POST "http://localhost:8080/assessments/grade-from-images" `
  -F "exam_image=@C:\path\to\exam.jpg" `
  -F "submission_image=@C:\path\to\submission.jpg"
```

Ghi chú:

- Model mặc định: `gemini-1.5-flash` (có thể đổi bằng env `GEMINI_MODEL`).
- Nếu thiếu SDK hoặc API key, endpoint trả HTTP 500 kèm thông báo lỗi.
