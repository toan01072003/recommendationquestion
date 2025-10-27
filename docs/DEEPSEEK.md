## DeepSeek OCR integration

Bạn có thể chạy DeepSeek‑OCR hoàn toàn nội bộ (không cần API) bằng mô hình HuggingFace `deepseek-ai/DeepSeek-OCR` hoặc dùng API (tùy chọn). Repo này mặc định ưu tiên LOCAL.

### Cách 1: LOCAL (không cần API)

1) Cài đặt phụ thuộc (GPU khuyến nghị):

```bash
pip install torch==2.6.0 transformers==4.46.3 tokenizers==0.20.3 einops addict easydict
# Tùy chọn (tăng tốc CUDA):
pip install flash-attn==2.7.3 --no-build-isolation
```

2) Không cần cấu hình thêm. Module `deepseek_ocr.py` sẽ tự tải mô hình từ HuggingFace lần đầu tiên chạy.

Biến môi trường (tùy chọn):

```powershell
$env:DEEPSEEK_OCR_MODEL = "deepseek-ai/DeepSeek-OCR"
$env:DS_ATTN_IMPL = "flash_attention_2"   # đặt khi đã cài flash-attn
$env:DEEPSEEK_OCR_BASE_SIZE = "1024"      # 512/640/1024/1280
$env:DEEPSEEK_OCR_IMAGE_SIZE = "640"
$env:DEEPSEEK_OCR_CROP = "true"
```

### Cách 2: API (tùy chọn)

Nếu bạn muốn dùng API DeepSeek, hãy đặt:

```powershell
$env:DEEPSEEK_API_KEY = "<your_api_key>"
$env:DEEPSEEK_API_BASE = "https://api.deepseek.com"
$env:DEEPSEEK_VISION_MODEL = "deepseek-chat"
```

Module sẽ ưu tiên LOCAL; nếu LOCAL không có sẵn, có thể chuyển qua API (hoặc trả về None tùy cấu hình mã nguồn).

### Sử dụng trong server

- `POST /assessments/grade-from-images` sẽ tự gọi OCR nội bộ trên ảnh bài làm và đưa kết quả vào `ocr_hint` để hỗ trợ ghép mục/chấm điểm.
- `POST /ocr/deepseek-extract` cho phép test OCR trực tiếp, trả `{lines:[], char_count}`.

