
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json, os, math, io
import tempfile
import mimetypes
try:
    # Load environment from .env at repo root
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
except Exception:
    pass
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EXAMS_DIR = os.path.join(DATA_DIR, "exams")

app = FastAPI(title="EduRec Demo v2")

class AgentReq(BaseModel):
    grade: int = Field(..., ge=6, le=9)
    problem: str
    student_steps: List[str]
    skill_ids: List[str]
    current_hint_level: int = Field(1, ge=1, le=4)
    wrong_attempts_on_this_step: int = 0
    locale: str = "vi"
    max_hint_level: int = 4

class PlaylistReq(BaseModel):
    studentId: str
    maxItems: int = 6
    focusSkills: Optional[List[str]] = None
    timeBudgetSec: Optional[int] = None
    allowExploration: bool = False

class ProfileReq(BaseModel):
    studentId: str
    topKSkills: int = 3

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_items():
    return load_json(os.path.join(DATA_DIR, "items.json"))

def load_goals():
    return load_json(os.path.join(DATA_DIR, "student_goals.json"))

def load_exams():
    exams = []
    for fname in sorted(os.listdir(EXAMS_DIR)):
        if fname.endswith(".json"):
            exams.append(load_json(os.path.join(EXAMS_DIR, fname)))
    return exams

def sigmoid(x): return 1/(1+math.exp(-x))

def bin5_by_pcorrect(p):
    if p >= 0.85: return "L1"
    if p >= 0.70: return "L2"
    if p >= 0.50: return "L3"
    if p >= 0.30: return "L4"
    return "L5"

def compute_pbis(item_obs, totals_without):
    X = np.array(item_obs, dtype=float)
    Y = np.array(totals_without, dtype=float)
    if len(X) < 3 or np.std(X) == 0 or np.std(Y) == 0:
        return 0.0
    return float(np.corrcoef(X, Y)[0,1])

# -------- Gemini setup --------
_GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

def _get_gemini_model():
    """Return configured Gemini model or raise HTTPException if no API key."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GOOGLE_API_KEY env var for Gemini.")
    try:
        import google.generativeai as genai
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini SDK not installed: {e}")
    genai.configure(api_key=api_key)
    # Use JSON responses by default for structured extraction
    generation_config = {"temperature": 0.2, "response_mime_type": "application/json"}
    return genai.GenerativeModel(model_name=_GEMINI_MODEL_NAME, generation_config=generation_config)

def _upload_to_gemini(tmp_path: str, display_name: str):
    import google.generativeai as genai
    return genai.upload_file(path=tmp_path, display_name=display_name)

def _unique_skills_from_bank():
    # Deprecated for LLM-only flow; kept for backwards compatibility.
    return []

def _parse_goal(goal_score: Optional[float], goal_text: Optional[str]):
    """Return goal as fraction 0..1 if possible, else None.
    Supports numeric (<=1 treated as fraction; >1 and <=100 treated as percentage),
    and text forms like "8/10", "80%".
    """
    if goal_score is not None:
        try:
            g = float(goal_score)
            if 0 <= g <= 1:
                return g
            if 1 < g <= 100:
                return g/100.0
        except Exception:
            pass
    if goal_text:
        t = goal_text.strip().replace(" ", "")
        if "/" in t:
            try:
                a,b = t.split("/",1)
                a = float(a)
                b = float(b)
                if b != 0:
                    return max(0.0, min(1.0, a/b))
            except Exception:
                pass
        if t.endswith("%"):
            try:
                return max(0.0, min(1.0, float(t[:-1])/100.0))
            except Exception:
                pass
        try:
            g = float(t)
            if 0 <= g <= 1:
                return g
            if 1 < g <= 100:
                return g/100.0
        except Exception:
            pass
    return None

def _is_likely_image_bytes(b: bytes) -> bool:
    if not b or len(b) < 4:
        return False
    # PNG
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    # JPEG
    if b.startswith(b"\xff\xd8"):
        return True
    # GIF
    if b.startswith(b"GIF8"):
        return True
    # WebP
    if b.startswith(b"RIFF") and b[8:12] == b"WEBP":
        return True
    # HEIC/HEIF signatures vary; skip strict check
    return False

async def _maybe_upload_image(file: Optional[UploadFile], default_name: str) -> Optional[Any]:
    """Read an optional UploadFile and upload to Gemini if non-empty and looks like an image.
    Returns a Gemini file reference or None.
    """
    if file is None:
        return None
    try:
        content = await file.read()
    except Exception:
        return None
    if not content or len(content) == 0:
        return None
    if not _is_likely_image_bytes(content):
        # Avoid sending invalid files to Gemini
        return None
    suffix = os.path.splitext(getattr(file, 'filename', '') or default_name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(content)
        tmp_path = tf.name
    try:
        return _upload_to_gemini(tmp_path, display_name=os.path.basename(getattr(file, 'filename', '') or default_name))
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _normalize_level_tag(tag: str) -> str:
    t = (tag or "").strip().lower()
    mapping = {
        "easy": "easy", "de": "easy", "dễ": "easy", "l1": "easy", "l2": "easy",
        "medium": "medium", "trungbinh": "medium", "tb": "medium", "l3": "medium",
        "hard": "hard", "kho": "hard", "khó": "hard", "l4": "hard", "l5": "hard",
    }
    return mapping.get(t, t or "medium")

def _parse_levels_spec(spec: Optional[str]):
    """Parse levels spec like 'easy:2,medium:3,hard:1' into list of (level, count)."""
    if not spec:
        return [("easy", 2), ("medium", 2), ("hard", 2)]
    out = []
    for part in str(spec).split(','):
        if not part.strip():
            continue
        if ':' in part:
            k, v = part.split(':', 1)
            try:
                c = int(v)
            except Exception:
                c = 1
            out.append((_normalize_level_tag(k), max(0, c)))
        else:
            out.append((_normalize_level_tag(part), 1))
    # Merge same levels
    agg = {}
    for lvl, cnt in out:
        agg[lvl] = agg.get(lvl, 0) + cnt
    return [(k, v) for k, v in agg.items() if v > 0]

def _build_support_plan(weak: List[Dict[str, Any]], goal_frac: Optional[float], user_frac: Optional[float], total_q: int) -> List[Dict[str, Any]]:
    """
    Turn weak skills + goal/current gap into a generation plan per skill with counts per difficulty.
    Returns a list of {skillId, counts: {easy, medium, hard}}.
    """
    if total_q <= 0 or not weak:
        return []
    # Normalize severities to weights
    weights = []
    for w in weak:
        sid = w.get("skillId")
        try:
            sev = float(w.get("severity", 0.3))
        except Exception:
            sev = 0.3
        if not sid:
            continue
        weights.append((sid, max(0.01, sev)))
    if not weights:
        return []
    # Keep up to 3 skills for focus
    weights = sorted(weights, key=lambda x: -x[1])[:3]
    sw = sum(w for _, w in weights) or 1.0
    # Allocate counts per skill (round, then fix sum)
    alloc = [(sid, max(1, round(total_q * w / sw))) for sid, w in weights]
    diff = total_q - sum(c for _, c in alloc)
    # Adjust to match total
    idx = 0
    while diff != 0 and alloc:
        sid, c = alloc[idx % len(alloc)]
        if diff > 0:
            alloc[idx % len(alloc)] = (sid, c+1)
            diff -= 1
        else:
            if c > 1:
                alloc[idx % len(alloc)] = (sid, c-1)
                diff += 1
        idx += 1

    # Difficulty mix based on severity and gap
    delta = None
    if goal_frac is not None and user_frac is not None:
        delta = max(-1.0, min(1.0, float(goal_frac) - float(user_frac)))

    plan = []
    sev_map = {w.get("skillId"): float(w.get("severity", 0.3)) for w in weak if w.get("skillId")}
    for sid, count in alloc:
        sev = sev_map.get(sid, 0.3)
        # base mixes (favor exam-like medium/hard for better alignment)
        if sev >= 0.7:
            # serious weakness: still include scaffolding but emphasize exam-like medium/hard
            mix = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
        elif sev >= 0.4:
            mix = {"easy": 0.1, "medium": 0.5, "hard": 0.4}
        else:
            mix = {"easy": 0.05, "medium": 0.45, "hard": 0.5}
        # adjust by delta (aim higher => more hard)
        if delta is not None:
            if delta >= 0.2:
                mix["easy"] = max(0.0, mix["easy"] - 0.1)
                mix["hard"] = min(0.7, mix["hard"] + 0.1)
            elif delta <= -0.1:
                mix["easy"] = min(0.5, mix["easy"] + 0.1)
                mix["hard"] = max(0.1, mix["hard"] - 0.1)
        # convert to integer counts
        e = max(0, round(count * mix["easy"]))
        m = max(0, round(count * mix["medium"]))
        h = max(0, round(count * mix["hard"]))
        # fix rounding
        tot = e + m + h
        while tot < count:
            # add to the largest remainder bucket (prefer medium)
            if m <= max(e, h):
                m += 1
            elif e <= h:
                e += 1
            else:
                h += 1
            tot = e + m + h
        while tot > count:
            # subtract from the largest
            if m >= max(e, h) and m > 0:
                m -= 1
            elif e >= h and e > 0:
                e -= 1
            elif h > 0:
                h -= 1
            tot = e + m + h
        plan.append({"skillId": sid, "counts": {"easy": int(e), "medium": int(m), "hard": int(h)}})
    return plan

def _analyze_exam_style(model, exam_ref, language: str):
    """Ask LLM to summarize exam style to better align generated questions.
    Returns a dict with fields like: preferred_response_type (FR/MCQ), notation, typical_points,
    difficulty_examples, topic_distribution, solution_format.
    """
    try:
        prompt = {
            "task": "profile_exam_style",
            "instructions": [
                "Analyze the exam image and summarize style so new questions can match it.",
                "Report: preferred_response_type (FR or MCQ), notation highlights (fractions, algebraic forms), typical points per question, topic_distribution (rough categories), example_difficulty_markers (what makes a question easy/medium/hard), and solution_format notes.",
                "Return strict JSON only."
            ],
            "output_schema": {
                "type": "object",
                "properties": {
                    "preferred_response_type": {"type": ["string","null"]},
                    "notation": {"type": ["string","null"]},
                    "typical_points": {"type": ["number","null"]},
                    "topic_distribution": {"type": ["string","array","null"]},
                    "example_difficulty_markers": {"type": ["string","array","null"]},
                    "solution_format": {"type": ["string","null"]}
                },
                "additionalProperties": True
            },
            "locale": language
        }
        resp = model.generate_content([json.dumps(prompt), exam_ref])
        text = getattr(resp, "text", None) or "{}"
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}
    except Exception:
        return {}

@app.get("/assessments/analyze-batch")
def analyze_batch():
    exams = load_exams()
    out = []
    for exam in exams:
        students = sorted({r["studentId"] for r in exam["responses"]})
        totals = {sid: 0.0 for sid in students}
        for r in exam["responses"]:
            totals[r["studentId"]] += float(r.get("score", 0.0))
        total_vals = [totals[sid] for sid in students]
        scoreStats = {
            "mean": round(float(np.mean(total_vals)),2),
            "std": round(float(np.std(total_vals, ddof=0)),2),
            "max": round(float(np.max(total_vals)),2),
            "min": round(float(np.min(total_vals)),2),
            "histBins": np.histogram(total_vals, bins=5)[0].tolist()
        }

        item_ids = [it["itemId"] for it in exam["items"]]
        totals_without = {iid: {sid: totals[sid] for sid in students} for iid in item_ids}
        for iid in item_ids:
            for sid in students:
                sc = 0.0
                for r in exam["responses"]:
                    if r["studentId"]==sid and r["itemId"]==iid:
                        sc = float(r.get("score",0.0))
                        break
                totals_without[iid][sid] -= sc

        itemStats = []
        for it in exam["items"]:
            iid = it["itemId"]
            obs = [int(r["isCorrect"]) for r in exam["responses"] if r["itemId"]==iid]
            p = sum(obs)/len(obs) if obs else 0.0
            pbis = compute_pbis(obs, [totals_without[iid][sid] for sid in students])
            itemStats.append({
                "itemId": iid,
                "pCorrect": round(p,2),
                "pbis": round(pbis,2),
                "level": bin5_by_pcorrect(p),
                "skillIds": it.get("skillIds",[])
            })

        counts = {"L1":0,"L2":0,"L3":0,"L4":0,"L5":0}
        for st in itemStats:
            counts[st["level"]] += 1
        difficultyBins = [{"level":k,"count":counts[k]} for k in ["L1","L2","L3","L4","L5"]]

        out.append({
            "examId": exam["examId"],
            "scoreStats": scoreStats,
            "difficultyBins": difficultyBins,
            "itemStats": itemStats
        })
    return {"exams": out}

@app.get("/assessments/score-chart/{examId}")
def score_chart(examId: str):
    exams = load_exams()
    exam = next((e for e in exams if e["examId"]==examId), None)
    if not exam:
        raise HTTPException(404, f"examId {examId} not found")

    students = sorted({r["studentId"] for r in exam["responses"]})
    totals = {sid: 0.0 for sid in students}
    for r in exam["responses"]:
        totals[r["studentId"]] += float(r.get("score", 0.0))
    total_vals = [totals[sid] for sid in students]

    fig, ax = plt.subplots()
    ax.hist(total_vals, bins=5)
    ax.set_title(f"Score Histogram: {examId}")
    ax.set_xlabel("Total score")
    ax.set_ylabel("Count")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/assessments/extract-score")
async def extract_score(submission_image: UploadFile = File(...), language: str = Form("vi")):
    """
    Use Gemini to extract the printed/handwritten total score from a submission image.
    Returns JSON with best-guess score and supporting details.
    """
    model = _get_gemini_model()

    # Persist the upload to a temp file so we can use upload_file API reliably
    suffix = os.path.splitext(submission_image.filename or "uploaded")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        content = await submission_image.read()
        tf.write(content)
        tmp_path = tf.name

    try:
        file_ref = _upload_to_gemini(tmp_path, display_name=os.path.basename(submission_image.filename or "submission"))
        prompt = {
            "task": "extract_submission_score",
            "instructions": [
                "Read the score shown on the submission image.",
                "Score may be handwritten or typed; look for patterns like 'Score', 'Điểm', 'Marks', 'X/Y'.",
                "Return strictly valid JSON.",
            ],
            "output_schema": {
                "type": "object",
                "properties": {
                    "extracted_score_text": {"type": "string"},
                    "score_value": {"type": "number"},
                    "score_denominator": {"type": ["number", "null"]},
                    "confidence": {"type": "number"},
                    "notes": {"type": "string"}
                },
                "required": ["extracted_score_text", "confidence"],
                "additionalProperties": False
            },
            "locale": language
        }
        resp = model.generate_content([json.dumps(prompt), file_ref])
        text = getattr(resp, "text", None) or "{}"
        try:
            data = json.loads(text)
        except Exception:
            data = {"raw": text}
        return data
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/assessments/grade-from-images")
async def grade_from_images(
    exam_image: UploadFile = File(...),
    submission_image: UploadFile = File(...),
    language: str = Form("vi"),
):
    """
    Use Gemini to analyze an exam image and a student's submission image, extract the printed score,
    attempt item-level mapping (if layout/labels are clear), and provide a structured evaluation.
    """
    model = _get_gemini_model()

    # Save both images if valid and upload
    exam_bytes = await exam_image.read()
    sub_bytes = await submission_image.read()
    if not _is_likely_image_bytes(exam_bytes) or not _is_likely_image_bytes(sub_bytes):
        raise HTTPException(status_code=400, detail="Both exam_image and submission_image must be valid images.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(exam_image.filename or "exam.png")[1] or ".png") as ef:
        ef.write(exam_bytes)
        exam_tmp = ef.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(submission_image.filename or "submission.png")[1] or ".png") as sf:
        sf.write(sub_bytes)
        sub_tmp = sf.name

    try:
        exam_ref = _upload_to_gemini(exam_tmp, display_name=os.path.basename(exam_image.filename or "exam"))
        sub_ref = _upload_to_gemini(sub_tmp, display_name=os.path.basename(submission_image.filename or "submission"))

        prompt = {
            "task": "grade_submission_against_exam",
            "instructions": [
                "You are a strict, consistent grader.",
                "Analyze the exam image to understand question numbering, question text, and any visible answer key or points.",
                "Analyze the submission image to extract the student's answers and the printed score on the paper (if present).",
                "If visible, use checkmarks/crosses to infer correctness. If not, attempt to judge correctness from content with a brief rationale.",
                "Return strictly valid JSON using the schema below. Be concise in rationales.",
            ],
            "output_schema": {
                "type": "object",
                "properties": {
                    "extracted_submission_score": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "value": {"type": ["number", "null"]},
                            "denominator": {"type": ["number", "null"]},
                            "confidence": {"type": "number"}
                        },
                        "required": ["text", "confidence"],
                        "additionalProperties": False
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "question": {"type": ["string", "null"]},
                                "student_answer": {"type": ["string", "null"]},
                                "is_marked_correct": {"type": ["boolean", "null"]},
                                "llm_judgement_correct": {"type": ["boolean", "null"]},
                                "points": {"type": ["number", "null"]},
                                "points_earned": {"type": ["number", "null"]},
                                "rationale": {"type": ["string", "null"]}
                            },
                            "required": ["label"],
                            "additionalProperties": False
                        }
                    },
                    "totals": {
                        "type": "object",
                        "properties": {
                            "points_earned": {"type": ["number", "null"]},
                            "points_total": {"type": ["number", "null"]},
                            "computed_accuracy": {"type": ["number", "null"]}
                        },
                        "additionalProperties": False
                    },
                    "notes": {"type": "string"}
                },
                "required": ["extracted_submission_score", "items"],
                "additionalProperties": False
            },
            "locale": language
        }

        resp = model.generate_content([json.dumps(prompt), exam_ref, sub_ref])
        text = getattr(resp, "text", None) or "{}"
        try:
            data = json.loads(text)
        except Exception:
            data = {"raw": text}
        return data
    finally:
        for p in (exam_tmp, sub_tmp):
            try:
                os.remove(p)
            except Exception:
                pass

@app.get("/chatbot", response_class=HTMLResponse)
def chatbot_page():
    return """
<!doctype html>
<html><head><meta charset=\"utf-8\"><title>EduRec Chatbot</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:960px;margin:24px auto;padding:0 12px}
.card{border:1px solid #e3e3e3;border-radius:8px;padding:12px;margin:12px 0}
.row{display:flex;gap:12px;flex-wrap:wrap}
label{display:block;margin:6px 0}
table{border-collapse:collapse;width:100%}
th,td{border:1px solid #eee;padding:6px;text-align:left}
.ok{color:#0a7b34;font-weight:600}
.bad{color:#b00020;font-weight:600}
.muted{color:#666}
</style>
</head>
<body>
  <h2>EduRec Chatbot</h2>
  <form id=\"f\" class=\"card\" enctype=\"multipart/form-data\">
    <div class=\"row\">
      <div><label>Exam image<input type=\"file\" name=\"exam_image\"></label></div>
      <div><label>Submission image<input type=\"file\" name=\"submission_image\"></label></div>
    </div>
    <div class=\"row\">
      <div><label>Goal score (e.g. 8/10, 80%)<input name=\"goal_score_text\" placeholder=\"8/10\"></label></div>
      <div><label>Your score (optional, e.g. 6/10)<input name=\"user_score_text\" placeholder=\"6/10\"></label></div>
      <div><label>Max questions<input name=\"max_questions\" type=\"number\" value=\"6\"></label></div>
      <div><label>Language<input name=\\"language\\" value=\\"vi\\"></label><input type=\\"hidden\\" name=\\"source\\" value=\\"llm\\" /></div>
    </div>
    <button type=\"submit\">Suggest questions</button>
  </form>

  <div id=\"summary\" class=\"card\"></div>
  <div id=\"eval\" class=\"card\"></div>
  <div id=\"gen\" class=\"card\"></div>

  <script>
  function esc(s){return (s??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
  function rSummary(data){
    const g = data.goal_fraction!=null? (Math.round(data.goal_fraction*100)+'%') : '—';
    const u = data.user_score_fraction!=null? (Math.round(data.user_score_fraction*100)+'%') : null;
    const e = data.extracted_submission_score;
    const et = e&&e.text? e.text : (e&&e.extracted_score_text? e.extracted_score_text : '—');
    const ec = e&&e.confidence!=null? (' conf '+e.confidence) : '';
    return `<h3>Kết quả tổng quan</h3>
      <div class=muted>${esc(data.assistant_message||'')}</div>
      <div>Mục tiêu: <b>${esc(g)}</b>${u?` • Điểm bạn nhập: <b>${esc(u)}</b>`:''} • Điểm đọc từ ảnh: <b>${esc(et)}</b><span class=muted>${esc(ec)}</span></div>`
  }
  function rEval(ev){
    if(!ev||!Array.isArray(ev.items)) return '<h3>Đánh giá bài làm</h3><div class=muted>Không có dữ liệu mục.</div>'
    let rows = ev.items.map(it=>{
      const ok = (it.is_marked_correct===true || it.llm_judgement_correct===true);
      const badge = ok? '<span class=ok>Đúng</span>' : '<span class=bad>Sai</span>';
      return `<tr>
        <td>${esc(it.label||'—')}</td>
        <td>${badge}</td>
        <td>${esc(it.skillId||'—')}</td>
        <td>${esc(it.points_earned!=null? it.points_earned : '—')} / ${esc(it.points!=null? it.points : '—')}</td>
        <td>${esc(it.student_answer||'')}</td>
        <td class=muted>${esc(it.rationale||'')}</td>
      </tr>`
    }).join('')
    return `<h3>Đánh giá bài làm</h3>
    <table><thead><tr><th>Mục</th><th>KQ</th><th>Kỹ năng</th><th>Điểm</th><th>Bài làm</th><th>Giải thích</th></tr></thead>
    <tbody>${rows}</tbody></table>`
  }
  
  function rGen(list){
    if(!Array.isArray(list)||!list.length) return ''
    const rows = list.map((x,i)=>`<tr><td>${i+1}</td><td>${esc(x.skillId||'')}</td><td>${esc(x.question||'')}</td><td class=muted>${esc(x.answer||'')}</td></tr>`).join('')
    return `<h3>Câu hỏi LLM sinh thêm</h3><table><thead><tr><th>#</th><th>Skill</th><th>Câu hỏi</th><th>Đáp án</th></tr></thead><tbody>${rows}</tbody></table>`
  }
  const f=document.getElementById('f');
  f.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const data = new FormData(f);
    const res = await fetch('/agent/suggest-questions', { method:'POST', body:data }); if(!res.ok){ const err = await res.text(); document.getElementById('summary').innerHTML = '<div class=\\'card\\' style=\\'color:#b00020\\'>' + 'Error ' + res.status + ': ' + err + '</div>'; return; }
    const js = await res.json();
    document.getElementById('summary').innerHTML = rSummary(js);
    document.getElementById('eval').innerHTML = rEval(js.evaluation);
    document.getElementById('gen').innerHTML = rGen(js.llm_generated_questions);
  });
  </script>
</body></html>
"""

@app.post("/agent/suggest-questions")
async def agent_suggest_questions(
    exam_image: Optional[UploadFile] = File(None),
    submission_image: Optional[UploadFile] = File(None),
    goal_score: Optional[float] = Form(None),
    goal_score_text: Optional[str] = Form(None),
    user_score_text: Optional[str] = Form(None),
    user_points_earned: Optional[float] = Form(None),
    user_points_total: Optional[float] = Form(None),
    max_questions: int = Form(6),
    language: str = Form("vi"),
    source: str = Form("llm"),  # llm | bank | hybrid
):
    """
    Chatbot-like endpoint: user provides exam/submission images and a goal score.
    The agent analyzes images with Gemini, infers weak skills from available skill set, and
    returns a tailored set of questions from the item bank plus LLM-generated practice if needed.
    """
    model = _get_gemini_model()

    # Save and upload images if present
    exam_ref = None
    sub_ref = None
    temp_paths = []
    try:
        exam_ref = await _maybe_upload_image(exam_image, "exam.png")
        sub_ref = await _maybe_upload_image(submission_image, "submission.png")

        analysis = {"derived_from": "evaluation"}

        # Evaluate submission vs exam to get per-item judgments and map to skills
        evaluation = None
        if exam_ref is not None or sub_ref is not None:
            eval_prompt = {
                "task": "evaluate_submission_items",
                "instructions": [
                    "Analyze the exam and submission images.",
                    "Use OCR to extract each question text succinctly as 'question'.",
                    "Return per-item judgments with a free-form 'skill_tag' that describes the math skill/topic (e.g., FRAC.SIMPLIFY, EQ.SOLVE_1VAR).",
                    "Include a short 'question' text if you can parse it from the exam image.",
                    "If question numbering is visible, use it as 'label'.",
                    "Return JSON only; be concise in rationale.",
                ],
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "question": {"type": ["string","null"]},
                                    "skill_tag": {"type": ["string","null"]},
                                    "student_answer": {"type": ["string","null"]},
                                    "is_marked_correct": {"type": ["boolean","null"]},
                                    "llm_judgement_correct": {"type": ["boolean","null"]},
                                    "points": {"type": ["number","null"]},
                                    "points_earned": {"type": ["number","null"]},
                                    "rationale": {"type": ["string","null"]}
                                },
                                "required": ["label"],
                                "additionalProperties": False
                            }
                        },
                        "extracted_submission_score": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "value": {"type": ["number","null"]},
                                "denominator": {"type": ["number","null"]},
                                "confidence": {"type": "number"}
                            },
                            "required": ["text", "confidence"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False
                },
                "locale": language
            }
            parts_ev = [json.dumps(eval_prompt)]
            if exam_ref: parts_ev.append(exam_ref)
            if sub_ref: parts_ev.append(sub_ref)
            evresp = model.generate_content(parts_ev)
            evtext = getattr(evresp, "text", None) or "{}"
            try:
                evaluation = json.loads(evtext)
            except Exception:
                evaluation = {"raw": evtext}
        # Normalize evaluation skill field for UI (prefer skill_tag)
        if isinstance(evaluation, dict) and isinstance(evaluation.get("items"), list):
            for it in evaluation["items"]:
                if isinstance(it, dict):
                    if it.get("skill_tag") and not it.get("skillId"):
                        it["skillId"] = it.get("skill_tag")

        # Analyze exam style to align difficulty and format
        exam_style = None
        if exam_ref is not None:
            exam_style = _analyze_exam_style(model, exam_ref, language)

        # Extract on-paper score if possible from submission image
        extracted_score = None
        if sub_ref is not None:
            score_prompt = {
                "task": "extract_submission_score",
                "instructions": [
                    "Read and extract the printed/handwritten score on the submission if present.",
                    "Return JSON only.",
                ],
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "value": {"type": ["number", "null"]},
                        "denominator": {"type": ["number", "null"]},
                        "confidence": {"type": "number"}
                    },
                    "required": ["text", "confidence"],
                    "additionalProperties": False
                },
                "locale": language
            }
            sresp = model.generate_content([json.dumps(score_prompt), sub_ref])
            stext = getattr(sresp, "text", None) or "{}"
            try:
                extracted_score = json.loads(stext)
            except Exception:
                extracted_score = {"raw": stext}

        # Determine goal fraction
        goal_frac = _parse_goal(goal_score, goal_score_text)
        # Determine user score fraction (manual input)
        user_frac = None
        if user_points_earned is not None and user_points_total:
            try:
                if float(user_points_total) != 0:
                    user_frac = max(0.0, min(1.0, float(user_points_earned)/float(user_points_total)))
            except Exception:
                user_frac = None
        if user_frac is None and user_score_text:
            user_frac = _parse_goal(None, user_score_text)
        # Determine simple severity->theta mapping
        def theta_from_severity(sev: Optional[float]):
            if sev is None:
                return -0.2
            try:
                s = float(sev)
            except Exception:
                return -0.2
            if s >= 0.7:
                return -0.6
            if s >= 0.4:
                return -0.35
            return -0.1

        # Build weak skills + procedural error buckets purely from evaluation
        weak = []
        proc_buckets = {}
        # Augment weak skills using evaluation wrong items
        if isinstance(evaluation, dict) and isinstance(evaluation.get("items"), list):
            wrong_count = {}
            total_wrong = 0
            for it in evaluation["items"]:
                ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
                if ok:
                    continue
                sid = it.get("skill_tag") or it.get("skillId") or it.get("skill")
                if not sid:
                    continue
                wrong_count[sid] = wrong_count.get(sid, 0) + 1
                total_wrong += 1
                etype = it.get("error_type") or "UNKNOWN"
                key = (sid, etype)
                b = proc_buckets.get(key, {"skillId": sid, "error_type": etype, "count": 0, "example": None})
                b["count"] += 1
                if not b["example"]:
                    b["example"] = {
                        "label": it.get("label"),
                        "question": it.get("question"),
                        "solution_steps": it.get("solution_steps"),
                        "error_step_index": it.get("error_step_index"),
                        "error_explanation": it.get("error_explanation"),
                        "procedure_name": it.get("procedure_name")
                    }
                proc_buckets[key] = b
            if total_wrong > 0:
                for sid, cnt in wrong_count.items():
                    sev = cnt/total_wrong
                    weak.append({"skillId": sid, "severity": round(float(sev),2), "evidence": f"{cnt}/{total_wrong} items wrong"})
        # Merge duplicates: keep max severity
        merged = {}
        for w in weak:
            sid = w.get("skillId")
            if not sid:
                continue
            cur = merged.get(sid, {"skillId": sid, "severity": 0.0})
            sev = w.get("severity")
            try:
                sev = float(sev) if sev is not None else 0.0
            except Exception:
                sev = 0.0
            cur["severity"] = max(cur.get("severity", 0.0), sev)
            if w.get("evidence"):
                cur["evidence"] = w["evidence"]
            merged[sid] = cur
        weak = list(merged.values())
        # Fallback if model failed
        if not weak:
            weak = [{"skillId": "GENERAL.REVIEW", "severity": 0.5}]

        selected = []

        # Always generate support questions with a plan based on weak skills and goal gap
        support_plan = _build_support_plan(weak, goal_frac, user_frac, max_questions)
        gen_questions = []
        # Build error summary for UI/debug
        error_summary = []
        if proc_buckets:
            error_summary = [
                {"skillId": b["skillId"], "error_type": b["error_type"], "count": b["count"]}
                for b in proc_buckets.values()
            ]

        if support_plan:
            # Build a single generation request according to the plan
            gen_prompt = {
                "task": "generate_support_practice",
                "instructions": [
                    "Generate short, clear math questions suited for middle school.",
                    "Follow the plan: for each skillId, produce the requested counts per difficulty (easy/medium/hard).",
                    "Calibrate difficulty RELATIVE to the exam style: use similar notation, length and steps; medium/hard should feel like the exam.",
                    "Use seed questions (from the student's wrong items) to create close variants that preserve structure but change numbers/parameters.",
                    "Provide final answers and a concise solution_outline using the same language style; avoid LaTeX and images.",
                    "Return JSON array only.",
                ],
                "plan": support_plan,
                "goal_fraction": goal_frac,
                "user_score_fraction": user_frac,
                "aim": "increase" if (goal_frac or 0) > (user_frac or 0) else "consolidate",
                "observed_errors": [
                    {k: it.get(k) for k in ("label","question","skill_tag","skillId","student_answer","points","points_earned","rationale")}
                    for it in (evaluation.get("items") if isinstance(evaluation, dict) else []) if isinstance(it, dict)
                ],
                "procedural_focus": list(proc_buckets.values()),
                "exam_style": exam_style,
                "output_schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skillId": {"type": "string"},
                            "difficulty_tag": {"type": "string"},
                            "question": {"type": "string"},
                            "answer": {"type": ["string","number"]},
                            "solution_outline": {"type": ["string","null"]}
                        },
                        "required": ["skillId","difficulty_tag","question","answer"],
                        "additionalProperties": False
                    }
                },
                "locale": language
            }
            parts = [json.dumps(gen_prompt)]
            if exam_ref: parts.append(exam_ref)
            if sub_ref: parts.append(sub_ref)
            gresp = model.generate_content(parts)
            gtext = getattr(gresp, "text", None) or "[]"
            try:
                gen_questions = json.loads(gtext)
                if not isinstance(gen_questions, list):
                    gen_questions = []
            except Exception:
                gen_questions = [{"raw": gtext}]

        # Build assistant message summary
        goal_str = goal_score_text or (f"{goal_score*100:.0f}%" if isinstance(goal_score, (int,float)) else None)
        summary = {
            "vi": "Mình đã phân tích bài làm/đề thi và sinh bộ câu hỏi bổ trợ theo mục tiêu.",
            "en": "I analyzed your exam/submission and generated targeted practice for your goal."
        }
        assistant_message = summary.get("vi" if language.startswith("vi") else "en")

        return {
            "assistant_message": assistant_message,
            "goal_fraction": goal_frac,
            "user_score_fraction": user_frac,
            "analysis": analysis,
            "extracted_submission_score": extracted_score,
            "evaluation": evaluation,
            "error_summary": error_summary,
            "recommended_items_from_bank": [],
            "llm_generated_questions": gen_questions
        }
    finally:
        # temp files are cleaned in _maybe_upload_image
        pass

@app.post("/agent/generate-questions-by-levels")
async def generate_questions_by_levels(
    base_question: Optional[str] = Form(None),
    levels: Optional[str] = Form("easy:2,medium:2,hard:2"),
    exam_image: Optional[UploadFile] = File(None),
    submission_image: Optional[UploadFile] = File(None),
    skills_hint: Optional[str] = Form(None),  # comma separated skillIds
    language: str = Form("vi"),
):
    """
    Generate new practice questions across requested difficulty levels.
    - If base_question is provided, produce similar questions at varying difficulty.
    - If exam/submission images are provided, infer topic/skills and then generate.
    Returns JSON with a list of questions, each including: level, skillId, question, answer.
    """
    model = _get_gemini_model()
    lvl_spec = _parse_levels_spec(levels)
    available_skills = _unique_skills_from_bank()
    skills_hint_list = [s.strip() for s in (skills_hint or "").split(',') if s.strip()]

    # upload optional images
    exam_ref = None
    sub_ref = None
    temp_paths: List[str] = []
    try:
        if exam_image is not None:
            e_suffix = os.path.splitext(exam_image.filename or "exam.png")[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=e_suffix) as ef:
                ef.write(await exam_image.read())
                exam_tmp = ef.name
                temp_paths.append(exam_tmp)
            exam_ref = _upload_to_gemini(exam_tmp, display_name=os.path.basename(exam_image.filename or "exam"))
        if submission_image is not None:
            s_suffix = os.path.splitext(submission_image.filename or "submission.png")[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=s_suffix) as sf:
                sf.write(await submission_image.read())
                sub_tmp = sf.name
                temp_paths.append(sub_tmp)
            sub_ref = _upload_to_gemini(sub_tmp, display_name=os.path.basename(submission_image.filename or "submission"))

        # Build prompt
        prompt = {
            "task": "generate_questions_by_levels",
            "instructions": [
                "Generate concise math questions suitable for middle school students.",
                "Create variants at the requested difficulty levels.",
                "Each question should be solvable without external resources.",
                "Return strictly valid JSON only.",
            ],
            "levels": [{"level": lvl, "count": cnt} for (lvl, cnt) in lvl_spec],
            "base_question": base_question,
            "available_skills": available_skills,
            "skills_hint": skills_hint_list,
            "output_schema": {
                "type": "object",
                "properties": {
                    "inferred_skills": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "level": {"type": "string"},
                                "skillId": {"type": ["string","null"]},
                                "question": {"type": "string"},
                                "answer": {"type": ["string","number"]},
                                "solution_outline": {"type": ["string","null"]}
                            },
                            "required": ["level","question","answer"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["questions"],
                "additionalProperties": False
            },
            "locale": language
        }
        parts = [json.dumps(prompt)]
        if exam_ref: parts.append(exam_ref)
        if sub_ref: parts.append(sub_ref)
        resp = model.generate_content(parts)
        text = getattr(resp, "text", None) or "{}"
        try:
            data = json.loads(text)
        except Exception:
            data = {"raw": text}

        # Post-process: normalize level tags
        if isinstance(data, dict) and isinstance(data.get("questions"), list):
            for q in data["questions"]:
                if isinstance(q, dict) and "level" in q:
                    q["level"] = _normalize_level_tag(str(q.get("level")))
        return data
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

@app.post("/agent/generate-support-practice")
async def generate_support_practice(
    exam_image: Optional[UploadFile] = File(None),
    submission_image: Optional[UploadFile] = File(None),
    goal_score: Optional[float] = Form(None),
    goal_score_text: Optional[str] = Form(None),
    user_score_text: Optional[str] = Form(None),
    user_points_earned: Optional[float] = Form(None),
    user_points_total: Optional[float] = Form(None),
    max_questions: int = Form(6),
    language: str = Form("vi"),
):
    """
    Generate supportive practice based on exam+submission evaluation, target score, and user's score.
    Returns a plan and generated questions tailored to weak skills and improvement gap.
    """
    model = _get_gemini_model()
    # Save and upload images if present
    exam_ref = None
    sub_ref = None
    temp_paths = []
    try:
        exam_ref = await _maybe_upload_image(exam_image, "exam.png")
        sub_ref = await _maybe_upload_image(submission_image, "submission.png")

        # Evaluate
        evaluation = None
        if exam_ref is not None or sub_ref is not None:
            eval_prompt = {
                "task": "evaluate_submission_items",
                "instructions": [
                    "Analyze the exam and submission images.",
                    "Use OCR to extract each question text succinctly as 'question'.",
                    "Return per-item judgments with a free-form 'skill_tag' that describes the math skill/topic (e.g., FRAC.SIMPLIFY, EQ.SOLVE_1VAR).",
                    "Extract the student's solution_steps as an ordered list of concise steps (strings).",
                    "If the solution is incorrect, classify the primary error_type and locate which step it occurs at (error_step_index).",
                    "Use one of these error types when possible: ARITHMETIC_MISTAKE, MISAPPLY_UNIT_MEANING, WRONG_OPERATION, SIGN_ERROR, STEP_SKIPPED, TRANSPOSITION_ERROR, FRACTION_COMMON_DENOMINATOR, ORDER_OF_OPERATIONS.",
                    "If question numbering is visible, use it as 'label'.",
                    "Return JSON only; be concise in rationale.",
                ],
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "question": {"type": ["string","null"]},
                                    "skill_tag": {"type": ["string","null"]},
                                    "solution_steps": {
                                        "type": ["array","null"],
                                        "items": {"type": "string"}
                                    },
                                    "error_type": {"type": ["string","null"]},
                                    "error_step_index": {"type": ["number","null"]},
                                    "error_explanation": {"type": ["string","null"]},
                                    "procedure_name": {"type": ["string","null"]},
                                    "student_answer": {"type": ["string","null"]},
                                    "is_marked_correct": {"type": ["boolean","null"]},
                                    "llm_judgement_correct": {"type": ["boolean","null"]},
                                    "points": {"type": ["number","null"]},
                                    "points_earned": {"type": ["number","null"]},
                                    "rationale": {"type": ["string","null"]}
                                },
                                "required": ["label"],
                                "additionalProperties": False
                            }
                        },
                        "extracted_submission_score": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "value": {"type": ["number","null"]},
                                "denominator": {"type": ["number","null"]},
                                "confidence": {"type": "number"}
                            },
                            "required": ["text", "confidence"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False
                },
                "locale": language
            }
            parts_ev = [json.dumps(eval_prompt)]
            if exam_ref: parts_ev.append(exam_ref)
            if sub_ref: parts_ev.append(sub_ref)
            evresp = model.generate_content(parts_ev)
            evtext = getattr(evresp, "text", None) or "{}"
            try:
                evaluation = json.loads(evtext)
            except Exception:
                evaluation = {"raw": evtext}

        # Weak skills
        weak = []
        if isinstance(evaluation, dict) and isinstance(evaluation.get("items"), list):
            wrong_count = {}
            total_wrong = 0
            for it in evaluation["items"]:
                ok = (it.get("is_marked_correct") is True) or (it.get("llm_judgement_correct") is True)
                if ok:
                    continue
                sid = it.get("skill_tag") or it.get("skillId") or it.get("skill")
                if not sid:
                    continue
                wrong_count[sid] = wrong_count.get(sid, 0) + 1
                total_wrong += 1
            if total_wrong > 0:
                for sid, cnt in wrong_count.items():
                    sev = cnt/total_wrong
                    weak.append({"skillId": sid, "severity": round(float(sev),2), "evidence": f"{cnt}/{total_wrong} items wrong"})
        if not weak:
            weak = [{"skillId": "GENERAL.REVIEW", "severity": 0.5}]

        # Goal and user fraction
        goal_frac = _parse_goal(goal_score, goal_score_text)
        user_frac = None
        if user_points_earned is not None and user_points_total:
            try:
                if float(user_points_total) != 0:
                    user_frac = max(0.0, min(1.0, float(user_points_earned)/float(user_points_total)))
            except Exception:
                user_frac = None
        if user_frac is None and user_score_text:
            user_frac = _parse_goal(None, user_score_text)

        # Build support plan
        support_plan = _build_support_plan(weak, goal_frac, user_frac, max_questions)

        # Generate questions
        gen_questions = []
        if support_plan:
            gen_prompt = {
                "task": "generate_support_practice",
                "instructions": [
                    "Generate short, clear math questions suited for middle school.",
                    "Follow the plan: for each skillId, produce the requested counts per difficulty (easy/medium/hard).",
                    "Calibrate difficulty RELATIVE to the exam style when possible.",
                    "Use seed questions (from the student's wrong items) to create close variants targeting the same procedural pitfalls.",
                    "Provide final answers and a concise solution_outline; avoid LaTeX and images.",
                    "Return JSON array only.",
                ],
                "plan": support_plan,
                "observed_errors": [
                    {k: it.get(k) for k in ("label","question","skill_tag","skillId","student_answer","points","points_earned","rationale")}
                    for it in (evaluation.get("items") if isinstance(evaluation, dict) else []) if isinstance(it, dict)
                ],
                "procedural_focus": list(proc_buckets.values()),
                "output_schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "skillId": {"type": "string"},
                            "difficulty_tag": {"type": "string"},
                            "question": {"type": "string"},
                            "answer": {"type": ["string","number"]},
                            "solution_outline": {"type": ["string","null"]}
                        },
                        "required": ["skillId","difficulty_tag","question","answer"],
                        "additionalProperties": False
                    }
                },
                "locale": language
            }
            parts = [json.dumps(gen_prompt)]
            if exam_ref: parts.append(exam_ref)
            if sub_ref: parts.append(sub_ref)
            gresp = model.generate_content(parts)
            gtext = getattr(gresp, "text", None) or "[]"
            try:
                gen_questions = json.loads(gtext)
                if not isinstance(gen_questions, list):
                    gen_questions = []
            except Exception:
                gen_questions = [{"raw": gtext}]

        assistant_message = "Đã tạo bộ bài luyện bổ trợ theo khoảng khó và kỹ năng yếu." if language.startswith("vi") else "Generated support practice based on weak skills and goal gap."
        return {
            "assistant_message": assistant_message,
            "goal_fraction": goal_frac,
            "user_score_fraction": user_frac,
            "evaluation": evaluation,
            "support_plan": support_plan,
            "support_questions": gen_questions
        }
    finally:
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

@app.post("/student/profile")
def student_profile(req: ProfileReq):
    exams = load_exams()
    goals = load_goals()
    items_bank = load_items()

    responses = []
    per_exam_score = []
    for exam in exams:
        stu = [r for r in exam["responses"] if r["studentId"]==req.studentId]
        if not stu:
            continue
        responses.extend([(r, exam) for r in stu])
        tot = sum(float(r.get("score",0.0)) for r in stu)
        per_exam_score.append({"examId": exam["examId"], "total": tot})

    if not responses:
        raise HTTPException(404, f"No responses for {req.studentId} in demo data.")

    skill_obs = {}
    for r, exam in responses:
        skills = next((it["skillIds"] for it in exam["items"] if it["itemId"]==r["itemId"]), [])
        for s in skills:
            skill_obs.setdefault(s, []).append(1 if r["isCorrect"] else 0)
    mastery = [{"skillId": s, "p": round(float(np.mean(v)),2)} for s,v in skill_obs.items()]

    targets = {g["skillId"]: g["targetP"] for g in goals.get(req.studentId,[]) if g["goal"]=="skill_mastery"}
    gaps = []
    for m in mastery:
        tgt = targets.get(m["skillId"], 0.7)
        gap = round(tgt - m["p"], 2)
        if gap > 0:
            gaps.append({"skillId": m["skillId"], "p": m["p"], "target": tgt, "gap": gap})
    gaps = sorted(gaps, key=lambda x: x["gap"], reverse=True)[:req.topKSkills]

    guidance = []
    for exam in exams:
        students = sorted({r["studentId"] for r in exam["responses"]})
        totals = {sid: 0.0 for sid in students}
        for r in exam["responses"]:
            totals[r["studentId"]] += float(r.get("score", 0.0))
        item_ids = [it["itemId"] for it in exam["items"]]
        totals_without = {iid: {sid: totals[sid] for sid in students} for iid in item_ids}
        for iid in item_ids:
            for sid in students:
                sc = 0.0
                for r in exam["responses"]:
                    if r["studentId"]==sid and r["itemId"]==iid:
                        sc = float(r.get("score",0.0))
                        break
                totals_without[iid][sid] -= sc
        pbis_map = {}
        for it in exam["items"]:
            iid = it["itemId"]
            obs = [int(r["isCorrect"]) for r in exam["responses"] if r["itemId"]==iid]
            if obs:
                pbis_map[iid] = float(np.corrcoef(np.array(obs, dtype=float), np.array([totals_without[iid][sid] for sid in students], dtype=float))[0,1])
            else:
                pbis_map[iid] = 0.0

        wrongs = [r for r in exam["responses"] if r["studentId"]==req.studentId and not r["isCorrect"]]
        wrongs = sorted(wrongs, key=lambda x: pbis_map.get(x["itemId"],0.0), reverse=True)
        if wrongs:
            w = wrongs[0]
            skills = next((it["skillIds"] for it in exam["items"] if it["itemId"]==w["itemId"]), [])
            guidance.append({
                "examId": exam["examId"],
                "itemId": w["itemId"],
                "skillId": skills[0] if skills else "UNKNOWN",
                "why": f"Sai {w['itemId']} (pbis cao) ⇒ ưu tiên ôn kỹ năng liên quan.",
                "how": "Ôn quy tắc cốt lõi, làm 2 ví dụ tương tự khác số, sau đó làm lại câu tương tự."
            })

    return {
        "studentId": req.studentId,
        "perExam": per_exam_score,
        "mastery": mastery,
        "gaps": gaps,
        "guidance": guidance
    }

ERROR_ENUM = [
  "ARITHMETIC_MISTAKE",
  "MISAPPLY_UNIT_MEANING",
  "WRONG_OPERATION",
  "SIGN_ERROR",
  "STEP_SKIPPED",
  "TRANSPOSITION_ERROR",
  "FRACTION_COMMON_DENOMINATOR",
  "ORDER_OF_OPERATIONS"
]

@app.post("/agent/diagnose-hint")
def agent_diagnose(req: AgentReq):
    text = " ".join(req.student_steps).lower()
    problem = req.problem.lower()
    error_type = "ARITHMETIC_MISTAKE"
    root_cause = "Chưa rõ."
    step_index = 0

    if "box" in problem or "hộp" in problem:
        if ("120/8" in text or "120 ÷ 8" in text or "120 / 8" in text) and ("- 3" in text or "− 3" in text or "trừ 3" in text):
            error_type = "MISAPPLY_UNIT_MEANING"
            root_cause = "Nhầm 12 là tổng. 12 là số bánh MỖI hộp; cần nhân 8 hộp rồi mới trừ phần đã bán."
            step_index = 0

    if ("phân số" in problem or "/" in problem) and ("+" in text or "cộng" in text):
        if "quy đồng" not in text and "mẫu chung" not in text:
            error_type = "FRACTION_COMMON_DENOMINATOR"
            root_cause = "Thiếu bước quy đồng mẫu số trước khi cộng/trừ."
            step_index = 0

    next_level = min(req.max_hint_level, max(req.current_hint_level + (1 if req.wrong_attempts_on_this_step >= 2 else 0), 1))

    if error_type == "MISAPPLY_UNIT_MEANING":
        hints = {
            1: "Em đã tìm được số bánh trong MỖI hộp. Kiểm tra lại: tổng bánh là bao nhiêu?",
            2: "Quy tắc: Tổng = (số hộp) × (bánh MỖI hộp). Bán 3 HỘP ⇒ trừ bánh của 3 HỘP.",
            3: "Ví dụ: 100 kẹo chia đều 5 túi. Mỗi túi bao nhiêu? Bán 2 túi còn bao nhiêu?",
            4: "Bước: (1) Bánh mỗi hộp. (2) Tổng = 8 × (mỗi hộp). (3) Trừ bánh của 3 hộp."
        }
        socratic = "Nếu biết bánh MỖI hộp và số hộp, phép tính nào cho tổng? Khi bán 3 HỘP thì trừ bao nhiêu chiếc?"
    elif error_type == "FRACTION_COMMON_DENOMINATOR":
        hints = {
            1: "Mẫu số đã giống nhau chưa?",
            2: "Quy tắc: Quy đồng mẫu (LCM), rồi cộng tử, giữ mẫu.",
            3: "Ví dụ: 1/3 + 1/2 ⇒ mẫu chung 6 ⇒ 2/6 + 3/6 = ?",
            4: "Bước: Tìm LCM → đổi về mẫu chung → cộng tử, giữ mẫu."
        }
        socratic = "Vì sao cần quy đồng mẫu trước khi cộng? Mẫu chung nhỏ nhất là gì?"
    else:
        hints = {
            1: "Xem lại ý nghĩa từng số và phép tính.",
            2: "Đổi về cùng đơn vị, chọn phép tính đúng theo ý nghĩa.",
            3: "Thử ví dụ nhỏ hơn để kiểm tra suy luận.",
            4: "Liệt kê đại lượng → áp dụng phép tính cho từng bước."
        }
        socratic = "Nếu đổi góc nhìn khác, phép tính bước này có hợp lý không?"

    return {
        "diagnosis": {
            "error_type": error_type,
            "root_cause": root_cause,
            "step_index": step_index,
            "confidence": 0.8
        },
        "next_hint": {
            "hint_level": next_level,
            "hint_text": hints[next_level],
            "socratic_question": socratic,
            "do_not_reveal_answer": True
        },
        "grade_explanation": f"Lớp {req.grade}: Chọn phép tính theo ý nghĩa đại lượng trước khi trừ phần đã bán.",
        "safety_flags": {"reveals_final_answer": False, "off_grade": False}
    }

@app.post("/recommendations/playlist")
def playlist(req: PlaylistReq):
    items_bank = load_items()
    exams = load_exams()

    skill_obs = {}
    for exam in exams:
        for r in exam["responses"]:
            if r["studentId"] == req.studentId:
                skills = next((it["skillIds"] for it in exam["items"] if it["itemId"]==r["itemId"]), [])
                for s in skills:
                    skill_obs.setdefault(s, []).append(1 if r["isCorrect"] else 0)

    if not skill_obs:
        raise HTTPException(404, f"No responses for student {req.studentId} in demo data.")

    skill_theta = {s: (float(np.mean(v))*2 - 1) for s,v in skill_obs.items()}

    cand = []
    for it in items_bank:
        if req.focusSkills and not any(s in req.focusSkills for s in it["skillIds"]):
            continue
        s = it["skillIds"][0]
        theta = skill_theta.get(s, -0.2)
        b = it.get("difficulty_b", 0.0)
        p_hat = 1/(1+math.exp(-(theta - b)))
        if 0.6 <= p_hat <= 0.8:
            cand.append({
                "itemId": it["itemId"],
                "skillId": s,
                "level": "L3" if p_hat<=0.7 else "L2",
                "pHat": round(float(p_hat),2),
                "reason": f"ZPD {round(float(p_hat),2)} trên kỹ năng {s}",
                "due": "2025-10-25"
            })

    cand = sorted(cand, key=lambda x: -x["pHat"])
    return {"studentId": req.studentId, "playlist": cand[:req.maxItems]}







