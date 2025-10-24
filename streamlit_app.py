import os
import io
import json
import tempfile
from typing import Optional, Any, Dict, List

import streamlit as st
from dotenv import load_dotenv


# Load .env for GOOGLE_API_KEY when running locally
load_dotenv(override=False)

st.set_page_config(page_title="EduRec – Luyện đề bằng AI (Standalone)", page_icon="🧠", layout="centered")
st.title("🧠 EduRec – Phân tích ảnh đề + bài làm và sinh bài luyện")
st.caption("Chạy hoàn toàn trên Streamlit; cần GOOGLE_API_KEY trong môi trường.")


def get_model():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Thiếu GOOGLE_API_KEY/GEMINI_API_KEY trong môi trường hoặc Streamlit secrets.")
        st.stop()
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = {"temperature": 0.2, "response_mime_type": "application/json"}
    return genai.GenerativeModel(model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"), generation_config=generation_config)


def is_likely_image_bytes(b: bytes) -> bool:
    if not b or len(b) < 4:
        return False
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if b.startswith(b"\xff\xd8"):
        return True
    if b.startswith(b"GIF8"):
        return True
    if b.startswith(b"RIFF") and b[8:12] == b"WEBP":
        return True
    return False


def upload_bytes_to_gemini(name: str, data: bytes):
    import google.generativeai as genai
    suffix = os.path.splitext(name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
        tf.write(data)
        path = tf.name
    try:
        return genai.upload_file(path=path, display_name=name)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def parse_goal(goal_text: Optional[str]) -> Optional[float]:
    if not goal_text:
        return None
    t = goal_text.strip().replace(" ", "")
    try:
        if t.endswith("%"):
            return max(0.0, min(1.0, float(t[:-1]) / 100.0))
        if "/" in t:
            a, b = t.split("/", 1)
            a = float(a); b = float(b)
            if b != 0:
                return max(0.0, min(1.0, a / b))
        v = float(t)
        if 0 <= v <= 1:
            return v
        if 1 < v <= 100:
            return v / 100.0
    except Exception:
        return None
    return None


def build_support_plan(weak: List[Dict[str, Any]], goal_frac: Optional[float], user_frac: Optional[float], total_q: int) -> List[Dict[str, Any]]:
    if total_q <= 0 or not weak:
        return []
    weights = [(w.get("skillId"), float(w.get("severity", 0.3))) for w in weak if w.get("skillId")]
    weights = [(sid, max(0.01, sev)) for sid, sev in weights]
    if not weights:
        return []
    weights = sorted(weights, key=lambda x: -x[1])[:3]
    sw = sum(w for _, w in weights) or 1.0
    alloc = [(sid, max(1, round(total_q * w / sw))) for sid, w in weights]
    diff = total_q - sum(c for _, c in alloc)
    i = 0
    while diff != 0 and alloc:
        sid, c = alloc[i % len(alloc)]
        if diff > 0:
            alloc[i % len(alloc)] = (sid, c + 1); diff -= 1
        else:
            if c > 1:
                alloc[i % len(alloc)] = (sid, c - 1); diff += 1
        i += 1
    delta = None
    if goal_frac is not None and user_frac is not None:
        delta = max(-1.0, min(1.0, float(goal_frac) - float(user_frac)))
    plan = []
    sev_map = {w.get("skillId"): float(w.get("severity", 0.3)) for w in weak if w.get("skillId")}
    for sid, count in alloc:
        sev = sev_map.get(sid, 0.3)
        if sev >= 0.7:
            mix = {"easy": 0.2, "medium": 0.5, "hard": 0.3}
        elif sev >= 0.4:
            mix = {"easy": 0.1, "medium": 0.5, "hard": 0.4}
        else:
            mix = {"easy": 0.05, "medium": 0.45, "hard": 0.5}
        if delta is not None:
            if delta >= 0.2:
                mix["easy"] = max(0.0, mix["easy"] - 0.1)
                mix["hard"] = min(0.7, mix["hard"] + 0.1)
            elif delta <= -0.1:
                mix["easy"] = min(0.5, mix["easy"] + 0.1)
                mix["hard"] = max(0.1, mix["hard"] - 0.1)
        e = max(0, round(count * mix["easy"]))
        m = max(0, round(count * mix["medium"]))
        h = max(0, round(count * mix["hard"]))
        tot = e + m + h
        while tot < count:
            if m <= max(e, h): m += 1
            elif e <= h: e += 1
            else: h += 1
            tot = e + m + h
        while tot > count:
            if m >= max(e, h) and m > 0: m -= 1
            elif e >= h and e > 0: e -= 1
            elif h > 0: h -= 1
            tot = e + m + h
        plan.append({"skillId": sid, "counts": {"easy": int(e), "medium": int(m), "hard": int(h)}})
    return plan


def analyze_exam_style(model, exam_ref, language: str):
    try:
        prompt = {
            "task": "profile_exam_style",
            "instructions": [
                "Analyze the exam image and summarize style so new questions can match it.",
                "Report: preferred_response_type (FR or MCQ), notation highlights (fractions, algebraic forms), typical points per question, topic_distribution, example_difficulty_markers, and solution_format notes.",
                "Return strict JSON only."
            ],
            "output_schema": {"type": "object"},
            "locale": language
        }
        resp = model.generate_content([json.dumps(prompt), exam_ref])
        return json.loads(getattr(resp, "text", "{}"))
    except Exception:
        return {}


with st.sidebar:
    st.subheader("Thiết lập")
    lang = st.selectbox("Ngôn ngữ", options=["vi", "en"], index=0)
    max_q = st.number_input("Số câu gợi ý", min_value=1, max_value=20, value=6)
    time_budget = st.number_input("Thời gian luyện (phút, tuỳ chọn)", min_value=0, max_value=240, value=0)
    st.markdown("GOOGLE_API_KEY phải có trong môi trường (hoặc secrets)")

exams = st.file_uploader("Ảnh đề thi (có thể nhiều trang)", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=True)
subs = st.file_uploader("Ảnh bài làm (có thể nhiều trang)", type=["png","jpg","jpeg","webp","gif"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    goal_text = st.text_input("Mục tiêu điểm (vd 8/10 hoặc 80%)", value="")
with col2:
    user_text = st.text_input("Điểm của bạn (vd 6/10 hoặc 60%)", value="")

run = st.button("Phân tích & Gợi ý luyện tập", type="primary")

if run:
    model = get_model()

    exam_refs = []
    sub_refs = []
    # Upload multiple pages (if any); warn on tiny files
    for f in (exams or []):
        data = f.getvalue()
        if data and len(data) < 20_000:
            st.warning(f"Ảnh đề '{f.name}' dung lượng thấp ({len(data)}B) – có thể OCR kém.")
        if is_likely_image_bytes(data):
            exam_refs.append(upload_bytes_to_gemini(f.name, data))
    for f in (subs or []):
        data = f.getvalue()
        if data and len(data) < 20_000:
            st.warning(f"Ảnh bài làm '{f.name}' dung lượng thấp ({len(data)}B) – có thể OCR kém.")
        if is_likely_image_bytes(data):
            sub_refs.append(upload_bytes_to_gemini(f.name, data))

    goal_frac = parse_goal(goal_text)
    user_frac = parse_goal(user_text)

    # Evaluate submission vs exam: extract steps, error types, skill tags
    evaluation = {}
    if exam_refs or sub_refs:
        eval_prompt = {
            "task": "evaluate_submission_items",
            "instructions": [
                "Analyze the exam and submission images.",
                "Use OCR to extract each question text succinctly as 'question'.",
                "Return per-item judgments with a free-form 'skill_tag' (e.g., FRAC.SIMPLIFY, EQ.SOLVE_1VAR).",
                "Extract the student's solution_steps (ordered strings) by TRANSCRIBING the student's handwriting exactly where possible (do not invent steps).",
                "For each item, add steps_confidence (0..1) that reflects confidence that steps were read from the student's work rather than inferred.",
                "If incorrect, classify error_type and locate error_step_index (0-based).",
                "Use error types: ARITHMETIC_MISTAKE, MISAPPLY_UNIT_MEANING, WRONG_OPERATION, SIGN_ERROR, STEP_SKIPPED, TRANSPOSITION_ERROR, FRACTION_COMMON_DENOMINATOR, ORDER_OF_OPERATIONS.",
                "If teacher marks/checks/crosses are visible, set is_marked_correct accordingly and increase confidence.",
                "If numbering is visible, use it as 'label'.",
                "Return JSON only; be concise in rationale.",
            ],
            "output_schema": {"type": "object"},
            "locale": lang,
        }
        parts = [json.dumps(eval_prompt)]
        parts += exam_refs
        parts += sub_refs
        evresp = model.generate_content(parts)
        try:
            evaluation = json.loads(getattr(evresp, "text", "{}"))
        except Exception:
            evaluation = {"raw": getattr(evresp, "text", "{}")}

    # Build weak skills and procedural buckets
    weak: List[Dict[str, Any]] = []
    proc_buckets: Dict[Any, Any] = {}
    err_freq_by_skill: Dict[str, int] = {}
    points_by_skill: Dict[str, List[float]] = {}
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
            err_freq_by_skill[sid] = err_freq_by_skill.get(sid, 0) + 1
            if isinstance(it.get("points"), (int, float)):
                points_by_skill.setdefault(sid, []).append(float(it.get("points")))
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
                }
            proc_buckets[key] = b
        if total_wrong > 0:
            for sid, cnt in wrong_count.items():
                sev = cnt / total_wrong
                weak.append({"skillId": sid, "severity": round(float(sev), 2), "evidence": f"{cnt}/{total_wrong} items wrong"})
    if not weak:
        weak = [{"skillId": "GENERAL.REVIEW", "severity": 0.5}]

    # Exam style to align difficulty/format
    exam_style = analyze_exam_style(model, exam_refs[0], lang) if exam_refs else {}

    # Aggregate: pbis by skill from demo data (if available)
    def load_demo_exams() -> List[dict]:
        base = os.path.join(os.path.dirname(__file__), "data", "exams")
        exams = []
        try:
            for fname in sorted(os.listdir(base)):
                if fname.endswith(".json"):
                    with open(os.path.join(base, fname), "r", encoding="utf-8") as f:
                        exams.append(json.load(f))
        except Exception:
            pass
        return exams

    def compute_pbis(arr_x: List[int], arr_y: List[float]) -> float:
        try:
            import numpy as np
            X = np.array(arr_x, dtype=float)
            Y = np.array(arr_y, dtype=float)
            if len(X) < 3 or np.std(X) == 0 or np.std(Y) == 0:
                return 0.0
            return float(np.corrcoef(X, Y)[0, 1])
        except Exception:
            return 0.0

    def pbis_by_skill_from_demo() -> Dict[str, float]:
        exams = load_demo_exams()
        if not exams:
            return {}
        pbis_items = []  # (skillId, pbis)
        for exam in exams:
            students = sorted({r["studentId"] for r in exam.get("responses", [])})
            totals = {sid: 0.0 for sid in students}
            for r in exam.get("responses", []):
                totals[r["studentId"]] += float(r.get("score", 0.0))
            item_ids = [it["itemId"] for it in exam.get("items", [])]
            totals_without = {iid: {sid: totals[sid] for sid in students} for iid in item_ids}
            for iid in item_ids:
                for sid in students:
                    sc = 0.0
                    for r in exam.get("responses", []):
                        if r["studentId"] == sid and r["itemId"] == iid:
                            sc = float(r.get("score", 0.0))
                            break
                    totals_without[iid][sid] -= sc
            for it in exam.get("items", []):
                iid = it["itemId"]
                obs = [int(r.get("isCorrect")) for r in exam.get("responses", []) if r.get("itemId") == iid]
                if not obs:
                    continue
                pb = compute_pbis(obs, [totals_without[iid][sid] for sid in students])
                for s in it.get("skillIds", []):
                    pbis_items.append((s, pb))
        by_skill: Dict[str, List[float]] = {}
        for s, v in pbis_items:
            by_skill.setdefault(s, []).append(float(v))
        return {s: round(sum(vs)/len(vs), 3) for s, vs in by_skill.items() if vs}

    pbis_by_skill = pbis_by_skill_from_demo()

    # Weighted plan under goal constraints (severity + points + pbis)
    # Normalize points and pbis for weak skills only
    def norm_map(d: Dict[str, float]) -> Dict[str, float]:
        if not d:
            return {}
        vals = [v for v in d.values() if isinstance(v, (int, float))]
        if not vals:
            return {}
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return {k: 0.5 for k in d}
        return {k: (float(v) - mn) / (mx - mn) for k, v in d.items()}

    avg_points_by_skill = {k: (sum(v)/len(v)) for k, v in points_by_skill.items() if v}
    norm_points = norm_map(avg_points_by_skill)
    norm_pbis = norm_map({k: pbis_by_skill.get(k, 0.0) for k in {w["skillId"] for w in weak}})

    # Blend into severity
    blended_weak: List[Dict[str, Any]] = []
    for w in weak:
        sid = w.get("skillId")
        base = float(w.get("severity", 0.3))
        pw = norm_points.get(sid, 0.5)
        pb = norm_pbis.get(sid, 0.5)
        blended = 0.6 * base + 0.25 * pw + 0.15 * pb
        blended_weak.append({"skillId": sid, "severity": round(float(blended), 3), "base_severity": base, "points_w": pw, "pbis_w": pb})

    # Build plan with blended weights
    support_plan = build_support_plan(blended_weak, goal_frac, user_frac, int(max_q))

    # Time constraint: estimate time per difficulty and trim to budget
    def estimate_time_minutes(plan: List[Dict[str, Any]]) -> float:
        t = 0.0
        for p in plan:
            c = p["counts"]
            t += c.get("easy", 0) * 1.5 + c.get("medium", 0) * 3.0 + c.get("hard", 0) * 5.0
        return round(t, 1)

    def trim_plan_to_time(plan: List[Dict[str, Any]], budget_min: int) -> List[Dict[str, Any]]:
        if not budget_min or budget_min <= 0:
            return plan
        cur = json.loads(json.dumps(plan))
        def total(curplan):
            return estimate_time_minutes(curplan)
        while total(cur) > budget_min:
            # greedily reduce hardest counts first
            reduced = False
            for p in cur:
                if p["counts"].get("hard", 0) > 0:
                    p["counts"]["hard"] -= 1; reduced = True; break
            if not reduced:
                for p in cur:
                    if p["counts"].get("medium", 0) > 0:
                        p["counts"]["medium"] -= 1; reduced = True; break
            if not reduced:
                for p in cur:
                    if p["counts"].get("easy", 0) > 0:
                        p["counts"]["easy"] -= 1; reduced = True; break
            if not reduced:
                break
        return cur

    planned_time = estimate_time_minutes(support_plan)
    if time_budget and time_budget > 0:
        support_plan = trim_plan_to_time(support_plan, int(time_budget))
    planned_time_after = estimate_time_minutes(support_plan)

    # Build plan and generate questions
    gen_questions: List[Dict[str, Any]] = []
    if support_plan:
        gen_prompt = {
            "task": "generate_support_practice",
            "instructions": [
                "Generate short, clear math questions suited for middle school.",
                "Follow the plan: for each skillId, produce the requested counts per difficulty (easy/medium/hard).",
                "Calibrate difficulty relative to the exam style; medium/hard should feel like the exam.",
                "Use seed questions (from the student's wrong items) to create close variants targeting the same procedural pitfalls.",
                "Provide final answers and a concise solution_outline; avoid LaTeX and images.",
                "Return JSON array only.",
            ],
            "plan": support_plan,
            "goal_fraction": goal_frac,
            "user_score_fraction": user_frac,
            "aim": "increase" if (goal_frac or 0) > (user_frac or 0) else "consolidate",
            "observed_errors": [
                {k: it.get(k) for k in ("label","question","skill_tag","student_answer","rationale","solution_steps","error_type","error_step_index","steps_confidence")}
                for it in (evaluation.get("items") if isinstance(evaluation, dict) else []) if isinstance(it, dict)
            ],
            "procedural_focus": list(proc_buckets.values()),
            "exam_style": exam_style,
            "output_schema": {"type": "array"},
            "locale": lang,
        }
        parts = [json.dumps(gen_prompt)]
        parts += exam_refs
        parts += sub_refs
        gresp = model.generate_content(parts)
        try:
            gen_questions = json.loads(getattr(gresp, "text", "[]"))
            if not isinstance(gen_questions, list):
                gen_questions = []
        except Exception:
            gen_questions = []

    # Render results
    st.subheader("Tóm tắt")
    ca, cb, cc = st.columns(3)
    with ca: st.metric("Mục tiêu", f"{round(goal_frac*100):d}%" if isinstance(goal_frac,(int,float)) else "–")
    with cb: st.metric("Điểm của bạn", f"{round(user_frac*100):d}%" if isinstance(user_frac,(int,float)) else "–")
    with cc:
        st.metric("Phong cách đề", exam_style.get("preferred_response_type") or "–")

    st.subheader("Đánh giá bài làm (theo câu)")
    items = evaluation.get("items") if isinstance(evaluation, dict) else []
    if items:
        rows = []
        for it in items:
            steps = it.get("solution_steps")
            if isinstance(steps, list):
                steps = " | ".join([str(s) for s in steps])
            rows.append({
                "Mục": it.get("label"),
                "Câu hỏi": it.get("question"),
                "Kỹ năng": it.get("skill_tag") or it.get("skillId"),
                "Đúng?": True if it.get("is_marked_correct") or it.get("llm_judgement_correct") else False,
                "Lỗi": it.get("error_type"),
                "Bước sai": it.get("error_step_index"),
                "Steps (trích)": steps,
                "Conf": it.get("steps_confidence"),
                "Bài làm": it.get("student_answer"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Chưa trích xuất được mục nào từ ảnh (kiểm tra ảnh rõ nét, đầy đủ).")

    if proc_buckets:
        st.subheader("Tổng hợp lỗi thủ tục")
        st.table([{"skillId": b["skillId"], "error_type": b["error_type"], "count": b["count"]} for b in proc_buckets.values()])

    # Goal constraints & aggregates
    st.subheader("Ràng buộc theo mục tiêu & tổng hợp")
    colg1, colg2, colg3 = st.columns(3)
    with colg1:
        st.metric("Tổng thời gian kế hoạch", f"{planned_time_after} phút")
    with colg2:
        st.metric("Ngân sách thời gian", f"{time_budget} phút" if time_budget else "—")
    with colg3:
        st.metric("Tần suất sai (kỹ năng hàng đầu)", max(err_freq_by_skill.values()) if err_freq_by_skill else 0)

    if err_freq_by_skill:
        st.write("Tần suất sai theo kỹ năng:")
        st.table([{"skillId": k, "wrong_count": v} for k, v in sorted(err_freq_by_skill.items(), key=lambda x: -x[1])])
    if pbis_by_skill:
        st.write("Độ phân biệt (PBIS) theo kỹ năng (demo data):")
        want_skills = {w["skillId"] for w in weak}
        rows = [{"skillId": k, "pbis": pbis_by_skill.get(k)} for k in sorted(want_skills)]
        st.table(rows)
    if avg_points_by_skill:
        st.write("Trọng số điểm mục (trung bình theo kỹ năng):")
        st.table([{ "skillId": k, "avg_points": round(v,2)} for k,v in avg_points_by_skill.items()])

    st.subheader("Câu hỏi LLM sinh thêm")
    if gen_questions:
        rows = []
        for i, q in enumerate(gen_questions, 1):
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

    with st.expander("JSON đánh giá"):
        st.code(json.dumps(evaluation, ensure_ascii=False, indent=2))
    with st.expander("JSON câu hỏi sinh"):
        st.code(json.dumps(gen_questions, ensure_ascii=False, indent=2))
