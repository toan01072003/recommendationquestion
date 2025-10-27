## Deploy Streamlit UI for EduRec

This repo ships two Streamlit UIs:
1) A chatbot-style standalone app that talks to Gemini directly (no FastAPI).
2) A dashboard-style app (also standalone) for power users.

### Option A) Chatbot (Standalone, recommended)

Run the conversational UI that uploads the exam + submission and returns the breakdown by Bài/ý, a gradebook, hints, and practice questions.

```bash
pip install -r requirements.txt
streamlit run streamlit_chat.py
```

Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in your environment or Streamlit secrets.

Notes:
- Geometry support: when the topic is hình học, the model returns a small inline SVG in `diagram_svg`; the chatbot renders it directly. If a diagram cannot be drawn, it falls back to a short `diagram_description`.

### Option B) Dashboard (Standalone)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Set `GOOGLE_API_KEY` in your environment (or Streamlit secrets) so the UI can call Gemini directly. This mode runs both "server logic" and UI inside Streamlit.

### Option B) Run the FastAPI backend

```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Set your Gemini key for the backend (FastAPI):

```powershell
$env:GOOGLE_API_KEY = "<your_api_key>"
```

### Run Streamlit as an API client (to FastAPI)

```bash
streamlit run streamlit_app.py
```

In older versions, `streamlit_app.py` could call the FastAPI API. The current app defaults to Standalone mode. If you want the client mode, revert to an earlier commit or adapt the file to call `POST /agent/suggest-questions`.

### Deploy on Streamlit Cloud

Common steps:

- Push this repo to GitHub.
- Create a new Streamlit app and pick one entry script:
  - `streamlit_chat.py` for the chatbot experience
  - `streamlit_app.py` for the dashboard view
- In Streamlit Cloud, set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in Secrets.
- Optionally set `GEMINI_MODEL` (default `gemini-1.5-flash`).

### Notes

Both Streamlit apps are self‑contained and call `google-generativeai` directly, mirroring the prompts in `app.py`. You do not need to run FastAPI unless you want the HTTP API.
