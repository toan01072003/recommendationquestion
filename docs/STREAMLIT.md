## Deploy Streamlit UI for EduRec

This repo ships a Streamlit UI that calls the FastAPI backend to analyze images and generate practice.

### Option A) Standalone Streamlit (no FastAPI)

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

- Push this repo to GitHub.
- Create a new Streamlit app pointing at `streamlit_app.py`.
- Set `GOOGLE_API_KEY` in Streamlit Cloud secrets so the app can call Gemini directly.
- If you want a separate backend, deploy `app.py` elsewhere and adapt the Streamlit app to call it.

### Notes

- This Streamlit UI is a thin client: it does not perform Gemini calls itself; it forwards images and parameters to the FastAPI server.
- If you need a self-contained Streamlit (no separate backend), adapt `streamlit_app.py` to call `google-generativeai` directly and replicate the prompts used in `app.py`.
