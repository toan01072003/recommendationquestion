## Deploy Streamlit UI for EduRec

This repo ships a Streamlit UI that calls the FastAPI backend to analyze images and generate practice.

### 1) Run the FastAPI backend

```bash
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Set your Gemini key for the backend (FastAPI):

```powershell
$env:GOOGLE_API_KEY = "<your_api_key>"
```

### 2) Run Streamlit locally

```bash
streamlit run streamlit_app.py
```

In the left sidebar, set `API base URL` to your backend address, e.g. `http://localhost:8080`.

### 3) Deploy on Streamlit Cloud

- Push this repo to GitHub.
- Create a new Streamlit app pointing at `streamlit_app.py`.
- In Streamlit Cloud, set a secret `API_BASE_URL` to your public backend URL (for example, a Render/Fly/Heroku/Cloud Run deployment of `app.py`).
- The backend (FastAPI) still needs the `GOOGLE_API_KEY` set on its own host.

### Notes

- This Streamlit UI is a thin client: it does not perform Gemini calls itself; it forwards images and parameters to the FastAPI server.
- If you need a self-contained Streamlit (no separate backend), adapt `streamlit_app.py` to call `google-generativeai` directly and replicate the prompts used in `app.py`.
