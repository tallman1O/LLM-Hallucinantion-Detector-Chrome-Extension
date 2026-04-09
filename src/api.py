from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.api_adapter import run_api

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"

app = FastAPI(
    title="LLM Hallucination Verifier",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerifyRequest(BaseModel):
    text: str

@app.post("/verify")
def verify(req: VerifyRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Prompt text is required.")

    return run_api(req.text)

@app.get("/")
def frontend():
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")

    return FileResponse(FRONTEND_FILE)

@app.get("/health")
def health():
    return {"status": "ok"}
