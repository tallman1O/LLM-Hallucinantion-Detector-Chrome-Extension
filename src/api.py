from fastapi import FastAPI
from pydantic import BaseModel
from src.api_adapter import run_api

app = FastAPI(
    title="LLM Hallucination Verifier",
    version="1.0"
)

class VerifyRequest(BaseModel):
    text: str

@app.post("/verify")
def verify(req: VerifyRequest):
    return run_api(req.text)

@app.get("/")
def health():
    return {"status": "ok"}