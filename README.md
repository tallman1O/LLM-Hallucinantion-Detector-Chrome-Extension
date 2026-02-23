# LLM Hallucination Detection Engine

This project implements a real-time hallucination verification backend for LLM outputs.

## Features

- Claim-level verification
- FAISS-based semantic retrieval
- Weighted evidence scoring
- Wikipedia + research paper grounding
- Safe claim rewriting
- REST API (FastAPI)

## Usage

Run the API:

```bash
uvicorn src.api:app --reload
```
