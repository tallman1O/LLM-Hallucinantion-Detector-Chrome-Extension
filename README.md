# LLM Hallucination Detection Engine

This project provides a FastAPI backend for claim-level hallucination verification and a Chrome extension frontend that analyzes LLM responses on ChatGPT, Claude, and Gemini.

## Features

- Claim-level verification
- Three-way claim classification: `SUPPORTED`, `CONTRADICTED`, `UNVERIFIABLE`
- FAISS-based semantic retrieval
- Weighted evidence scoring
- Wikipedia + research paper grounding
- Safe claim rewriting
- REST API (FastAPI)
- Chrome extension that sends assistant replies to the local backend

## Project Structure

- `src/`: Python backend and verification pipeline
- `LLM-Hallucinantion-Detector-Chrome-Extension/`: Chrome extension frontend

## Run The Backend

Start the API from the repo root:

```bash
uvicorn src.api:app --reload
```

Useful health check:

```bash
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Run The Chrome Extension

Build the extension from its folder:

```bash
cd LLM-Hallucinantion-Detector-Chrome-Extension
npm install
npm run build
```

If you use `bun`, this works too:

```bash
bun install
bun run build
```

Then load the unpacked extension in Chrome:

1. Open `chrome://extensions`
2. Enable Developer Mode
3. Click `Load unpacked`
4. Select `LLM-Hallucinantion-Detector-Chrome-Extension/dist`

## How It Works

1. The content script watches the latest assistant reply on supported LLM sites.
2. The extension sends that text to `POST /verify` on your local FastAPI backend.
3. The backend splits the reply into claims, retrieves evidence, and classifies each claim.
4. The extension displays claim status, confidence, and top evidence in the floating panel.

## Claim Labels

- `SUPPORTED`: strong retrieved evidence supports the claim
- `CONTRADICTED`: evidence sources strongly disagree with the claim
- `UNVERIFIABLE`: evidence is weak, mixed, missing, or not strong enough to support the claim

## Notes

- The backend log lines `POST /verify HTTP/1.1 200 OK` confirm the extension is using the backend successfully.
- If you see `LLaMA model not found. Rewrite disabled.`, the core verification still works, but optional rewrite/explanation features are unavailable until the model file is added.
