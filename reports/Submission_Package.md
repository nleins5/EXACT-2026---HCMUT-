# EXACT 2026 Submission Package

**Team:** AI WITH BRO

## 1. Public API Endpoint

**Public URL:** `https://cupping-frisbee-bottom.ngrok-free.dev`

The endpoint must expose:
- `POST /predict`
- `GET /health`
- `GET /v1/models`

Do not submit a placeholder URL. Replace this field only after the FastAPI service or Cloudflare gateway is publicly reachable and returns the official JSON schema.

Expected `/predict` response contract:
- Mandatory: `answer`, `explanation`
- Encouraged for reasoning depth: `fol`, `cot`, `premises`, `confidence`

## 2. One-Page Solution Description PDF

File:
`reports/EXACT2026_Solution_Description_AI_WITH_BRO.pdf`

## 3. Data Disclosure Document PDF

File:
`reports/EXACT2026_Data_Disclosure_AI_WITH_BRO.pdf`

## Final Pre-Submission Checks

- No GPT, Claude, Gemini, or other closed-source/commercial LLM API is configured.
- All external datasets are disclosed.
- If Electro textbook data is used, attach the full textbook source list. If not available, remove Electro-derived training/RAG artifacts before submission.
- `/predict` returns HTTP 200 with valid JSON for both Type 1 and Type 2 requests.
- Every `/predict` response includes non-empty `answer` and `explanation` for P1/P2 scoring.
- Optional P3 evidence fields are normalized: `fol` string, `cot` list, `premises` list, and `confidence` in `[0, 1]`.
- Request budget stays below 60 seconds.
- `/v1/models` exposes auditable self-hosted model metadata.
