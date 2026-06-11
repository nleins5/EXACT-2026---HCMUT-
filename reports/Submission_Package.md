# EXACT 2026 Deployment & Submission Package
**Team:** AI WITH BRO | **Infrastructure:** Self-hosted FastAPI service and tunnel config

This package lists the deployment details and validation steps for the EXACT 2026 system.

## 1. Public API Configuration
- **Base Endpoint URL:** `https://cupping-frisbee-bottom.ngrok-free.dev`
- **Port:** `8000` (FastAPI backend)

### API Endpoint Specification
- `POST /predict`: Unified routing node. Receives the query payload and processes it through the LangGraph pipeline.
- `GET /v1/models`: Returns local model metadata to confirm self-hosted weights.
- `GET /health`: Returns service running status, model lock state, and active process count.

### Input JSON Schema (Unified — per Submission Guide §3)
```json
{
  "query_id": "T1_0001",
  "type": "type1",
  "query": "Is Student A eligible for graduation?",
  "premises": ["A student with >= 120 credits is eligible.", "Student A has 118 credits."],
  "options": ["Yes", "No", "Uncertain"]
}
```

### Response JSON Schema (List-wrapped — per Submission Guide §4)
```json
[
  {
    "query_id": "T1_0001",
    "answer": "No",
    "unit": "",
    "explanation": "Student A has 118 credits, below the 120 required.",
    "premises_used": [0, 1],
    "reasoning": {
      "type": "fol",
      "steps": ["118 < 120", "not Eligible(StudentA)"]
    }
  }
]
```

## 2. Solution Artifacts
- **Technical Description:** `reports/EXACT2026_Solution_Description_AI_WITH_BRO.pdf`
- **Data Disclosure Details:** `reports/EXACT2026_Data_Disclosure_AI_WITH_BRO.pdf`
- **Submission Package Archive:** `AI_WITH_BRO.zip`
  - `solution.pdf`
  - `source_code.zip`
  - `urls.txt`
  - `notation_mapping.csv`

## 3. Engineering Pre-Flight Check
- Run local unit tests (`pytest`) to verify AST script parser and sandbox constraints.
- Verify `llama-server` process supervisor is responsive on target ports.
- Test endpoint latency: ensure response generation time stays below the 60-second timeout.
- Verify `/predict` returns a JSON list (even for single queries).
- Verify `query_id` in response matches the input `query_id`.
- For choice questions (options non-empty): verify `answer` matches exactly one option.
- For Type 1: verify `premises_used` contains 0-based indices.
- For Type 2: verify `unit` field is in ASCII (e.g., A, V, ohm).
- Verify the system does not attempt external API connections during inference.
- Verify `/v1/models` endpoint is reachable and reports declared models.
- Register one-hour grading slot via the form.
