# EXACT 2026 Deployment & Submission Package
**Team:** AI WITH BRO | **Infrastructure:** Self-hosted FastAPI service and tunnel config

This package lists the deployment details and validation steps for the EXACT 2026 system.

## 1. Public API Configuration
- **Base Endpoint URL:** `https://cupping-frisbee-bottom.ngrok-free.dev`
- **Port:** `8000` (FastAPI backend)

### API Endpoint Specification
- `POST /predict`: Unified routing node. Receives the query payload and processes it through the LangGraph pipeline.
- `GET /health`: Returns service running status, model lock state, and active process count.
- `GET /v1/models`: Returns local model metadata to confirm self-hosted weights.

### Response JSON Schema Validation
```json
{
  "answer": "string",
  "explanation": "string",
  "fol": "string",
  "cot": ["string"],
  "premises": ["string"],
  "confidence": 0.0
}
```

## 2. Solution Artifacts
- **Technical Description:** `reports/EXACT2026_Solution_Description_AI_WITH_BRO.pdf`
- **Data Disclosure Details:** `reports/EXACT2026_Data_Disclosure_AI_WITH_BRO.pdf`
- **Submission Package Archive:** `reports/EXACT2026_Submission_Package_AI_WITH_BRO.zip` (contains both PDF files and this markdown document).

## 3. Engineering Pre-Flight Check
- Run local unit tests (`pytest`) to verify AST script parser and sandbox constraints.
- Verify `llama-server` process supervisor is responsive on target ports.
- Test endpoint latency: ensure response generation time stays below the 58-second threshold.
- Check confidence normalization: confirm values are floats clamped between `[0, 1]`.
- Verify the system does not attempt external API connections during inference.
