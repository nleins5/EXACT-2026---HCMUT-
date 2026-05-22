# EXACT 2026 Cloudflare Gateway

Cloudflare Workers cannot run this repository's FastAPI/Python/LangGraph stack directly. This Worker is a public edge gateway for the self-hosted EXACT API required by the PDFs:

- `POST /predict` accepts the unified EXACT input stream.
- Response always follows the EXACT JSON shape, even on timeout or origin failure.
- `GET /health` reports gateway and origin status.
- `GET /v1/models` proxies model metadata when `MODEL_BASE_URL` points at a public vLLM/llama-server origin.

## Configure

Set the Worker variables after deploying:

```bash
cd cloudflare/exact-2026-gateway
npx wrangler deploy
npx wrangler secret put ORIGIN_URL
npx wrangler secret put MODEL_BASE_URL
```

Use public HTTPS URLs:

- `ORIGIN_URL`: public FastAPI base URL, for example `https://exact-api.example.com`
- `MODEL_BASE_URL`: public OpenAI-compatible model server base URL, for example `https://exact-api.example.com` if it exposes `/v1/models`, or another controlled self-hosted endpoint.

Do not point this gateway to third-party inference APIs. The Q&A requires the LLM to be self-hosted and verifiable.

## Local Test

```bash
curl "$WORKER_URL/health"
curl -X POST "$WORKER_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"question":"R1=30Ω, R2=60Ω parallel. Find R_eq."}'
```

