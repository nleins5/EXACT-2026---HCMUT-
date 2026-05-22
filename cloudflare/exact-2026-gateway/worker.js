const JSON_HEADERS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
};

const REQUEST_TIMEOUT_MS = 58_000;

function json(data, init = {}) {
  return new Response(JSON.stringify(data), {
    ...init,
    headers: {
      ...JSON_HEADERS,
      ...(init.headers || {}),
    },
  });
}

function normalizeBaseUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function fallback(reason = "Gateway could not reach the self-hosted EXACT API.") {
  return {
    answer: "Unknown",
    explanation: reason,
    fol: "",
    cot: [],
    premises: [],
    confidence: 0.0,
  };
}

function sanitizePredictResponse(value) {
  if (!value || typeof value !== "object") {
    return fallback("Origin returned an invalid response body.");
  }

  const answer = String(value.answer || "Unknown");
  if (answer.toLowerCase() === "error") {
    return fallback(String(value.explanation || "Origin reported an internal error."));
  }

  const confidence = Number(value.confidence ?? 0);
  return {
    answer,
    explanation: String(value.explanation || ""),
    fol: String(value.fol || ""),
    cot: Array.isArray(value.cot) ? value.cot.map(String) : [],
    premises: Array.isArray(value.premises) ? value.premises.map(String) : [],
    confidence: Number.isFinite(confidence) ? Math.min(1, Math.max(0, confidence)) : 0.0,
  };
}

async function fetchWithTimeout(url, init = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort("timeout"), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function docsPage(env) {
  const originConfigured = Boolean(normalizeBaseUrl(env.ORIGIN_URL));
  const modelConfigured = Boolean(normalizeBaseUrl(env.MODEL_BASE_URL));
  return new Response(
    `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EXACT 2026 Gateway</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.5; color: #172033; }
    code, pre { background: #f3f5f8; border-radius: 6px; padding: 2px 5px; }
    pre { padding: 14px; overflow: auto; }
    .ok { color: #147a3e; }
    .warn { color: #a85b00; }
  </style>
</head>
<body>
  <h1>EXACT 2026 Cloudflare Gateway</h1>
  <p>This Worker exposes the public edge endpoint for the self-hosted EXACT API.</p>
  <ul>
    <li><code>GET /health</code></li>
    <li><code>POST /predict</code></li>
    <li><code>GET /v1/models</code> when <code>MODEL_BASE_URL</code> is configured</li>
  </ul>
  <p>Origin API: <strong class="${originConfigured ? "ok" : "warn"}">${originConfigured ? "configured" : "missing ORIGIN_URL"}</strong></p>
  <p>Model metadata: <strong class="${modelConfigured ? "ok" : "warn"}">${modelConfigured ? "configured" : "missing MODEL_BASE_URL"}</strong></p>
  <pre>curl -X POST "$WORKER_URL/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"question":"R1=30Ω, R2=60Ω parallel. Find R_eq."}'</pre>
</body>
</html>`,
    {
      headers: {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      },
    },
  );
}

async function handleHealth(env) {
  const origin = normalizeBaseUrl(env.ORIGIN_URL);
  if (!origin) {
    return json({
      status: "degraded",
      gateway: "ok",
      origin_configured: false,
      model_metadata_configured: Boolean(normalizeBaseUrl(env.MODEL_BASE_URL)),
    });
  }

  try {
    const response = await fetchWithTimeout(`${origin}/health`, { method: "GET" }, 5_000);
    const body = await response.json().catch(() => ({}));
    return json({
      status: response.ok ? "ok" : "degraded",
      gateway: "ok",
      origin_configured: true,
      origin_status: response.status,
      origin: body,
      model_metadata_configured: Boolean(normalizeBaseUrl(env.MODEL_BASE_URL)),
    });
  } catch (error) {
    return json({
      status: "degraded",
      gateway: "ok",
      origin_configured: true,
      origin_error: String(error && error.message ? error.message : error),
      model_metadata_configured: Boolean(normalizeBaseUrl(env.MODEL_BASE_URL)),
    });
  }
}

async function handlePredict(request, env) {
  if (request.method !== "POST") {
    return json({ error: "Method Not Allowed" }, { status: 405, headers: { allow: "POST" } });
  }

  let payload;
  try {
    payload = await request.json();
  } catch {
    return json(fallback("Invalid JSON request body."));
  }

  if (!payload || typeof payload.question !== "string" || !payload.question.trim()) {
    return json(fallback("Missing required field: question."));
  }

  const origin = normalizeBaseUrl(env.ORIGIN_URL);
  if (!origin) {
    return json(fallback("Cloudflare gateway is deployed, but ORIGIN_URL is not configured."));
  }

  try {
    const response = await fetchWithTimeout(`${origin}/predict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      return json(fallback(`Origin returned HTTP ${response.status}.`));
    }

    const body = await response.json().catch(() => null);
    return json(sanitizePredictResponse(body));
  } catch (error) {
    const reason = error && error.name === "AbortError"
      ? "Request exceeded the gateway timeout budget."
      : `Origin request failed: ${String(error && error.message ? error.message : error)}`;
    return json(fallback(reason));
  }
}

async function handleModels(env) {
  const modelBaseUrl = normalizeBaseUrl(env.MODEL_BASE_URL || env.ORIGIN_URL);
  if (!modelBaseUrl) {
    return json({ error: "MODEL_BASE_URL is not configured." }, { status: 503 });
  }

  try {
    const modelsUrl = modelBaseUrl.endsWith("/v1")
      ? `${modelBaseUrl}/models`
      : `${modelBaseUrl}/v1/models`;
    const response = await fetchWithTimeout(modelsUrl, { method: "GET" }, 10_000);
    return new Response(response.body, {
      status: response.status,
      headers: {
        "content-type": response.headers.get("content-type") || "application/json; charset=utf-8",
        "cache-control": "no-store",
      },
    });
  } catch (error) {
    return json({ error: String(error && error.message ? error.message : error) }, { status: 502 });
  }
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/" || url.pathname === "/docs") {
      return docsPage(env);
    }
    if (url.pathname === "/health") {
      return handleHealth(env);
    }
    if (url.pathname === "/predict") {
      return handlePredict(request, env);
    }
    if (url.pathname === "/v1/models") {
      return handleModels(env);
    }

    return json({ error: "Not Found" }, { status: 404 });
  },
};
