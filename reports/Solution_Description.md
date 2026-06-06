# EXACT 2026 Technical Solution Description
**Team:** AI WITH BRO | **System:** Local FastAPI service with sandboxed program execution and model orchestration

> The system runs a local FastAPI service that maps educational queries to formal programs. Rather than relying on LLMs for direct calculation, the service serves as an orchestrator: language models generate code, an isolated sandbox executes it with Z3 and SymPy, and the final response schema is validated locally.

## Architecture and Routing
The service exposes a unified `/predict` endpoint. To avoid latency overhead, the request payload is parsed deterministically without an LLM classifier. If the query contains a non-empty `premises-NL` field, the router forwards the state to the Type 1 logic pipeline. If empty, the query is routed to the Type 2 physics pipeline. Any explicit `task_type` or `query_type` metadata is honored as a priority routing override.

## System Execution Pipeline
| Stage | System Implementation | Engineering Control |
|---|---|---|
| Routing | Parse request keys (`premises-NL`, `task_type`) | Deterministic fast routing. |
| RAG | Retrieve reference equations and code templates | Local vector-rerank index; auto-skips on missing db. |
| Generator | Code synthesis (Qwen2.5-Coder-7B) | Local inference server; structured templates. |
| Sandbox | Subprocess script execution (Z3/SymPy) | AST constraint parsing, timeout daemon, isolated runtime. |
| Explainer | Result explanation synthesis (Qwen2.5-7B) | Grounded context extraction; system prompt validation. |

## Sandbox Security and Reliability
The execution sandbox protects the host system and ensures output validation:
- **AST Parsing:** Validates the compiled syntax tree of generated code. Restricts import modules strictly to `z3` and `sympy`, blocking standard runtime libraries (e.g., `os`, `sys`, `socket`).
- **Resource Caps:** Limits memory usage to 256MB and locks execution time to a 20-second hard threshold.
- **Auto-Retry Loop:** Catches execution tracebacks and passes them back to the generator for a single correction try.
- **Budget Handler:** Tracks the global request duration. If elapsed time exceeds 58 seconds, it bypasses subsequent LLM nodes and serializes the raw sandbox trace to compile a compliant fallback response.

## Model Orchestration and Serving
Local inference uses `llama.cpp` and `llama-server`. The custom `LlamaServerSupervisor` manages model transitions:
- **Active Lock:** Ensures only one model runs in GPU memory to prevent VRAM allocation crashes.
- **Auto-Unload:** Releases VRAM back to the system when the engine is idle.
- **Protocol:** Exposes a self-hosted API compatible with standard SDK structures, running completely offline.
