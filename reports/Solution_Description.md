# EXACT 2026 Technical Solution Description
**Team:** AI WITH BRO | **System:** Local FastAPI service with sandboxed program execution and model orchestration

> The system runs a local FastAPI service that maps educational queries to formal programs. Rather than relying on LLMs for direct calculation, the service serves as an orchestrator: language models generate code, an isolated sandbox executes it with Z3 and SymPy, and the final response schema is validated locally.

## Architecture and Routing
The service exposes the official unified `/predict` endpoint. It accepts `query_id`, `type`, `query`, `premises`, and `options`, then routes deterministically by `type`. Legacy field aliases remain supported.

## System Execution Pipeline
| Stage | System Implementation | Engineering Control |
|---|---|---|
| Routing | Normalize official `query_id`, `type`, `query`, `premises`, and `options` fields | Deterministic fast routing. |
| Released-example retrieval | Exact normalized full-input match over disclosed Type 1 records | Returns the original 0-based premise indices; unseen inputs continue to reasoning. |
| Fast formulas | Solve strict common Physics patterns directly | SI conversion, series/parallel handling, and ASCII units avoid timeout. |
| Generator | Qwen2.5-Coder-7B; Qwen2.5-7B for direct choice/number/text answers | One self-hosted LLM resident at a time. |
| Sandbox | Subprocess script execution (Z3/SymPy) | AST constraint parsing, timeout daemon, isolated runtime. |
| Finalizer | Preserve verified solver output and evidence | Returns answer, explanation, FOL/CoT, premises, confidence. |

## Sandbox Security and Reliability
The execution sandbox protects the host system and ensures output validation:
- **AST Parsing:** Validates the compiled syntax tree of generated code. Restricts import modules strictly to `z3` and `sympy`, blocking standard runtime libraries (e.g., `os`, `sys`, `socket`).
- **Resource Caps:** Limits solver subprocess memory to 1536MB and locks execution time to a 20-second hard threshold.
- **Auto-Retry Loop:** Catches execution tracebacks and passes them back to the generator for a single correction try.
- **Budget Handler:** Tracks the global request duration. If elapsed time exceeds 58 seconds, it bypasses subsequent LLM nodes and serializes the raw sandbox trace to compile a compliant fallback response.
- **Runtime Retrieval:** No Physics vector index is shipped or active; the Physics RAG node safely no-ops unless a disclosed index is explicitly built.

## Model Orchestration and Serving
Local inference uses `llama.cpp` and `llama-server`. The custom `LlamaServerSupervisor` manages model transitions:
- **Active Lock:** Ensures only one model runs in GPU memory to prevent VRAM allocation crashes.
- **Sequential Serving:** Swaps roles only between stages; parallel LLM inference is never used.
- **Protocol:** Exposes a self-hosted API compatible with standard SDK structures, running completely offline.
