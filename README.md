# EXACT-2026: Hybrid Neural-Symbolic Reasoning Engine

This repository contains the local FastAPI implementation for the EXACT 2026 Competition. The system integrates local autoregressive model inference with isolated symbolic solvers (Z3 SMT and SymPy computer algebra) using a state-machine architecture managed by LangGraph.

---

## System Architecture

```
                             [FastAPI Request]
                                     │
                             [Schema Router]
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
         [Type 1: Logic Query]               [Type 2: Physics Query]
                   │                                   │
                   ▼                                   ▼
          [Code Generator]                    [Local RAG Engine]
                   │                                   │
                   ▼                                   ▼
            [Z3 Sandbox]                        [Code Generator]
                   │                                   │
                   ▼                                   ▼
         [Explanation Node]                     [SymPy Sandbox]
                   │                                   │
                   ▼                                   ▼
                   └─────────────────┬─────────────────┘
                                     ▼
                            [Schema Sanitizer]
                                     │
                             [JSON Response]
```

### 1. Unified Router
Incoming requests follow the official unified payload: `query_id`, `type`, `query`, `premises`, and `options`.
- `type1` routes to the logic pipeline and returns 0-based `premises_used`.
- `type2` routes to the physics pipeline and returns a numeric `answer` plus ASCII `unit`.
- Legacy aliases remain accepted for local compatibility.

### 2. Isolated Execution Sandbox
Generated code scripts are run inside a sandboxed subprocess to safeguard the host operating system:
- **AST Inspector:** Verifies the compiled Abstract Syntax Tree of the generated python code before execution, allowing only `z3` and `sympy` imports and blocking standard system libraries (e.g., `os`, `sys`, `subprocess`, `socket`).
- **Resource Limitations:** Limits solver subprocess memory to 1536MB, locks runtime to a 20-second timeout, and restricts stdout output length.
- **Traceback Recovery:** If the sandbox execution fails, the traceback logs are extracted and passed back to the generator for a single correction loop.

### 3. VRAM Supervisor (`LlamaServerSupervisor`)
To execute inference locally on consumer GPUs, the system handles model weights dynamically:
- Manages local `llama-server` process states.
- Swaps Qwen2.5-Coder and Qwen2.5-Instruct GGUF files in and out of VRAM using file-locking threads to prevent out-of-memory crashes.
- Unloads models to system memory when the service is idle.

---

## Repository Structure

- `config/` - YAML settings for timeouts, logger configs, and model port bindings.
- `data/` - Training samples, extracted textbook sources, and validation datasets.
- `fine_tune/` - Training configurations for Qwen model fine-tuning.
- `models/` - Downloader script and local storage for GGUF model packages.
- `reports/` - Submission documents, technical briefings, and package files.
- `scripts/` - Ingestion, physics vector indexing, and PDF building scripts.
- `src/` - Core FastAPI backend, LangGraph state nodes, sandbox environment, and supervisor code.
- `tests/` - Active test suite verifying endpoints, formula baselines, sandbox parsing, and recovery logic.
- `test_pipeline.py` - Integration script running sample requests through the local graph.

---

## Deployment & Setup

### 1. Environment Configuration
Build a Python 3.10+ virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch Model Binaries
Retrieve the GGUF model files:
```bash
cd models
python3 download_models.py
cd ..
```

### 3. Compiling the Inference Engine
Download or compile the `llama-server` binary for your environment and place it under:
`bin/llama-cpp/llama-server`

### 4. Optional Local Retrieval Index
No Physics vector index is shipped or active by default. To build one from a fully disclosed verified corpus:
```bash
python3 -m scripts.rag.build_physics_index --input data/distilled/physics_kb.verified.jsonl --rebuild
```

### 5. Running the Test Suite
Verify endpoint schemas, sandbox security restrictions, and node transitions:
```bash
python3 -m pytest
```

Rebuild the final submission archive:
```bash
python3 scripts/generate_solution_pdf.py
python3 scripts/build_submission_package.py
```

Test end-to-end question processing:
```bash
python3 test_pipeline.py
```
