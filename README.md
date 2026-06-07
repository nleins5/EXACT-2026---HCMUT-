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
Incoming requests are parsed dynamically by FastAPI payload keys. The router uses the payload format to direct the state machine:
- Task 1 payloads use `question` plus a non-empty `premises-NL` list and route to the logic pipeline.
- Task 2 payloads use `question` without `premises-NL` and route to the physics pipeline.
- Explicit query type overrides (`task_type`, `query_type`) bypass default classification rules.

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

### 4. Build Local Retrieval Indexes
Compile the local formulas retrieval index:
```bash
python3 -m scripts.rag.build_physics_index
```

### 5. Running the Test Suite
Verify endpoint schemas, sandbox security restrictions, and node transitions:
```bash
python3 -m pytest
```

Test end-to-end question processing:
```bash
python3 test_pipeline.py
```
