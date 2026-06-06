# EXACT-2026: Structured Neural-Symbolic Reasoning Engine for Logic and Quantitative Physics

This repository presents a production-grade Neural-Symbolic reasoning system designed for the EXACT 2026 Competition (IEEE IJCNN 2026). The system addresses two distinct classes of academic problems:
- **Type 1 (Logic-Based Educational Queries):** First-order logic reasoning, inference, and validity checking from premises.
- **Type 2 (Quantitative Physics Problems):** Numerical and symbolic physics solving requiring formula derivation and computation.

Rather than relying on unconstrained text generation from large language models, this system separates code generation (neural program synthesis) from execution and reasoning verification (symbolic solving). It integrates local fine-tuned language models with formal mathematical/logical engines (Z3 SMT Solver and SymPy Computer Algebra System) using a deterministic state-transition graph built on LangGraph.

---

## Architectural Taxonomy

```
                                 [Input Query]
                                       │
                         ┌─────────────┴─────────────┐
                         │   Deterministic Router    │
                         └─────────────┬─────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼ Task Classification                 ▼
          [Type 1: Logic Query]                 [Type 2: Physics Query]
                    │                                     │
                    ▼                                     ▼
          ┌──────────────────┐                  ┌──────────────────┐
          │ logic_formalizer │                  │   physics_rag    │
          │  (Program Gen)   │                  └─────────┬────────┘
          └─────────┬────────┘                            │ Formula Retrieve
                    │                                     ▼
                    │                           ┌──────────────────┐
                    │                           │physics_formalizer│
                    │                           │  (Program Gen)   │
                    │                           └─────────┬────────┘
                    ▼                                     ▼
          ┌──────────────────┐                  ┌──────────────────┐
          │   logic_solver   │                  │  physics_solver  │
          │   (Z3 Sandbox)   │                  │ (SymPy Sandbox)  │
          └─────────┬────────┘                  └─────────┬────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼ Symbolic Trace
                            ┌─────────────────────┐
                            │ logic/phys_explain  │
                            │ (Response Synth)    │
                            └──────────┬──────────┘
                                       ▼ Output Schema
                             [Verified API Output]
```

### 1. Deterministic Router & Task Classification
To eliminate the latency and accuracy overhead of LLM-based query classification, routing is performed deterministically based on the incoming schema payload. If `premises-NL` is populated, the query is routed to the **Type 1 Logic** pipeline. Otherwise, it is routed to the **Type 2 Physics** pipeline. Explicit metadata signals (`task_type`, `query_type`) are also honored immediately to ensure perfect alignment with evaluator test streams.

### 2. Neural Program Synthesis (Formalization)
For each task type, a local, fine-tuned `Qwen2.5-Coder-7B-Instruct` model serves as a formalizer. Instead of solving the math or logic directly, the model is prompted to synthesize a complete Python script:
- **Type 1:** Generates formal logical declarations using the Z3 SMT solver API to evaluate consistency (SAT/UNSAT) or entailment.
- **Type 2:** Synthesizes symbolic math equations using SymPy to isolate target variables and perform unit conversions.

### 3. Isolated Symbolic Sandbox (Solver)
The generated Python code is written to disk and executed in a restricted subprocess environment. The execution loop enforces:
- **Abstract Syntax Tree (AST) Inspection:** Rejects import statements outside of a strict allowlist (e.g., preventing access to `os`, `sys`, or network calls).
- **Resource Constraints:** Enforces strict execution timeouts (20 seconds), memory allocation limits, and output length truncation.
- **Execution Feedback Loop:** If execution fails with a runtime exception, the traceback is fed back to the formalizer node for a single self-correction attempt.

### 4. Grounded Explanation Synthesis (Explainer)
Once symbolic execution completes, the output trace (variable values, solver model assertions, and constraints) is extracted. A fine-tuned `Qwen2.5-7B-Instruct` model processes the trace alongside the original query. Its goal is to translate the symbolic solution back into a natural language explanation, guaranteeing that the final output is factually grounded in the execution trace.

---

## Memory & Runtime Infrastructure

### 1. Dynamic Server Orchestration (`LlamaServerSupervisor`)
To execute the neural-symbolic pipeline on consumer-grade hardware with limited VRAM, the system implements a resident memory manager. The `LlamaServerSupervisor` manages instances of `llama-server` (from the `llama.cpp` project) and handles dynamic loading and unloading of model weights in GGUF format:
- **Idle Timeout:** Automatically unloads models when inactive to free up system VRAM.
- **Preemptive Swapping:** If the coder model is loaded and the pipeline requests an explanation, the supervisor unloads the coder, loads the instruct model, and routes the request.
- **Concurrency Locks:** Prevents race conditions during model transition states.

### 2. Physical Knowledge Retrieval (RAG)
For quantitative physics problems, the system uses a hybrid retrieval pipeline combining BM25 and vector search (using local sentence-transformers) against a curated database of normalized physics formulas and solved examples.
- **Dynamic Fallback:** If the index is not built or retrieval fails, the pipeline automatically skips the RAG step to prevent pipeline blocking.
- **Data Compliance:** No external APIs or cloud models are used for vectorization or indexing.

### 3. Hard Latency Budgeting
The API enforces a strict budget control system. The absolute evaluator timeout is 60 seconds. The API monitors elapsed time at each node transition:
- **Max Node Budget:** Total execution is capped at 58 seconds.
- **Fallback Activation:** If the time remaining is insufficient to load the explanation model or run another inference, the pipeline immediately triggers a fast solver-fallback. This constructs a valid JSON response from the raw symbolic trace, avoiding evaluator timeouts.

---

## Technical Specifications

### Project Structure
- `config/` - YAML files for logging, system timeouts, and port allocations.
- `data/` - Training data, formula databases, and processed evaluation datasets.
- `fine_tune/` - Jupyter/Colab notebooks documenting the supervised fine-tuning configurations.
- `models/` - Downloader script and storage directory for the GGUF model binaries.
- `reports/` - Competition artifacts, technical solution summaries, and data disclosures.
- `scripts/` - RAG index building, evaluation utilities, and PDF generation tools.
- `src/` - Core source modules (LangGraph topology, sandboxed execution, server supervisors).
- `tests/` - Active test suite containing 84 unit and integration tests.
- `test_pipeline.py` - Integration validation script for end-to-end execution.

---

## Getting Started

### 1. Virtual Environment & Package Installation
Initialize a clean Python virtual environment and install all packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Weight Retrieval
Acquire the custom fine-tuned model files (in quantized GGUF format):
```bash
cd models
python3 download_models.py
cd ..
```

### 3. Server Engine Assembly
Download or compile the `llama-server` binary for your target architecture from the official `llama.cpp` releases and place it inside:
`bin/llama-cpp/llama-server`

### 4. Build Knowledge Index
Compile the physics formula reference index:
```bash
python3 -m scripts.rag.build_physics_index
```

### 5. Verification
Run the unit test suite to verify module configurations and sandbox safety constraints:
```bash
python3 -m pytest
```

Execute an end-to-end run through the entire LangGraph pipeline:
```bash
python3 test_pipeline.py
```

---

## Competition Alignment

- **P1 Correctness:** Answer extraction is anchored to mathematical output from SymPy or logical consistency checks from Z3, removing model hallucination.
- **P2 Explanation Quality:** Explanations map the logical premise chain or outline the formulas and steps utilized in computation.
- **P3 Reasoning Depth:** The output returns structured chains of thought (`cot`), formalizations (`fol`), extracted premises, and confidence estimates calibrated against solver success.
