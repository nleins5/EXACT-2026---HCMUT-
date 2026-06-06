# EXACT-2026: Agentic AI for Logic & Physics

This repository implements a production-grade Agentic AI system built on LangGraph, designed to solve two classes of educational problems:
- Type 1: Logic-Based Educational Queries (logical reasoning from premises)
- Type 2: Physics Problems (quantitative physics solving)

As an AI agent, I have structured this system to combine Large Language Models (LLMs) with formal symbolic solvers (Z3 for logic, SymPy for physics) to ensure absolute precision, safety, and compliance with the EXACT 2026 evaluation framework.

---

## Agentic System Architecture

The workflow is modeled as a directed graph using LangGraph. Each request is classified and routed through a series of dedicated agent nodes:

```
Classify (Rule-based or fallback classifier)
  ├── Logic Route:
  │   logic_formalizer (Coder) -> logic_solver (Z3 Sandbox) -> logic_explanation (Instruct)
  │
  └── Physics Route:
      physics_rag -> physics_formalizer (Coder) -> physics_solver (SymPy Sandbox) -> physics_explanation (Instruct)
```

### Node Execution Lifecycle
1. Classification: The request is routed based on explicit metadata tags (task_type, query_type, type) or parsed using premises.
2. Formalization (Coder): A specialized fine-tuned Qwen2.5-Coder model generates executable Python code (using Z3 or SymPy).
3. Symbolic Execution (Solver): Code is executed within a restricted subprocess sandbox that enforces resource constraints (memory, CPU, output limits) and a strict 20-second timeout.
4. Explanation (Instruct): A fine-tuned Qwen2.5-Instruct model synthesizes the final answer and structured chain-of-thought explanation using solver artifacts.
5. Failures and Fallbacks: If a solver fails or throws an exception, the explanation node processes the traceback as context to attempt a recovery. If the overall time budget is exhausted, a deterministic fallback ensures compliance.

---

## Core Features

- Dual-Specialist Models: Leverages two fine-tuned open-source model weights (Qwen2.5-Coder-7B and Qwen2.5-7B-Instruct) for separation of concerns between code generation and language synthesis.
- Single-Resident Memory Management: The LlamaServerSupervisor monitors and swaps GGUF models dynamically in and out of GPU memory using llama-server (llama.cpp) to prevent out-of-memory errors on limited hardware.
- Safe Execution Subprocess: Code execution is sandboxed using AST validation to reject unsafe imports or private attribute access, safeguarding the host environment.
- Hybrid Retrieval: Integrates BM25 and vector search with a reranker for physics reference formula lookup. If the index is missing, retrieval skips gracefully.
- Strict Budget Compliance: Enforces a maximum response latency of 58 seconds to guarantee answers remain under the hard 60-second limit.
- Zero External API Policy: The entire system is self-hosted and offline-capable. No commercial API keys or external services are used.

---

## Project Structure

- config/ - Setting and logging YAML configurations.
- data/ - Collected textbooks, distilled physics formulae, fine-tuning datasets, and evaluation files.
- fine_tune/ - Colab notebooks containing the fine-tuning pipelines.
- models/ - GGUF model downloader and directory.
- reports/ - Official submission reports:
  - Solution_Description.md (PDF)
  - Data_Disclosure.md (PDF)
- scripts/ - Scripts for RAG indexing, fine-tune dataset generation, and PDF rendering.
- src/ - Main source code containing the LangGraph logic, server supervisor, schemas, and utility packages.
- tests/ - Comprehensive test suite (84 automated checks).
- test_pipeline.py - Integration pipeline sanity test.

---

## Getting Started

### 1. Environment Setup

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Model Acquisition

Download the fine-tuned model weights (GGUF format) from the repository:

```bash
cd models
python download_models.py
```

### 3. Engine Setup

Download the pre-compiled binary of llama-server from the official llama.cpp release page and place it inside the `bin/llama-cpp/` directory.

### 4. Build Reference Database

Generate the persistent database index for the physics retrieval system:

```bash
python -m scripts.rag.build_physics_index
```

### 5. Validation

Execute the test suite to verify code compliance and correct pipeline routing:

```bash
python -m pytest
```

Execute the end-to-end integration pipeline with sample questions:

```bash
python test_pipeline.py
```

---

## Evaluation Compliance Highlights

- Models Endpoint: The /v1/models route exposes self-hosted metadata for transparency.
- Chain-of-Thought Integration: Solver execution steps are formatted directly into the cot list to verify logical derivation.
- Input Normalization: Resolves all custom evaluator request fields (e.g., premises-NL, task_type aliases) to prevent deserialization errors.

---

## License

This project is built for the EXACT 2026 Competition at IEEE IJCNN 2026.
