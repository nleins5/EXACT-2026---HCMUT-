# EXACT 2026 Data Disclosure Document
**Team:** AI WITH BRO | **Scope:** training, fine-tuning, retrieval, evaluation, and deployment data

## Complete Data Registry
| Dataset / model source | Size used | Purpose | Source and processing |
|---|---:|---|---|
| EXACT 2026 official release (May 15) | Type 1: 411 records / 808 questions; Type 2: 1,352 problems | Fine-tuning, validation, format definition, exact-match retrieval for released Type 1 examples | [ura.hcmut.edu.vn/exact](https://ura.hcmut.edu.vn/exact). Parsed locally; the updated release already removes invalid Type 2 QA rows. |
| FOLIO | 1,204 train/validation examples loaded; 199 executable examples retained for coder tuning | Logic NL-to-FOL/Z3 and explanation tuning | [huggingface.co/datasets/yale-nlp/FOLIO](https://huggingface.co/datasets/yale-nlp/FOLIO). Converted locally to executable Z3 scripts; invalid conversions are discarded. |
| Internal electro textbook-derived corpus | 242 records | Additional Physics/SymPy and explanation tuning | Manually collected text problems. Per-record source URLs were not retained; this provenance limitation is disclosed. Converted locally to SymPy templates and verified before use. |
| Fine-tuned Qwen GGUF models | Qwen2.5-Coder-7B and Qwen2.5-7B-Instruct, Q4_K_M | Runtime code generation and Logic MCQ answering | Public weights: [huggingface.co/HoangKhangHCMUS/exact-2026](https://huggingface.co/HoangKhangHCMUS/exact-2026). Self-hosted with `llama-server`. |

## Generated Artifacts And Runtime Retrieval
- Fine-tuning outputs contain 1,546 coder samples and 2,798 instruct samples. Scripts and aggregate statistics are included in the repository.
- No closed-source model generated synthetic data, and no larger teacher-model distillation was used.
- No external RAG corpus or persisted vector database is active. Runtime retrieval is limited to transparent exact-match lookup over the disclosed released Type 1 corpus; unseen questions continue through the reasoning pipeline.
- Runtime inference does not call proprietary LLM APIs or third-party inference providers. Z3, SymPy, deterministic formula rules, and the two self-hosted Qwen models are used.

## Controls
All preprocessing runs locally. Generated Python is AST-validated, executed in an isolated subprocess with time/output/resource limits, and covered by automated tests. Model metadata is auditable through `/v1/models`.
