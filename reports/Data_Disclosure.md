# EXACT 2026 Data Pipeline & Disclosure Statement
**Team:** AI WITH BRO | **Pipeline:** Local ingestion datasets, indexing formats, and self-hosted model details

This document outlines the data sources and ingestion pipelines configured in this repository. All datasets are processed locally using offline python scripts. The environment uses zero commercial API calls or closed-source platforms. The OpenAI-compatible endpoints refer to the API protocol structure running locally via `llama-server`.

## Dataset Registry & Preprocessing
| Source | Pipeline Stage | Schema Details | Processing and Engineering Controls | Compliance |
|---|---|---|---|---|
| EXACT 2026 Dataset | System validation | JSON/CSV task schema | Deterministically parsed by FastAPI handlers; filters out evaluation QA identifiers. | Disclosed |
| FOLIO Dataset | Code-gen fine-tuning | Natural language premises | Converted to formal Z3 variables and SMT assertions using structured parser scripts. | Disclosed |
| PhysicsFormulae | Retrieval indexing | Equation constants | Normalized into JSONL document entries mapping physics variables and constants. | Disclosed |
| Electro Textbook | Code-gen training | Text problems | Distilled into structured code templates and SymPy examples; excluded unless source list is attached. | Disclosed |
| RAG Database | Runtime retrieval | JSONL formula vectors | Built via local sentence-transformers; falls back to raw code-generation if missing. | Disclosed |

## Data Pipeline Controls
The system implements strict guidelines to enforce data isolation and safety:
- **Sanitized Parsing:** Input values are cleaned of escape sequences and script wrappers prior to pipeline execution.
- **Offline Indexing:** Retrieval databases (BM25 indexes and vector databases) are compiled locally on the server during the build phase.
- **No Cloud Dependencies:** Training, dataset indexing, retrieval, and inference are configured to execute in air-gapped environments.
- **Verification Suite:** Evaluates model weights against schema validation files to block any malformed output.
