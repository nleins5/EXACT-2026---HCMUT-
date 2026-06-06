# EXACT 2026 Data Disclosure Statement
**Team:** AI WITH BRO | **Scope:** Training data, retrieval corpora, preprocessing inputs, and evaluation assets

This statement discloses all external data sources used by the EXACT-2026 pipeline. The system does not use GPT, Claude, Gemini, commercial LLM APIs, or any closed-source model for training, data generation, preprocessing, retrieval, evaluation, or inference. "OpenAI-compatible" in the codebase refers only to the local HTTP serving protocol used by `llama-server`.

## Source Register
| Source | Role in pipeline | Scope used | Processing and controls | Disclosure status |
|---|---|---:|---|---|
| Official EXACT 2026 Dataset | Primary task data for Type 1 logic and Type 2 physics; validation of API schema and answer format. | Logic: 411 records / 808 questions. Physics: organizer-provided text-only CSV. | Parsed deterministically from JSON/CSV. Physics rows whose IDs start with `QA` are filtered out according to the official Q&A ruling. | Fully disclosed as organizer-provided data. |
| FOLIO Dataset | Improves natural-language-to-first-order-logic mapping and Z3 entailment patterns for Type 1 logic. | Approx. 1,082 records after availability and preprocessing filters. | Premises and conclusions are converted into formal logic examples using deterministic Python scripts and open-source symbolic tooling. | Public source: https://huggingface.co/datasets/yale-nlp/FOLIO |
| PhysicsFormulae Repository | Static formula and constants reference for Type 2 physics retrieval. | Approx. 28 verified formula/constant records. | Extracted, normalized, verified, and converted into RAG JSONL records. | Public source: https://github.com/BenjaminTMilnes/PhysicsFormulae |
| Electro Textbook Dataset | Supplementary physics examples for SymPy code-generation and explanation training. | 242 physics problems with solutions and verified SymPy code. | Internally digitized and normalized into supervised examples. Must be excluded unless a complete bibliography is attached. | Conditional disclosure required before final submission. |
| RAG Knowledge Base | Optional runtime retrieval for physics formalization. | Collections: `physics_formulas`, `physics_examples`, only when provisioned. | Must be built from fully disclosed verified records using embeddings/reranking and no closed-source LLM rewriting. The submitted repository does not ship a persisted runtime index; retrieval skips immediately when absent. | Derived corpus must be disclosed before deployment. |

## Closed-Source LLM Attestation
No closed-source or commercial LLM is used at any point in the pipeline. This includes: generating synthetic training data, rewriting datasets, preprocessing examples, building RAG content, running evaluation, or serving predictions. Runtime inference is performed only with self-hosted open-source 7B-class models through `llama.cpp` / `llama-server`.

## Submission Risk Note
The Electro textbook dataset is the only conditional source. If the team cannot provide title, author or publisher, edition/year, URL or acquisition source, and page/section range for every included item, Electro-derived artifacts should be removed from fine-tuning, retrieval indexes, and the final live API package.
