# EXACT 2026: Solution Description
**Team:** [Điền tên nhóm của bạn vào đây]

## 1. Approach Overview
Our solution is an **Agentic AI system** built on **LangGraph**, designed to explicitly decouple logical reasoning/mathematical computation from natural language explanation. Instead of relying on a single LLM to perform both reasoning and text generation, we employ a dual-specialist LLM architecture augmented with formal symbolic solvers.

The pipeline processes queries through a unified stream, classifying them into Type 1 (Logic) or Type 2 (Physics) based on the presence of natural-language premises. 

## 2. System Architecture
Our pipeline consists of the following sequential stages:
1. **Classifier (Rule-based):** Routes the query based on input schema (Logic vs. Physics).
2. **Retrieval-Augmented Generation (Physics only):** Uses a Hybrid Search (BM25 + Vector) with a reranker to retrieve relevant physics formulas and worked examples from a curated knowledge base.
3. **Formalizer:** A specialized Coder LLM translates the natural language query (and retrieved context) into executable code (Z3 for Logic, SymPy for Physics).
4. **Symbolic Solver:** The generated code is executed in an isolated Python subprocess with a strict 30-second timeout. This ensures the reasoning is mathematically precise and provably correct.
5. **Explanation Generator:** A specialized Instruct LLM takes the original problem, the generated code, and the execution output to synthesize a structured JSON response containing the final answer, chain-of-thought, and formal logic derivation.

*Error-Branching Mechanism:* If the symbolic solver fails or crashes, the Explanation LLM utilizes an "Error Branch" prompt. It treats the failed code as a hint to deduce the most logical answer, ensuring a robust fallback response.

## 3. Models Used
To comply with the ≤ 8B parameters rule and optimize VRAM via Single-Resident Swap (`llama-server`), we fine-tuned two specialized models:
- **`Qwen2.5-Coder-7B` (Fine-tuned):** Optimized specifically for translating natural language premises into Z3 syntax and Physics problems into SymPy equations.
- **`Qwen2.5-7B-Instruct` (Fine-tuned):** Optimized for generating human-readable explanations, Chain-of-Thought, and structured JSON outputs based on solver results.

## 4. Tools & External Components
- **Orchestration:** LangGraph, LangChain
- **Serving:** `llama.cpp` (OpenAI-compatible server)
- **Solvers:** `z3-solver` (Theorem Prover), `sympy` (Symbolic Mathematics)
- **Retrieval:** LlamaIndex, Qdrant (Vector DB)
- **Embeddings & Reranking:** `BAAI/bge-m3` (Embeddings), `BAAI/bge-reranker-base` (Cross-encoder reranking)
