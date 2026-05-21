# EXACT 2026: Data Disclosure Document
**Team:** [Điền tên nhóm của bạn vào đây]

In compliance with the EXACT 2026 guidelines, this document discloses all external datasets, synthetic data generation processes, and knowledge bases used for training, fine-tuning, and retrieval within our system.

---

## 1. Training & Fine-Tuning Datasets

To train our specialized `Qwen2.5-Coder` and `Qwen2.5-Instruct` models, we combined the official EXACT dataset with the following external sources:

### 1.1. FOLIO Dataset
- **Source URL:** [yale-nlp/FOLIO on HuggingFace](https://huggingface.co/datasets/yale-nlp/FOLIO)
- **Size:** ~1,082 records used.
- **Purpose:** Used to improve the model's ability to translate natural language premises into First-Order Logic (FOL) and Z3 syntax. This significantly enhanced performance on Type 1 (Logic) questions.

### 1.2. Custom Electro Textbook Dataset (Internal)
- **Source:** Internally collected and digitized from high school / university electromagnetism textbooks.
- **Size:** 242 physics problems with step-by-step solutions and verified SymPy code.
- **Purpose:** Used as supplementary training data to improve the model's arithmetic reliability, unit conversion accuracy, and SymPy code generation for Type 2 (Physics) problems.

### 1.3. EXACT 2026 Official Dataset
- **Source:** Provided by Organizers.
- **Preprocessing:** We filtered out 401 annotation errors in the Physics dataset (IDs starting with "QA-") as instructed in the Q&A document. The remaining valid samples were converted into SymPy / Z3 training templates.

---

## 2. Knowledge Base for Retrieval (RAG)

Our system uses a Retrieval-Augmented Generation (RAG) module for Type 2 Physics queries. The vector database (Qdrant) is populated with the following data:

### 2.1. PhysicsFormulae
- **Source URL:** [BenjaminTMilnes/PhysicsFormulae on GitHub](https://github.com/BenjaminTMilnes/PhysicsFormulae)
- **Size:** ~28 verified formulas and constants (after parsing).
- **Purpose:** Acts as a highly accurate, static formula sheet. The RAG module retrieves the most relevant formulas to provide as context to the Coder LLM, ensuring the generated SymPy code uses correct equations.

### 2.2. EXACT 2026 Distilled Solutions
- **Source:** Official EXACT 2026 Physics dataset.
- **Purpose:** Used as "Worked Examples" in the RAG context. The system retrieves semantically similar physics problems and their corresponding code implementations to serve as few-shot demonstrations at inference time.

---

## 3. Synthetic Data Generation

We utilized closed-source LLMs **strictly for offline data distillation and preprocessing**, in full compliance with the competition rules (no closed-source models are used at inference time).

### 3.1. Gemini API (Flash Lite)
- **Model Used:** `gemini-2.5-flash-lite` (via API).
- **Volume:** Processed ~1,594 records from the official BTC Physics dataset and our internal Electro dataset.
- **Purpose:** Used offline to distill raw Chain-of-Thought solutions into structured SymPy code templates and to extract standardized physics formulas for our RAG Knowledge Base.
