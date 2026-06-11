#!/usr/bin/env python3
"""Generate the 1-page solution description PDF for EXACT 2026 submission."""
from fpdf import FPDF

class SolutionPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "EXACT 2026 - Solution Description", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 5, "Team: AI WITH BRO", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def section(self, title):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 9)
        self.cell(5)
        self.multi_cell(0, 4.5, f"- {text}")


pdf = SolutionPDF()
pdf.set_auto_page_break(auto=True, margin=12)
pdf.add_page()

# 1. Datasets
pdf.section("1. Datasets Used")

pdf.set_font("Helvetica", "B", 9)
pdf.cell(55, 5, "Dataset", border=1)
pdf.cell(55, 5, "Source", border=1)
pdf.cell(25, 5, "Samples", border=1)
pdf.cell(55, 5, "Sample Entry", border=1, new_x="LMARGIN", new_y="NEXT")

rows = [
    ("EXACT Type 1 (Logic)", "Official EXACT 2026 release", "808 Qs",
     "Q: Which conclusion follows...? A: A"),
    ("EXACT Type 2 (Physics)", "Official EXACT 2026 release", "7351 Qs",
     "Q: Calculate energy in capacitor... A: 0.045 J"),
    ("Physics RAG Corpus", "Self-curated from Type 2 + textbook formulas", "~500 entries",
     "Ohm's law: V = IR, Coulomb: F = kq1q2/r^2"),
]

pdf.set_font("Helvetica", "", 8)
for name, source, count, sample in rows:
    y0 = pdf.get_y()
    pdf.multi_cell(55, 4, name, border=1, new_x="RIGHT", new_y="TOP", max_line_height=4)
    pdf.set_y(y0)
    pdf.set_x(65)
    pdf.multi_cell(55, 4, source, border=1, new_x="RIGHT", new_y="TOP", max_line_height=4)
    pdf.set_y(y0)
    pdf.set_x(120)
    pdf.multi_cell(25, 4, count, border=1, new_x="RIGHT", new_y="TOP", max_line_height=4)
    pdf.set_y(y0)
    pdf.set_x(145)
    pdf.multi_cell(55, 4, sample, border=1, new_x="LMARGIN", new_y="NEXT", max_line_height=4)

pdf.ln(2)

# No external or synthetic data
pdf.body_text(
    "No external datasets, crawled data, or synthetic data were used. "
    "All training and retrieval data comes exclusively from the official EXACT 2026 release. "
    "The RAG corpus was curated by extracting canonical formulas from the Type 2 dataset."
)

# 2. Approach
pdf.section("2. Approach and Method")
pdf.body_text(
    "Our system is a LangGraph-based agentic pipeline with two branches routed by the 'type' field:\n\n"
    "Type 1 (Logic): The query and premises are sent to an LLM (Qwen2.5-7B-Instruct) which formalizes "
    "them into Z3/Python code. A sandboxed code executor runs the Z3 solver to derive the answer. "
    "If the solver fails, a retry loop re-formalizes with error feedback. An explanation node "
    "generates the natural-language reasoning. For multiple-choice questions with options A-D, "
    "a direct solver path bypasses Z3 for efficiency. An exact-match retrieval layer checks "
    "against the released dataset first.\n\n"
    "Type 2 (Physics): A deterministic formula baseline handles common patterns (Ohm's law, "
    "Coulomb's law, capacitor energy, etc.) without LLM calls. For complex problems, a RAG "
    "retrieval module fetches relevant formulas, then Qwen2.5-Coder-7B generates SymPy code. "
    "The code is executed in a sandboxed subprocess with memory limits. The explanation node "
    "synthesizes the chain-of-thought reasoning.\n\n"
    "Both branches share: sequential request gating (1 request at a time), 58s budget with "
    "graceful cancellation, and a model supervisor that swaps coder/instruct models on the "
    "same GPU (only one LLM resident at a time)."
)

# 3. Model Size
pdf.section("3. Model Size Calculation")

pdf.set_font("Helvetica", "B", 9)
pdf.cell(60, 5, "Model", border=1)
pdf.cell(30, 5, "Parameters", border=1)
pdf.cell(35, 5, "Quantization", border=1)
pdf.cell(50, 5, "Usage", border=1, new_x="LMARGIN", new_y="NEXT")

models = [
    ("Qwen2.5-Coder-7B-Instruct", "7.6B", "Q4_K_M (GGUF)", "Type 2 code generation"),
    ("Qwen2.5-7B-Instruct", "7.6B", "Q4_K_M (GGUF)", "Type 1 FOL + explanations"),
    ("BAAI/bge-m3", "568M", "FP16", "RAG embedding (not LLM)"),
    ("BAAI/bge-reranker-base", "278M", "FP16", "RAG reranking (not LLM)"),
]

pdf.set_font("Helvetica", "", 8)
for name, params, quant, usage in models:
    pdf.cell(60, 5, name, border=1)
    pdf.cell(30, 5, params, border=1)
    pdf.cell(35, 5, quant, border=1)
    pdf.cell(50, 5, usage, border=1, new_x="LMARGIN", new_y="NEXT")

pdf.ln(2)
pdf.body_text(
    "The LlamaServerSupervisor ensures only ONE LLM is loaded on the GPU at any given moment. "
    "When the pipeline needs the coder model, it unloads the instruct model first, and vice versa. "
    "Therefore, the maximum LLM parameters active at any single moment = 7.6B, which is within "
    "the 8B-class limit per Q3 in the Official Q&A.\n\n"
    "The embedding model (bge-m3, 568M) and reranker (bge-reranker-base, 278M) are non-LLM tools "
    "and do not count toward the 8B limit per the submission guide.\n\n"
    "No MoE models are used. No closed-source or third-party inference APIs are used."
)

# 4. Infrastructure
pdf.section("4. Infrastructure")
pdf.body_text(
    "Self-hosted on Apple Silicon MacBook (M-series). FastAPI serves /predict and proxies "
    "/v1/models from the local llama-server. Public access via ngrok tunnel. "
    "All inference is local; no external API calls during evaluation."
)

output_path = "/Users/lananh/Documents/EXACT-2026/EXACT-2026/reports/solution.pdf"
pdf.output(output_path)
print(f"Generated: {output_path}")
