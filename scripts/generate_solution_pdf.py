#!/usr/bin/env python3
"""Generate the 1-page solution description PDF for EXACT 2026 submission."""
from pathlib import Path

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

# Column widths: Dataset=45, Source=50, Samples=22, Sample Entry=73 (total=190)
COL_W = [45, 50, 22, 73]

pdf.set_font("Helvetica", "B", 8)
headers = ["Dataset", "Source", "Samples", "Sample Entry"]
for i, h in enumerate(headers):
    pdf.cell(COL_W[i], 6, h, border=1, align="C")
pdf.ln()

rows = [
    ("EXACT Type 1 (Logic)", "Official EXACT 2026 release", "808 Qs",
     "Q: Which conclusion follows...? A: A"),
    ("EXACT Type 2 (Physics)", "Official EXACT 2026 release", "1,352 Qs",
     "Q: Calculate energy in capacitor... A: 0.045 J"),
    ("FOLIO", "Yale NLP / Hugging Face", "1,204 loaded\n199 coder",
     "NL logic premise and conclusion pairs"),
    ("Internal electro corpus", "Manually collected textbook problems", "242",
     "Circuit/electrostatics problem + verified answer"),
]

ROW_H = 10  # uniform height for all rows
pdf.set_font("Helvetica", "", 8)
for cells in rows:
    x0, y0 = pdf.get_x(), pdf.get_y()
    for i, text in enumerate(cells):
        pdf.set_xy(x0 + sum(COL_W[:i]), y0)
        pdf.multi_cell(COL_W[i], ROW_H / max(1, text.count("\n") + 1), text,
                        border=0, align="L")
    # Draw cell borders at uniform height
    for i in range(len(cells)):
        pdf.rect(x0 + sum(COL_W[:i]), y0, COL_W[i], ROW_H)
    pdf.set_xy(x0, y0 + ROW_H)

pdf.ln(2)

pdf.body_text(
    "No closed-source teacher model, proprietary API, or larger-model distillation was used. "
    "Runtime retrieval is limited to exact full-input matching over released Type 1 examples; "
    "no Physics vector index is shipped or active."
)

# 2. Approach
pdf.section("2. Approach and Method")
pdf.body_text(
    "Our system is a LangGraph-based agentic pipeline with two branches routed by the 'type' field:\n\n"
    "Type 1 (Logic): Exact full-input retrieval handles disclosed released examples. Otherwise "
    "Qwen2.5-Coder-7B formalizes entailment queries into Z3/Python; the sandbox executes and "
    "verifies them. Choice and number/text queries that do not fit entailment use a short-answer "
    "Qwen2.5-7B direct path. Responses preserve exact options and 0-based premise indices.\n\n"
    "Type 2 (Physics): A deterministic formula baseline handles common patterns (Ohm's law, "
    "series/parallel circuits, Coulomb's law, capacitor energy, etc.) with SI conversion and "
    "ASCII units. For complex problems, Qwen2.5-Coder-7B generates SymPy code, which runs in "
    "a sandboxed subprocess with time and memory limits.\n\n"
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
    ("Qwen2.5-Coder-7B-Instruct", "7.6B", "Q4_K_M (GGUF)", "Logic Z3 + Physics SymPy"),
    ("Qwen2.5-7B-Instruct", "7.6B", "Q4_K_M (GGUF)", "Direct answer / fallback"),
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
    "No MoE models are used. No closed-source or third-party inference APIs are used."
)

# 4. Infrastructure
pdf.section("4. Infrastructure")
pdf.body_text(
    "Self-hosted on Apple Silicon MacBook (M-series). FastAPI serves /predict and proxies "
    "the active llama-server /v1/models response. Public access uses an ngrok tunnel. "
    "All inference is local; no external API calls during evaluation."
)

output_path = Path(__file__).resolve().parents[1] / "reports" / "solution.pdf"
pdf.output(output_path)
print(f"Generated: {output_path}")
