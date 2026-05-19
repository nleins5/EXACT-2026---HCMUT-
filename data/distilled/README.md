# Distilled physics knowledge base

Thư mục này chứa knowledge base cho `physics_rag_node` của EXACT 2026.
**Scope cuộc thi**: chỉ 2 chủ đề — `electrostatics` + `electric_circuits`
(theo `data/EXACT_Slides.pdf` trang 22). KB ở đây bám đúng scope.

> [!NOTE]
> File `physics_kb.raw.jsonl`, `physics_kb.verified.jsonl`, `cost_log.jsonl`
> đã thêm vào `.gitignore` vì là output regen được. Chỉ
> `physics_kb.from_pf.jsonl` (small + curated) là commit lên git.

## File

| File | Tracked? | Mô tả |
|---|---|---|
| `physics_kb.from_pf.jsonl` | ✅ git | KB sạch lọc từ PhysicsFormulae — **22 formulas + 6 constants = 28 records**, tất cả `verified=true` |
| `physics_kb.raw.jsonl` | ❌ ignore | Output thô teacher LLM (chưa exec SymPy) |
| `physics_kb.verified.jsonl` | ❌ ignore | Sau khi `verify_kb.py` exec, mark `verified=true/false` |
| `cost_log.jsonl` | ❌ ignore | Log token + latency mỗi call teacher |

## Schema record (KBRecord)

```json
{
  "id": "btc_pb_42",
  "source": "btc_physics",
  "problem": "Hai dien tro R1=30Ohm, R2=60Ohm mac song song...",
  "topic": "electric_circuits",
  "formulas": ["R_eq = R1*R2/(R1+R2)"],
  "symbols": {"R1": "resistor 1 (Ohm)", "R2": "resistor 2 (Ohm)", "R_eq": "equivalent resistance (Ohm)"},
  "sympy_code": "import sympy as sp\nR1, R2 = sp.Rational(30), sp.Rational(60)\nR_eq = R1*R2/(R1+R2)\nprint(f'FINAL_ANSWER: {float(R_eq)} Ohm')",
  "answer": "20 Ohm",
  "derivation": "Two resistors in parallel: R_eq = R1*R2/(R1+R2)",
  "verified": true,
  "exec_output": "FINAL_ANSWER: 20.0 Ohm\n",
  "exec_error": "",
  "teacher_model": "gemini-2.5-flash-lite",
  "input_tokens": 312,
  "output_tokens": 180
}
```

`topic` chỉ nhận **`electrostatics` | `electric_circuits` | `other`**
(scope EXACT 2026 hẹp hơn rất nhiều so với physics tổng quát).

## Pipeline distillation — extract mode

> [!IMPORTANT]
> Dataset BTC đã có `cot` field chứa cả công thức và số liệu.
> Pipeline mới chuyển sang **extract mode**: teacher chỉ TRÍCH XUẤT
> formula list / symbol map / clean SymPy code từ CoT có sẵn,
> KHÔNG sinh từ đầu. Output ngắn hơn ~60% → rẻ hơn + ít hallucination.

Cấu hình `config/setting.yaml`:

```yaml
distillation:
  teacher:
    model_name: gemini-2.5-flash-lite   # rẻ nhất dòng Flash
    temperature: 0.1                    # extract cần deterministic
    max_output_tokens: 1024
  pipeline:
    mode: extract                       # extract | generate
    concurrency: 8
```

Ước tính chi phí toàn bộ ~1,594 problems (BTC 1,352 + electro 242):
- Free tier Gemini Flash Lite: 0 USD (qua đêm với rate-limit 15 RPM).
- Trả phí + Batch API: ~0.18 USD.

## Quy trình tái tạo

```powershell
# 0. Cài dep + set API key
.\venv\Scripts\pip.exe install google-generativeai
$env:GOOGLE_API_KEY = "<your-key>"

# 1. Distill (resumable — cancel/restart không mất tiến độ)
.\venv\Scripts\python.exe -m scripts.distill.distill_physics --source all
#   --source btc | electro | all   (default all)
#   --limit 10                     thử 10 record trước
#   --concurrency 4                giảm khi rate-limit
#   --dry-run                      chỉ liệt kê ID, không gọi LLM

# 2. Verify SymPy code (chạy từng record trong subprocess, timeout 10s)
.\venv\Scripts\python.exe -m scripts.distill.verify_kb

# 3. (Tùy chọn) Refresh KB từ PhysicsFormulae
.\venv\Scripts\python.exe -m scripts.distill.fetch_physics_formulae --include-constants

# 4. Build 2 collection vào Qdrant
.\venv\Scripts\python.exe -m scripts.rag.build_physics_index --rebuild
```

## Nguồn PhysicsFormulae (external)

`physics_kb.from_pf.jsonl` được sinh từ
[BenjaminTMilnes/PhysicsFormulae](https://github.com/BenjaminTMilnes/PhysicsFormulae)
(file `Compiled.json` ~655KB, đã ignore khỏi git).

- **Filter**: giữ duy nhất `Classical Electromagnetism` + `Electric Circuits`,
  bỏ mechanics / thermodynamics / optics / quantum / relativity. Trong constants
  giữ 6 hằng phục vụ điện từ: `epsilon_0`, `mu_0`, `e`, `k_e`, `m_e`, `m_p`.
- **Convert**: LaTeX → plain math (`\frac{a}{b}` → `(a)/(b)`, `\textbf{F}` → `F`)
  để embedding bắt được symbol.
- **Verified=true** mặc định vì là công thức textbook chuẩn, không cần exec sympy.
- **License**: repo source không có LICENSE → mặc định all rights reserved.
  Facts (công thức) không bản quyền được; LaTeX đã transform sang plain math
  (không copy nguyên văn). Phải khai báo trong `docs/DATA_DISCLOSURE.md`
  khi nộp BTC.

## Kiến trúc 2 collection

| Collection | Granularity | Dùng khi |
|---|---|---|
| `physics_examples` | per-record | Query runtime gần nghĩa với 1 bài cụ thể trong KB |
| `physics_formulas` | per-topic (gộp formula) | Query mới không match bài nào → fallback formula sheet |

`physics_rag_node` query song song cả 2, format thành 2 section trong context:

```
RELEVANT FORMULAS (apply these to derive the answer):
Topic: electric_circuits
Canonical formulas: R_eq = R1*R2/(R1+R2), I = U/R, ...

WORKED EXAMPLES (reference the SymPy code style; do NOT blindly copy):
Example 1:
Problem: ...
SymPy code: ...
```

## Tradeoff & ghi chú

- **Tại sao distill?** BTC chỉ ~1,352 bài, FT 7B trên dataset nhỏ dễ overfit.
  RAG giữ generality + bổ sung công thức chuẩn từ PhysicsFormulae.
- **Tại sao extract thay vì generate?** Dataset đã có CoT + answer chuẩn.
  Teacher chỉ extract → ít hallucination, rẻ token, deterministic.
- **Verified rate kỳ vọng** ~85-95% (extract mode strict hơn generate).
  Bài fail có thể distill round 2 với prompt strict.
- **BTC Q11 (Data Disclosure)**: phải khai báo dùng teacher LLM (Gemini)
  tạo corpus RAG. Sửa `docs/DATA_DISCLOSURE.md` khi nộp.
