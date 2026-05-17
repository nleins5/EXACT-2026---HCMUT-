"""
Convert Type 1 (Logic) dataset → SFT training samples.

Input:  Logic_Based_Educational_Queries.json (411 records, 808 questions)
Output: List[dict] — mỗi dict là 1 SFT sample dạng conversations.

Quy trình:
  1. Lặp qua từng record
  2. Cho mỗi question i: dùng idx[i] (1-based) để filter đúng premises
  3. Ghép premises-NL + premises-FOL vào user message
  4. Ghép explanation + FOL vào <think>, answer vào <answer>
"""

import json
from pathlib import Path

# Đường dẫn mặc định tới file data gốc
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "EXACT2026_dataset_2026-05-15"
    / "Logic_Based_Educational_Queries_Text_Only"
    / "Logic_Based_Educational_Queries.json"
)


def convert_logic(input_path: Path = None) -> list[dict]:
    """Chuyển đổi toàn bộ Logic dataset thành danh sách SFT samples.

    Args:
        input_path: Đường dẫn tới file JSON gốc. Mặc định dùng file trong data/.

    Returns:
        Danh sách các sample, mỗi sample có key "conversations" (list of messages).
    """
    if input_path is None:
        input_path = DEFAULT_INPUT

    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    samples = []
    skipped = 0

    for record_idx, record in enumerate(records):
        premises_nl = record["premises-NL"]
        premises_fol = record.get("premises-FOL", [])
        questions = record["questions"]
        answers = record["answers"]
        explanations = record.get("explanation", [])
        idx_map = record.get("idx", [])

        for q_i, question in enumerate(questions):
            # --- Lấy answer ---
            answer = answers[q_i] if q_i < len(answers) else None
            if not answer:
                skipped += 1
                continue

            # --- Filter premises theo idx (1-based) ---
            if q_i < len(idx_map) and idx_map[q_i]:
                relevant_indices = idx_map[q_i]  # list of 1-based ints
                selected_nl = []
                selected_fol = []
                for idx_1based in relevant_indices:
                    idx_0based = idx_1based - 1
                    if 0 <= idx_0based < len(premises_nl):
                        selected_nl.append(premises_nl[idx_0based])
                    if 0 <= idx_0based < len(premises_fol):
                        selected_fol.append(premises_fol[idx_0based])
            else:
                # Nếu không có idx mapping → dùng tất cả premises
                selected_nl = premises_nl
                selected_fol = premises_fol

            # --- Build user message ---
            premises_block = "\n".join(
                [f"{i+1}. {p}" for i, p in enumerate(selected_nl)]
            )
            user_content = (
                f"[LOGIC PROBLEM]\n"
                f"Premises:\n{premises_block}\n\n"
                f"Question:\n{question}"
            )

            # --- Build assistant response ---
            explanation = explanations[q_i] if q_i < len(explanations) else ""

            # Thêm FOL vào phần thinking nếu có
            fol_block = ""
            if selected_fol:
                fol_lines = "\n".join(
                    [f"  P{i+1}: {f}" for i, f in enumerate(selected_fol)]
                )
                fol_block = f"\n\nFormal Logic (FOL):\n{fol_lines}\n"

            think_content = explanation + fol_block
            assistant_content = (
                f"<think>\n{think_content.strip()}\n</think>\n"
                f"<answer>\n{answer.strip()}\n</answer>"
            )

            samples.append({
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "type": "logic",
            })

    print(f"[Logic] Converted {len(samples)} samples from {len(records)} records (skipped {skipped})")
    return samples


if __name__ == "__main__":
    samples = convert_logic()
    # Preview
    if samples:
        print("\n--- Sample 0 ---")
        for msg in samples[0]["conversations"]:
            print(f"[{msg['role']}]")
            print(msg["content"][:300])
            print()
