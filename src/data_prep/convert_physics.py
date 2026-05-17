"""
Convert Type 2 (Physics) dataset → SFT training samples.

Input:  Physics_Problems_Text_Only.csv (1352 problems)
Output: List[dict] — mỗi dict là 1 SFT sample dạng conversations.

Quy trình:
  1. Đọc CSV
  2. Cho mỗi row: ghép question vào user, cot vào <think>, answer+unit vào <answer>
  3. Xử lý edge cases: cot trống, unit trống
"""

import csv
from pathlib import Path

# Đường dẫn mặc định
DEFAULT_INPUT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "EXACT2026_dataset_2026-05-15"
    / "Physics_Problems_Text_Only"
    / "Physics_Problems_Text_Only.csv"
)


def convert_physics(input_path: Path = None) -> list[dict]:
    """Chuyển đổi toàn bộ Physics dataset thành danh sách SFT samples.

    Args:
        input_path: Đường dẫn tới file CSV gốc.

    Returns:
        Danh sách các sample, mỗi sample có key "conversations".
    """
    if input_path is None:
        input_path = DEFAULT_INPUT

    samples = []
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "").strip()
            cot = row.get("cot", "").strip()
            answer = row.get("answer", "").strip()
            unit = row.get("unit", "").strip()

            # Skip rows thiếu question hoặc answer
            if not question or not answer:
                skipped += 1
                continue

            # --- Build user message ---
            user_content = f"[PHYSICS PROBLEM]\n{question}"

            # --- Build assistant response ---
            # Chain-of-thought: dùng cot từ BTC, nếu trống thì ghi minimal
            if cot:
                think_content = cot
            else:
                think_content = f"Let me solve this problem step by step.\nThe answer is {answer} {unit}."

            # Ghép answer + unit
            final_answer = f"{answer} {unit}".strip() if unit else answer

            assistant_content = (
                f"<think>\n{think_content}\n</think>\n"
                f"<answer>\n{final_answer}\n</answer>"
            )

            samples.append({
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "type": "physics",
            })

    print(f"[Physics] Converted {len(samples)} samples (skipped {skipped})")
    return samples


if __name__ == "__main__":
    samples = convert_physics()
    if samples:
        print("\n--- Sample 0 ---")
        for msg in samples[0]["conversations"]:
            print(f"[{msg['role']}]")
            print(msg["content"][:300])
            print()
