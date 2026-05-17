"""
Build SFT Dataset — Merge Logic + Physics → JSONL.

Pipeline:
  1. convert_logic()  → 808 samples
  2. convert_physics() → 1352 samples
  3. Merge + Shuffle (seeded)
  4. Inject system prompt vào mỗi sample
  5. Split train/val (90/10)
  6. Export → train.jsonl + val.jsonl

Output sẵn sàng upload lên Google Drive / HuggingFace để Colab load.
"""

import json
import random
from pathlib import Path

from src.data_prep.convert_logic import convert_logic
from src.data_prep.convert_physics import convert_physics

# Output directory
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sft_dataset"

# System prompt cho SFT
SYSTEM_PROMPT = (
    "You are an expert educational AI assistant for the EXACT 2026 competition. "
    "For logic problems: analyze premises carefully, apply formal reasoning, "
    "and derive the correct conclusion. "
    "For physics problems: identify relevant formulas, show step-by-step calculations, "
    "and provide the final numerical answer with correct units. "
    "Always think step-by-step inside <think>...</think> tags, "
    "then give your final answer inside <answer>...</answer> tags."
)

SEED = 3407  # Reproducibility


def build_dataset(
    val_ratio: float = 0.1,
    output_dir: Path = None,
):
    """Build và export SFT dataset.

    Args:
        val_ratio: Tỷ lệ validation split (mặc định 10%).
        output_dir: Thư mục output. Mặc định data/sft_dataset/.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert
    logic_samples = convert_logic()
    physics_samples = convert_physics()

    # 2. Inject system prompt
    all_samples = []
    for sample in logic_samples + physics_samples:
        conversations = sample["conversations"]
        # Thêm system message ở đầu
        conversations.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        all_samples.append({
            "conversations": conversations,
            "type": sample["type"],
        })

    # 3. Shuffle
    random.seed(SEED)
    random.shuffle(all_samples)

    # 4. Split
    total = len(all_samples)
    val_count = int(total * val_ratio)
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]

    # 5. Export
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    _write_jsonl(train_samples, train_path)
    _write_jsonl(val_samples, val_path)

    # 6. Stats
    logic_count = sum(1 for s in all_samples if s["type"] == "logic")
    physics_count = sum(1 for s in all_samples if s["type"] == "physics")

    print(f"\n{'='*50}")
    print(f"SFT Dataset Built Successfully!")
    print(f"{'='*50}")
    print(f"  Total samples:   {total}")
    print(f"  |-- Logic:       {logic_count}")
    print(f"  |-- Physics:     {physics_count}")
    print(f"  Train set:       {len(train_samples)}")
    print(f"  Val set:         {len(val_samples)}")
    print(f"  Train file:      {train_path}")
    print(f"  Val file:        {val_path}")
    print(f"{'='*50}")

    return train_samples, val_samples


def _write_jsonl(samples: list[dict], path: Path):
    """Ghi danh sách samples ra file JSONL (1 JSON per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            # Chỉ export conversations, bỏ field "type" (metadata nội bộ)
            line = json.dumps(
                {"conversations": sample["conversations"]},
                ensure_ascii=False,
            )
            f.write(line + "\n")
    print(f"  Wrote {len(samples)} samples -> {path}")


if __name__ == "__main__":
    build_dataset()
