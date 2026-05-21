"""Download fine-tuned GGUF models from HuggingFace.

Usage:
    cd models/
    python download_models.py

Models:
    1. Qwen2.5-7B-Instruct.Q4_K_M.gguf   (fine-tuned instruct)
    2. qwen2.5-coder-7b-instruct.Q4_K_M.gguf (fine-tuned coder)

Source: https://huggingface.co/HoangKhangHCMUS/exact-2026
"""

from huggingface_hub import hf_hub_download
from pathlib import Path

REPO_ID = "HoangKhangHCMUS/exact-2026"
MODEL_DIR = Path(__file__).parent / "exact-2026"

MODELS = [
    "Qwen2.5-7B-Instruct.Q4_K_M.gguf",
    "qwen2.5-coder-7b-instruct.Q4_K_M.gguf",
]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading models from {REPO_ID}")
    print(f"Saving to: {MODEL_DIR.resolve()}\n")

    for filename in MODELS:
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"[SKIP] {filename} (already exists, {dest.stat().st_size / 1e9:.1f} GB)")
            continue
        print(f"[DOWNLOAD] {filename} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(MODEL_DIR),
        )
        print(f"  -> Done ({dest.stat().st_size / 1e9:.1f} GB)")

    print("\nAll models ready!")


if __name__ == "__main__":
    main()
