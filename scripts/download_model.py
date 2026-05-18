"""Download fine-tuned GGUF model từ Hugging Face Hub về local.

Sử dụng HF_API_KEY trong file .env để authenticate (cho repo private).
Sau khi download xong, model sẽ được lưu tại: models/<filename>.gguf

Cách dùng:
    python scripts/download_model.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout cho Windows (tránh UnicodeEncodeError trên cp1252 console)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
REPO_ID = "HoangKhangHCMUS/exact-2026-llama3-gguf"
FILENAME = "deepseek-r1-0528-qwen3-8b.Q4_K_M.gguf"

# Thư mục đích lưu model (đã được .gitignore)
LOCAL_DIR = PROJECT_ROOT / "models"


def main() -> None:
    # Load HF_API_KEY từ .env
    load_dotenv(PROJECT_ROOT / ".env")
    hf_token = os.getenv("HF_API_KEY")

    if not hf_token:
        print("[!] ERROR: Không tìm thấy HF_API_KEY trong file .env")
        print("    Hãy thêm dòng: HF_API_KEY=hf_xxxxxxxxxxxxxxxxxx")
        sys.exit(1)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    target_path = LOCAL_DIR / FILENAME

    if target_path.exists():
        size_mb = target_path.stat().st_size / (1024 * 1024)
        print(f"[=] Model đã tồn tại: {target_path}")
        print(f"    Size: {size_mb:.1f} MB")
        print("    Nếu muốn tải lại, hãy xoá file này trước.")
        return

    print(f"[+] Repo:     {REPO_ID}")
    print(f"[+] File:     {FILENAME}")
    print(f"[+] Dest dir: {LOCAL_DIR}")
    print("[+] Downloading (~5GB, may take a few minutes)...\n")

    t0 = time.time()
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(LOCAL_DIR),
        token=hf_token,
        # local_dir_use_symlinks=False  # Bỏ comment nếu muốn copy thật thay vì symlink
    )
    elapsed = time.time() - t0

    final_size_mb = Path(downloaded_path).stat().st_size / (1024 * 1024)
    print(f"\n[+] Download done in {elapsed:.1f}s")
    print(f"[+] File path:  {downloaded_path}")
    print(f"[+] File size:  {final_size_mb:.1f} MB")
    print("\n[OK] Ready to load with llama-cpp-python!")
    print("     -> python scratch/test_hf_llama_cpp.py")


if __name__ == "__main__":
    main()
