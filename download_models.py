"""Download fine-tuned models from HuggingFace to local models folder.

Usage:
    venv\Scripts\python.exe download_models.py
"""

from huggingface_hub import snapshot_download
import os

# Cấu hình
MODEL_REPO = "HoangKhangHCMUS/exact-2026"
MODEL_DIR = "models/exact-2026"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Downloading models from {MODEL_REPO}...")
print(f"Saving to: {os.path.abspath(MODEL_DIR)}")

# Download all files from the repo
snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,  # Download actual files, not symlinks
)

print("\nDownload completed!")
print(f"Files saved to: {MODEL_DIR}")
