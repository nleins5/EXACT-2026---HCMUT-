"""Guard against reintroducing exposed deployment secrets and scratch files."""
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebook_does_not_hardcode_ngrok_token():
    for relative in ("EXACT_2026_Colab_Deployment.ipynb", "scripts/create_colab_notebook.py"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert not re.search(r"""NGROK_AUTHTOKEN\s*=\s*["'][^"'\n]+["']""", text)


def test_root_scratch_files_are_not_present():
    for name in (".DS_Store", "pdf_content.txt", "pdf_content_utf8.txt"):
        assert not (ROOT / name).exists()
