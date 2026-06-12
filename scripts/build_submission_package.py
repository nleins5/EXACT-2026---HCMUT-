#!/usr/bin/env python3
"""Build the EXACT source archive and final team submission bundle."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCE_ZIP = REPORTS / "source_code.zip"
TEAM_ZIP = REPORTS / "AI_WITH_BRO.zip"
TEAM_DIR = REPORTS / "AI_WITH_BRO"

SOURCE_PREFIXES = (
    ".github/",
    "cloudflare/",
    "config/",
    "fine_tune/",
    "models/",
    "scripts/",
    "src/",
    "tests/",
)
SOURCE_ROOT_FILES = {
    ".env.example",
    ".gitattributes",
    ".gitignore",
    "EXACT_2026_Colab_Deployment.ipynb",
    "README.md",
    "requirements-dev.txt",
    "requirements.txt",
    "test_pipeline.py",
}
TEAM_FILES = ("solution.pdf", "source_code.zip", "urls.txt", "notation_mapping.csv")


def source_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    )
    selected: list[Path] = []
    for relative in sorted(set(output.splitlines())):
        path = ROOT / relative
        if not path.is_file():
            continue
        if relative in SOURCE_ROOT_FILES or relative.startswith(SOURCE_PREFIXES):
            if path.suffix.lower() not in {".gguf", ".pdf", ".zip"}:
                selected.append(path)
    return selected


def write_zip(path: Path, files: list[Path], base: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(base).as_posix())


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    files = source_files()
    write_zip(SOURCE_ZIP, files, ROOT)

    TEAM_DIR.mkdir(exist_ok=True)
    team_paths = []
    for name in TEAM_FILES:
        source = REPORTS / name
        if not source.exists():
            raise FileNotFoundError(f"Missing submission artifact: {source}")
        shutil.copy2(source, TEAM_DIR / name)
        team_paths.append(source)
    write_zip(TEAM_ZIP, team_paths, REPORTS)

    print(f"Built {SOURCE_ZIP} with {len(files)} source files")
    print(f"Built {TEAM_ZIP} with {len(team_paths)} submission artifacts")


if __name__ == "__main__":
    main()
