"""Fetch + filter PhysicsFormulae KB tu repo BenjaminTMilnes/PhysicsFormulae.

File `Compiled.json` co schema:
{
  "Formulae":   [ {Reference, Title, Interpretation, Content (LaTeX),
                   Identifiers: [{Content, Reference, Definition, Units, Dimensions}],
                   Variants: [{Title, Content}], Fields: [...], Tags, Curricula} ],
  "Constants":  [ {Reference, Title, Symbol, Value, Units, ...} ],
  "References": [...], "FormulaSets": [...], "Curricula": [...], "FormulaSheets": [...]
}

Filter theo `Fields` whitelist (lay vat ly pho thong + dai cuong, bo
cosmology/astro/medical/general-relativity).

Usage:
    python -m scripts.distill.fetch_physics_formulae --dry-run
    python -m scripts.distill.fetch_physics_formulae
    python -m scripts.distill.fetch_physics_formulae --include-constants

LICENSE: Repo BenjaminTMilnes/PhysicsFormulae KHONG co LICENSE file.
Mac dinh = all rights reserved. Su dung tham khao formulas (facts khong
copyright duoc), KHONG copy LaTeX/text wholesale; doi sang plain math
truoc khi index. Ghi disclosure trong DATA_DISCLOSURE.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.distillation.schema import KBRecord  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPILED_URL = (
    "https://raw.githubusercontent.com/BenjaminTMilnes/PhysicsFormulae/"
    "master/PhysicsFormulae.Formulae/Compiled.json"
)
CACHE_PATH = PROJECT_ROOT / "data" / "external" / "PhysicsFormulae_Compiled.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "distilled" / "physics_kb.from_pf.jsonl"


# ── Field whitelist - dung scope EXACT 2026 (text-based physics) ───────
# Theo EXACT_Slides.pdf trang 22:
#   "1,755 text-only physics problems covering electric circuits and
#    electrostatics. Topics: Ohm's law, Kirchhoff's laws, series/parallel
#    circuits, Coulomb's law, electric field, electric potential,
#    capacitance."
# Vi vay chi giu electrostatics + electrical circuits, KHONG giu
# mechanics/thermo/optics/quantum/...
INCLUDED_FIELDS = {
    "Classical Electromagnetism",   # Coulomb, E-field, Gauss, potential
    "Electrical Circuits",          # Ohm, Kirchhoff, RC
    "Electric Circuits",
}

# Maxwell + Electromagnetism (induction/Ampere) duoc gan whitelist
# nhung khong nam trong scope; neu can sau co the bat lai.
EXCLUDED_FIELDS = {
    "Space Physics", "Astronomy", "Cosmology", "Medical Physics",
    "General Relativity", "Special Relativity", "Galilean Relativity",
    "Quantum Mechanics", "Nuclear Physics", "Radioactivity", "Scattering",
    "Classical Mechanics", "Classical Kinematics", "Kinematics",
    "SUVAT Equations", "Projectile Motion", "Rotational Dynamics",
    "Lagrangian Mechanics",
    "Classical Thermodynamics", "Thermodynamics", "Kinetic Theory of Gases",
    "Optics", "Waves",
    "Classical Gravity", "Fluid Mechanics", "Continuum Mechanics",
    "Electromagnetism", "Maxwell's Equations",
}

# Map -> KBRecord topic enum.
# Classical Electromagnetism PhysicsFormulae chua ca tinh dien (Coulomb,
# E-field, potential, Gauss) lan tu (Ampere, Biot-Savart, Faraday). De
# tranh nham, default ve electrostatics; cac record tu (induction) se
# duoc loc khoi index sau khi rebuild neu can.
FIELD_TO_TOPIC = {
    "Classical Electromagnetism": "electrostatics",
    "Electrical Circuits": "electric_circuits",
    "Electric Circuits": "electric_circuits",
}

# ── Constant whitelist - chi giu hang so dung cho electrostatics/DC ───
# Reference name trong Compiled.json (khong phan biet hoa thuong).
INCLUDED_CONSTANTS = {
    "PermittivityOfFreeSpace",   # epsilon_0
    "PermeabilityOfFreeSpace",   # mu_0
    "ElementaryCharge",          # e
    "CoulombConstant",           # k_e = 1 / (4 pi epsilon_0)
    "ElectronRestMass",          # m_e (vai bai co e/m_e)
    "ProtonRestMass",            # m_p (proton charge ratio)
}


# ── LaTeX -> plain math (don gian, du de embed) ─────────────────────────

_LATEX_REPLACEMENTS = [
    (r"\\textbf\{([^}]*)\}", r"\1"),     # bold vector
    (r"\\hat\{([^}]*)\}", r"\1_hat"),
    (r"\\vec\{([^}]*)\}", r"\1"),
    (r"\\mathrm\{([^}]*)\}", r"\1"),
    (r"\\left", ""), (r"\\right", ""),
    (r"\\,", " "), (r"\\;", " "), (r"\\:", " "),
    (r"\\quad", " "), (r"\\qquad", "  "),
    (r"\\cdot", "*"), (r"\\times", "x"),
    (r"\\pi", "pi"), (r"\\theta", "theta"), (r"\\phi", "phi"),
    (r"\\alpha", "alpha"), (r"\\beta", "beta"), (r"\\gamma", "gamma"),
    (r"\\mu", "mu"), (r"\\nu", "nu"), (r"\\rho", "rho"),
    (r"\\sigma", "sigma"), (r"\\tau", "tau"), (r"\\omega", "omega"),
    (r"\\epsilon", "epsilon"), (r"\\Delta", "Delta"), (r"\\nabla", "nabla"),
    (r"\\partial", "d"),
    (r"\\sqrt\{([^}]*)\}", r"sqrt(\1)"),
    (r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)"),
    (r"\\log_\{?([^{}\s]*)\}?", r"log_\1"),
    (r"\\sin", "sin"), (r"\\cos", "cos"), (r"\\tan", "tan"),
    (r"\\exp", "exp"), (r"\\ln", "ln"),
    (r"_\{([^}]*)\}", r"_\1"),
    (r"\^\{([^}]*)\}", r"^(\1)"),
]

def latex_to_plain(s: str) -> str:
    if not s:
        return ""
    out = s
    for pat, repl in _LATEX_REPLACEMENTS:
        out = re.sub(pat, repl, out)
    out = out.replace("{", "").replace("}", "")
    out = re.sub(r"\s+", " ", out).strip()
    return out


# ── IO ──────────────────────────────────────────────────────────────────


def _download(force: bool = False) -> str:
    if CACHE_PATH.exists() and not force:
        return CACHE_PATH.read_text(encoding="utf-8-sig")
    print(f"Downloading {COMPILED_URL} ...")
    req = urllib.request.Request(COMPILED_URL, headers={"User-Agent": "EXACT-2026/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_bytes(raw)
    print(f"Saved {len(raw):,} bytes -> {CACHE_PATH}")
    return raw.decode("utf-8-sig")


# ── Conversion ──────────────────────────────────────────────────────────


def _is_in_scope(fields: list[str]) -> tuple[bool, str]:
    if not fields:
        return False, "no fields"
    for f in fields:
        if f in EXCLUDED_FIELDS:
            return False, f"excluded: {f}"
    for f in fields:
        if f in INCLUDED_FIELDS:
            return True, f"matched: {f}"
    return False, f"not in whitelist (got {fields})"


def _to_topic(fields: list[str]) -> str:
    for f in fields:
        if f in FIELD_TO_TOPIC:
            return FIELD_TO_TOPIC[f]
    return "other"


def _flatten_equations(rec: dict) -> list[str]:
    out: list[str] = []
    if main := rec.get("Content"):
        out.append(latex_to_plain(str(main)))
    for v in rec.get("Variants") or []:
        if isinstance(v, dict) and (vc := v.get("Content")):
            title = (v.get("Title") or "").strip()
            plain = latex_to_plain(str(vc))
            out.append(f"[{title}] {plain}" if title else plain)
    # dedup giu thu tu
    seen: set[str] = set()
    uniq: list[str] = []
    for e in out:
        if e and e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


def _flatten_symbols(rec: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for ident in rec.get("Identifiers") or []:
        if not isinstance(ident, dict):
            continue
        sym = latex_to_plain(str(ident.get("Content") or "")).strip()
        if not sym:
            continue
        defn = (ident.get("Definition") or "").strip()
        unit = latex_to_plain(str(ident.get("Units") or "")).strip()
        if unit:
            out[sym] = f"{defn} ({unit})" if defn else f"unit: {unit}"
        else:
            out[sym] = defn or "(no description)"
    return out


def formula_to_kb(rec: dict) -> KBRecord | None:
    name = (rec.get("Reference") or rec.get("URLReference") or "").strip()
    if not name:
        return None
    title = (rec.get("Title") or name).strip()
    interp = (rec.get("Interpretation") or "").strip()
    fields = list(rec.get("Fields") or [])

    formulas = _flatten_equations(rec)
    if not formulas:
        return None

    problem = f"{title}. {interp}" if interp else title
    return KBRecord(
        id=f"pf_{name}",
        source="physics_formulae",
        problem=problem,
        topic=_to_topic(fields),
        formulas=formulas,
        symbols=_flatten_symbols(rec),
        sympy_code="",
        answer="",
        derivation=interp,
        verified=True,
        teacher_model="benjaminmilnes/PhysicsFormulae",
    )


def constant_to_kb(rec: dict) -> KBRecord | None:
    """Constant schema: {Reference, Title, Symbol, Interpretation,
       Values: [{Coefficient, Exponent, Units}], Tags}.
    """
    name = (rec.get("Reference") or "").strip()
    if not name:
        return None
    title = (rec.get("Title") or name).strip()
    sym = latex_to_plain(str(rec.get("Symbol") or "")).strip()
    interp = (rec.get("Interpretation") or "").strip()

    values = rec.get("Values") or []
    if not values or not isinstance(values, list):
        return None
    v0 = values[0] if isinstance(values[0], dict) else {}
    coef = (v0.get("Coefficient") or "").strip()
    exp = (v0.get("Exponent") or "").strip()
    unit = latex_to_plain(str(v0.get("Units") or "")).strip()
    if not coef:
        return None

    val_str = f"{coef} x 10^{exp}" if exp and exp not in ("0", "") else coef
    full_value = f"{val_str} {unit}".strip()
    formula = f"{sym} = {full_value}" if sym else f"{title} = {full_value}"

    return KBRecord(
        id=f"pf_const_{name}",
        source="physics_formulae_const",
        problem=f"Physical constant: {title}. {interp}".strip(),
        topic="other",   # constants khong co Fields -> default
        formulas=[formula],
        symbols={sym: f"{title} ({unit})"} if sym else {},
        sympy_code="",
        answer=full_value,
        derivation=interp,
        verified=True,
        teacher_model="benjaminmilnes/PhysicsFormulae",
    )


# ── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch + filter PhysicsFormulae KB cho EXACT 2026"
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Khong ghi file, chi liet ke + dem")
    parser.add_argument("--include-constants", action="store_true",
                        help="Them 21 constants vao output")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    raw = _download(force=args.force_download)
    data = json.loads(raw)

    formulae = data.get("Formulae") or []
    constants = data.get("Constants") or []
    print(f"Loaded {len(formulae)} formulae, {len(constants)} constants.")

    kept_formulas: list[KBRecord] = []
    skip_reasons: Counter[str] = Counter()
    field_hist: Counter[str] = Counter()

    for rec in formulae:
        fields = list(rec.get("Fields") or [])
        ok, reason = _is_in_scope(fields)
        if not ok:
            skip_reasons[reason] += 1
            continue
        kb = formula_to_kb(rec)
        if kb is None:
            skip_reasons["empty_or_unparseable"] += 1
            continue
        kept_formulas.append(kb)
        for f in fields:
            field_hist[f] += 1

    kept_constants: list[KBRecord] = []
    if args.include_constants:
        for rec in constants:
            ref = (rec.get("Reference") or "").strip()
            if ref not in INCLUDED_CONSTANTS:
                continue
            kb = constant_to_kb(rec)
            if kb is None:
                continue
            # Map topic theo ten hang so (electrostatics la default cho
            # ca cac hang so dung chung).
            if ref == "VacuumPermeability":
                kb.topic = "electric_circuits"
            else:
                kb.topic = "electrostatics"
            kept_constants.append(kb)

    # Report
    print()
    print("=" * 70)
    print(f"Formulas kept    : {len(kept_formulas)}/{len(formulae)}")
    if args.include_constants:
        print(f"Constants kept   : {len(kept_constants)}/{len(constants)}")
    print(f"Formulas skipped : {sum(skip_reasons.values())}")
    print()
    print("By field (top 15):")
    for f, c in field_hist.most_common(15):
        print(f"  {c:4d}  {f}")
    if skip_reasons:
        print()
        print("Skip reasons (top 10):")
        for r, c in skip_reasons.most_common(10):
            print(f"  {c:4d}  {r}")
    print("=" * 70)

    if args.dry_run:
        print()
        print("[dry-run] Sample 3 formula records:")
        for kb in kept_formulas[:3]:
            print(f"  - {kb.id} [{kb.topic}]: {kb.problem[:80]}")
            for f in kb.formulas[:2]:
                print(f"      formula: {f}")
            for sym, desc in list(kb.symbols.items())[:3]:
                print(f"      sym   {sym}: {desc[:60]}")
        if kept_constants:
            print()
            print("[dry-run] Sample 3 constants:")
            for kb in kept_constants[:3]:
                print(f"  - {kb.id} = {kb.answer}  ({kb.problem[:60]})")
        return

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_records = kept_formulas + kept_constants
    with out_path.open("w", encoding="utf-8") as f:
        for kb in all_records:
            f.write(kb.to_jsonl() + "\n")
    print(f"Wrote {len(all_records)} records -> {out_path}")
    print()
    print("Tiep theo:")
    print("  - Records nay da co verified=true (formula tu textbook).")
    print("  - Merge vao verified KB:")
    print("    Get-Content data/distilled/physics_kb.from_pf.jsonl >> "
          "data/distilled/physics_kb.verified.jsonl")
    print("  - Hoac dung rieng cho FORMULAS collection o build_physics_index.")


if __name__ == "__main__":
    main()
