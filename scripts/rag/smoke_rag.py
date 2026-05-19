"""Smoke test physics_rag_node voi 3 BTC-style question."""
import sys, io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.agent.nodes.physics_rag import physics_rag_node


QUESTIONS = [
    "Tinh dien luong cua diem dat tai khoang cach r=0.5m, biet luc Coulomb F=10N va dien tich q1=2e-6 C",
    "Mot vat khoi luong 2kg roi tu do tu do cao h=10m. Tinh van toc khi cham dat (g=9.8 m/s^2)",
    "Mach RLC noi tiep co R=10 ohm, L=0.1 H, C=100 uF. Tinh tan so cong huong",
]

for i, q in enumerate(QUESTIONS, 1):
    print(f"\n{'='*70}\nQ{i}: {q}\n{'='*70}")
    r = physics_rag_node({"question": q})
    ctx = r.get("context", "")
    if not ctx:
        print("[empty context]")
    else:
        print(ctx[:1800])
