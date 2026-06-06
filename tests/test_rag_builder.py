"""Tests for the provider-independent physics RAG input schema."""
import json

from scripts.rag.build_physics_index import KBRecord, _load_verified


def test_kb_record_normalizes_question_and_formula():
    record = KBRecord.from_jsonl(
        json.dumps(
            {
                "id": "p1",
                "question": "Find resistance",
                "formulas": "R = V/I",
                "verified": True,
            }
        )
    )
    assert record.problem == "Find resistance"
    assert record.formulas == ["R = V/I"]
    assert record.verified is True


def test_load_verified_filters_unverified_and_qa_ids(tmp_path):
    path = tmp_path / "kb.jsonl"
    records = [
        {"id": "good", "verified": True},
        {"id": "bad", "verified": False},
        {"id": "QA-123", "verified": True},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    loaded = _load_verified([path])
    assert [record.id for record in loaded] == ["good"]
