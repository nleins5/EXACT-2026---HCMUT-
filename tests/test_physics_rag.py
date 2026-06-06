"""Tests for cheap no-index physics RAG behavior."""
from src.agent.nodes import physics_rag


def test_missing_index_skips_retriever(monkeypatch, tmp_path):
    monkeypatch.setattr(physics_rag, "_STORAGE_ROOT", tmp_path)
    result = physics_rag.physics_rag_node({"question": "What is force?"})
    assert result == {"context": ""}
