# src/retrieval/

Thư mục này chứa code RAG (Retrieval-Augmented Generation) cho bài toán Vật lý.

## Cấu trúc

```
src/retrieval/
├── README.md               # File này
├── engine.py               # Retriever (hybrid + reranker)
└── vector_db.py            # VectorDBManager (Qdrant + LlamaIndex)
```

## Mục đích

Thư mục `src/retrieval/` chứa:
- **Hybrid Retriever**: BM25 + vector search + reranker
- **Vector DB**: Qdrant local + LlamaIndex docstore

## Chi tiết từng module

### engine.py

**Retriever** - Hybrid retrieval pipeline:

```python
class Retriever:
    def __init__(self, collection_name: str):
        self.vector_retriever = VectorIndexRetriever(...)
        self.bm25_retriever = BM25Retriever(...)
        self.fusion_retriever = QueryFusionRetriever(...)
        self.reranker = SentenceTransformerRerank(...)
```

**Pipeline**:
1. **Vector retriever**: BGE-M3 embedding, cosine similarity, top-12
2. **BM25 retriever**: LlamaIndex BM25Retriever, top-12
3. **Fusion**: `QueryFusionRetriever` mode `reciprocal_rerank`
4. **Reranker**: `SentenceTransformerRerank` model `BAAI/bge-reranker-base`, top-3

**Output**: Top-3 chunks format thành 2 section:
- `RELEVANT FORMULAS`: Công thức vật lý
- `WORKED EXAMPLES`: SymPy code examples

### vector_db.py

**VectorDBManager** - Quản lý Qdrant local + LlamaIndex:

```python
class VectorDBManager:
    def __init__(self, storage_path: str):
        self._index_cache: dict[str, object] = {}  # Cache theo collection name
        self.qdrant_client = QdrantClient(path=storage_path)
```

**Features**:
- Qdrant local mode (embedded, không cần server riêng)
- Persist tại `storage/qdrant_storage/`
- LlamaIndex docstore persist tại `storage/{collection_name}/`
- `_index_cache`: Cache index theo collection name (đã sửa bug cache single field)

**Methods**:
- `get_index(collection_name)` — Trả về index, tạo mới nếu chưa có
- `build_index(collection_name, documents)` — Build index từ documents

## Workflow

```
physics_rag_node
  → Retriever.retrieve(query)
    → Vector retriever (BGE-M3, top-12)
    → BM25 retriever (top-12)
    → Fusion (reciprocal_rerank)
    → Reranker (BAAI/bge-reranker-base, top-3)
  → Format context (formulas + examples)
  → physics_formalizer
```

## Yêu cầu

```powershell
pip install qdrant-client llama-index FlagEmbedding
```

## Collections

| Collection | Granularity | Mục đích |
|------------|-------------|----------|
| `physics_examples` | per-record | Query giống 1 bài cụ thể trong KB |
| `physics_formulas` | per-topic | Query mới → fallback formula sheet |

## Environment variables

- `HF_API_KEY`: API key cho HuggingFace (embedding, reranker)
