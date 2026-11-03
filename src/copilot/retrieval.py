"""ChromaDB retrieval: embed, query top-k, upsert for feedback loop."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K
from .preprocess import clean_text


@dataclass
class RetrievedTicket:
    ticket_id: int
    similarity: float
    snippet: str
    answer: str
    metadata: dict[str, Any]


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def get_collection(create: bool = True):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    if create:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=COLLECTION_NAME)


def _distance_to_similarity(distance: float) -> float:
    """Chroma cosine space returns distance = 1 - cosine_similarity."""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def search(query: str, k: int = TOP_K) -> list[RetrievedTicket]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    q = clean_text(query, strict=False)
    q_emb = embed_texts([q])[0]
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[RetrievedTicket] = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    for doc, meta, dist, doc_id in zip(docs, metas, dists, ids):
        meta = meta or {}
        ticket_id = int(meta.get("ticket_id", doc_id.replace("ticket_", "") or 0))
        answer = str(meta.get("answer", ""))
        snippet = (doc or "")[:240]
        hits.append(
            RetrievedTicket(
                ticket_id=ticket_id,
                similarity=_distance_to_similarity(dist),
                snippet=snippet,
                answer=answer,
                metadata=dict(meta),
            )
        )
    return hits


def upsert(
    question: str,
    resolution: str,
    *,
    ticket_id: int | None = None,
    ticket_type: str = "Request",
    queue: str = "senior_resolved",
    priority: str = "medium",
    source: str = "senior",
) -> int:
    """Upsert a Q+A pair into the KB (feedback loop)."""
    collection = get_collection()
    text = clean_text(question, strict=False)
    if ticket_id is None:
        # Use a high offset so senior-resolved ids don't collide with dataset ids
        ticket_id = 1_000_000 + collection.count()

    emb = embed_texts([text])[0]
    doc_id = f"ticket_{ticket_id}"
    collection.upsert(
        ids=[doc_id],
        embeddings=[emb],
        documents=[text],
        metadatas=[
            {
                "ticket_id": ticket_id,
                "type": ticket_type,
                "queue": queue,
                "priority": priority,
                "answer": resolution,
                "source": source,
            }
        ],
    )
    return ticket_id


def bulk_add(
    ticket_ids: list[int],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    batch_size: int = 256,
) -> int:
    """Bulk index tickets into Chroma."""
    collection = get_collection()
    n = len(texts)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_texts = [clean_text(t, strict=False) for t in texts[start:end]]
        embeddings = embed_texts(batch_texts)
        ids = [f"ticket_{tid}" for tid in ticket_ids[start:end]]
        # Chroma metadata values must be str/int/float/bool
        clean_metas = []
        for m in metadatas[start:end]:
            clean_metas.append(
                {
                    "ticket_id": int(m["ticket_id"]),
                    "type": str(m.get("type", "")),
                    "queue": str(m.get("queue", "")),
                    "priority": str(m.get("priority", "")),
                    "answer": str(m.get("answer", ""))[:2000],
                    "source": str(m.get("source", "dataset")),
                }
            )
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=clean_metas,
        )
        print(f"  Indexed {end}/{n}")
    return collection.count()
