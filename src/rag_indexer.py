"""
rag_indexer.py — Build and query FAISS index over Uber_Support historical pairs.

Usage:
  # Build index (one-time)
  python src/rag_indexer.py --build

  # Query
  python src/rag_indexer.py --query "my driver never showed up"
"""

import json
import os
import numpy as np
import faiss
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBED_MODEL     = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
PAIRS_JSON      = "data/uber_pairs_full.json"
FAISS_INDEX     = os.getenv("FAISS_INDEX_PATH", "models/uber_faiss.index")
FAISS_META      = os.getenv("FAISS_META_PATH",  "models/uber_faiss_meta.json")
TOP_K           = int(os.getenv("TOP_K_RETRIEVAL", 5))


# ------------------------------------------------------------------
# Index construction
# ------------------------------------------------------------------

def build_index(pairs_path: str = PAIRS_JSON,
                index_path: str = FAISS_INDEX,
                meta_path:  str = FAISS_META,
                model_name: str = EMBED_MODEL,
                min_agent_len: int = 10):
    """Embed customer messages, build FAISS index, save metadata."""
    with open(pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)

    # Filter out pairs with trivially short agent replies (boilerplate acks)
    pairs = [p for p in pairs if len(p["agent_reply"]) >= min_agent_len]
    print(f"Indexing {len(pairs):,} pairs (after quality filter) …")

    texts = [p["customer_msg"] for p in pairs]

    model  = SentenceTransformer(model_name)
    embeds = model.encode(
        texts, batch_size=256, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )

    d     = embeds.shape[1]
    index = faiss.IndexFlatIP(d)        # exact cosine search (inner product on L2-normed vecs)
    index.add(embeds.astype("float32"))

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved → {index_path}  ({index.ntotal:,} vectors, dim={d})")

    # Save metadata (agent replies + context) aligned with index positions
    meta = [
        {
            "thread_id":    p["thread_id"],
            "customer_msg": p["customer_msg"],
            "agent_reply":  p["agent_reply"],
            "full_context": p.get("full_context", ""),
        }
        for p in pairs
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"Metadata saved     → {meta_path}")
    return index, meta


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------

class RAGRetriever:
    """Load FAISS index and retrieve similar historical Uber replies."""

    def __init__(self, index_path: str = FAISS_INDEX,
                 meta_path:  str = FAISS_META,
                 model_name: str = EMBED_MODEL,
                 top_k:      int = TOP_K):
        self.model  = SentenceTransformer(model_name)
        self.index  = faiss.read_index(index_path)
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.top_k  = top_k
        print(f"RAGRetriever ready — {self.index.ntotal:,} vectors, top_k={top_k}")

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return top-k most similar historical pairs for a query."""
        k = top_k or self.top_k
        emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, idxs = self.index.search(emb.astype("float32"), k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:          # FAISS returns -1 for no result
                continue
            item = self.meta[idx].copy()
            item["retrieval_score"] = float(score)
            results.append(item)
        return results

    @property
    def top_retrieval_score(self) -> float:
        """Helper: max cosine similarity of last retrieval (for escalation router)."""
        # Callers should use results[0]["retrieval_score"] directly
        return 0.0


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Uber RAG indexer")
    parser.add_argument("--build", action="store_true", help="Build FAISS index")
    parser.add_argument("--query", type=str,            help="Test retrieval query")
    parser.add_argument("--topk",  type=int, default=5, help="Number of results")
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.query:
        ret = RAGRetriever(top_k=args.topk)
        results = ret.retrieve(args.query)
        print(f"\nTop-{args.topk} results for: '{args.query}'\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['retrieval_score']:.3f}")
            print(f"    Customer : {r['customer_msg'][:90]}")
            print(f"    Agent    : {r['agent_reply'][:90]}")
            print()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
