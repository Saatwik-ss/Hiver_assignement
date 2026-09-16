"""
intent_discovery.py — Cluster Uber_Support customer messages to discover latent intents.

Steps:
  1. Embed all customer messages with bge-small-en-v1.5
  2. UMAP dimensionality reduction
  3. HDBSCAN clustering
  4. Print cluster summaries + centroids for human labelling
  5. Save cluster assignments to data/clusters.json

After running this, manually inspect clusters and fill data/few_shot_train.json
with 16 examples per intent.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
PAIRS_JSON  = "data/uber_pairs_full.json"
CLUSTER_OUT = "data/clusters.json"
EMBED_CACHE = "data/embeddings_cache.npy"

# Uber-specific intent taxonomy (discovered from clustering, then manually labelled)
# Update these labels after inspecting cluster output
INTENT_NAMES = {
    0:  "trip_issue",           # wrong route, bad GPS, navigation errors
    1:  "payment_billing",      # overcharge, refund, promo not applied
    2:  "driver_behavior",      # rude driver, professionalism complaints
    3:  "safety_incident",      # accident, harassment, threatening behavior
    4:  "account_access",       # login issues, account suspended/locked
    5:  "app_technical",        # app crash, surge pricing bug, UI issues
    6:  "ride_cancellation",    # driver cancelled, can't find driver, surge
    7:  "lost_item",            # left item in car, contact driver
    8:  "wait_time",            # driver taking too long, ETA wrong
    9:  "eats_order",           # Uber Eats: wrong order, late delivery, missing items
    10: "general_complaint",    # vague frustration, unclear intent
    11: "positive_feedback",    # praise, compliments
}


def load_pairs(path: str = PAIRS_JSON) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def embed_messages(texts: list[str], model_name: str = EMBED_MODEL,
                   cache_path: str = EMBED_CACHE) -> np.ndarray:
    """Encode texts; uses file cache to avoid re-embedding on re-runs."""
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Encoding {len(texts):,} messages with {model_name} …")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, embeddings)
    print(f"Saved embeddings → {cache_path}")
    return embeddings


def run_clustering(embeddings: np.ndarray, n_components: int = 15,
                   min_cluster_size: int = 80, min_samples: int = 10):
    """UMAP → HDBSCAN clustering."""
    try:
        import umap
        import hdbscan
    except ImportError:
        raise ImportError("Run: pip install umap-learn hdbscan")

    print("Running UMAP …")
    reducer = umap.UMAP(
        n_components=n_components,
        metric="cosine",
        n_neighbors=30,
        min_dist=0.0,
        random_state=42,
        low_memory=True,
    )
    reduced = reducer.fit_transform(embeddings)

    print("Running HDBSCAN …")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    return labels, reduced


def summarize_clusters(pairs: list[dict], labels: np.ndarray,
                       n_examples: int = 8) -> dict:
    """Group examples by cluster; print top representatives."""
    from collections import defaultdict
    clusters: dict[int, list[str]] = defaultdict(list)
    for pair, label in zip(pairs, labels):
        clusters[int(label)].append(pair["customer_msg"])

    summary = {}
    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY — assign intent names in INTENT_NAMES dict above")
    print("=" * 70)

    for cluster_id in sorted(clusters.keys()):
        examples = clusters[cluster_id]
        label_name = INTENT_NAMES.get(cluster_id, f"cluster_{cluster_id}")
        noise_flag = " [NOISE — review manually]" if cluster_id == -1 else ""
        print(f"\n── Cluster {cluster_id} ({len(examples)} examples){noise_flag}")
        print(f"   Suggested label: {label_name}")
        print("   Representative examples:")
        for ex in examples[:n_examples]:
            print(f"     • {ex[:100]}")
        summary[cluster_id] = {
            "count": len(examples),
            "label": label_name,
            "examples": examples[:n_examples],
        }
    return summary


def save_clusters(pairs: list[dict], labels: np.ndarray, out_path: str = CLUSTER_OUT):
    """Save cluster assignments to JSON."""
    data = []
    for pair, label in zip(pairs, labels):
        data.append({**pair, "cluster_id": int(label),
                     "intent": INTENT_NAMES.get(int(label), "unknown")})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved cluster assignments → {out_path}")


def sample_few_shot(pairs: list[dict], labels: np.ndarray,
                    shots: int = 16, out_path: str = "data/few_shot_train.json"):
    """Sample `shots` examples per cluster for SetFit training."""
    from collections import defaultdict
    import random
    random.seed(42)

    by_cluster: dict[int, list] = defaultdict(list)
    for pair, label in zip(pairs, labels):
        if label >= 0:  # skip noise cluster (-1)
            by_cluster[int(label)].append(pair)

    few_shot = []
    for cluster_id, cluster_pairs in by_cluster.items():
        sampled = random.sample(cluster_pairs, min(shots, len(cluster_pairs)))
        intent_name = INTENT_NAMES.get(cluster_id, f"cluster_{cluster_id}")
        for p in sampled:
            few_shot.append({
                "text":   p["customer_msg"],
                "label":  intent_name,
                "source": p["thread_id"],
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(few_shot, f, ensure_ascii=False, indent=2)

    print(f"\nFew-shot training set: {len(few_shot)} examples "
          f"({shots} × {len(by_cluster)} clusters) → {out_path}")
    print("⚠️  Review and correct labels in this file before training SetFit!")
    return few_shot


if __name__ == "__main__":
    pairs = load_pairs()
    texts = [p["customer_msg"] for p in pairs]

    embeddings = embed_messages(texts)
    labels, reduced = run_clustering(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = sum(1 for l in labels if l == -1)
    print(f"\nClusters found: {n_clusters}  |  Noise points: {n_noise:,}")

    summarize_clusters(pairs, labels)
    save_clusters(pairs, labels)
    sample_few_shot(pairs, labels, shots=int(os.getenv("SHOTS_PER_CLASS", 16)))
