"""
baselines.py — Two baselines required by assignment:

Baseline 1 (Trivial):
  - Intent: always predict most-common class
  - Reply:  fixed template string
  - Escalation: never escalate (always auto)

Baseline 2 (Simple):
  - Intent: nearest-neighbour cosine similarity (1 shot/class from few-shot set)
  - Reply:  return top-1 retrieved agent reply verbatim
  - Escalation: escalate if max cosine similarity < 0.40
"""

import json
import os
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

EMBED_MODEL   = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
FEW_SHOT_PATH = os.getenv("FEW_SHOT_DATA", "data/few_shot_train.json")
PAIRS_JSON    = "data/uber_pairs_full.json"

# ── Baseline 1 ────────────────────────────────────────────────────────────────

class TrivialBaseline:
    """Always predict most-common intent; reply with a fixed template."""

    TEMPLATE = ("Hi there! We're sorry for the trouble. Please DM us your account "
                "details and we'll sort this out right away. ^UberSupport")

    def __init__(self, few_shot_path: str = FEW_SHOT_PATH):
        with open(few_shot_path) as f:
            data = json.load(f)
        counts = Counter(d["label"] for d in data)
        self.most_common = counts.most_common(1)[0][0]
        print(f"[TrivialBaseline] most-common intent: {self.most_common}")

    def predict(self, texts: list[str]) -> list[dict]:
        return [
            {
                "intent":    self.most_common,
                "confidence": 1.0,
                "reply":     self.TEMPLATE,
                "action":    "auto",
                "escalate":  False,
                "reasons":   [],
            }
            for _ in texts
        ]

    def predict_one(self, text: str) -> dict:
        return self.predict([text])[0]


# ── Baseline 2 ────────────────────────────────────────────────────────────────

class CosineNNBaseline:
    """
    Nearest-neighbour intent classification using one embedding per class
    (centroid of few-shot examples).
    Reply = verbatim top-1 retrieved agent reply from full pairs.
    Escalate if max cosine sim < 0.40.
    """

    ESCALATION_THRESHOLD = 0.40

    def __init__(self, few_shot_path: str = FEW_SHOT_PATH,
                 pairs_path: str = PAIRS_JSON,
                 model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)

        # Build class centroids from few-shot examples
        with open(few_shot_path) as f:
            few_shot = json.load(f)

        by_label: dict[str, list[str]] = {}
        for item in few_shot:
            by_label.setdefault(item["label"], []).append(item["text"])

        self.labels: list[str] = sorted(by_label.keys())
        centroids = []
        for label in self.labels:
            embs = self.model.encode(
                by_label[label], normalize_embeddings=True, convert_to_numpy=True
            )
            centroids.append(embs.mean(axis=0))
        self.centroids = np.array(centroids)  # (n_classes, dim)

        # Build a mini FAISS index over real pairs for verbatim reply retrieval
        with open(pairs_path) as f:
            pairs = json.load(f)
        self.pairs = pairs
        self.pair_texts  = [p["customer_msg"] for p in pairs]
        self.pair_replies = [p["agent_reply"]  for p in pairs]

        pair_embs = self.model.encode(
            self.pair_texts, normalize_embeddings=True, convert_to_numpy=True,
            batch_size=256, show_progress_bar=True,
        )
        import faiss
        d = pair_embs.shape[1]
        self._pair_index = faiss.IndexFlatIP(d)
        self._pair_index.add(pair_embs.astype("float32"))
        print(f"[CosineNNBaseline] {len(self.labels)} classes, {len(pairs):,} pairs indexed")

    def predict(self, texts: list[str]) -> list[dict]:
        embs   = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        scores = embs @ self.centroids.T          # (N, n_classes)
        best_idx   = scores.argmax(axis=1)
        best_score = scores.max(axis=1)

        # Retrieve verbatim reply from top-1 pair
        pair_scores, pair_idxs = self._pair_index.search(
            embs.astype("float32"), 1
        )

        results = []
        for i, (b_idx, b_score) in enumerate(zip(best_idx, best_score)):
            intent     = self.labels[int(b_idx)]
            confidence = float(b_score)
            p_idx      = int(pair_idxs[i][0])
            reply      = self.pair_replies[p_idx] if p_idx >= 0 else self.FALLBACK_REPLY
            ret_score  = float(pair_scores[i][0]) if p_idx >= 0 else 0.0
            escalate   = ret_score < self.ESCALATION_THRESHOLD

            results.append({
                "intent":     intent,
                "confidence": confidence,
                "reply":      reply,
                "action":     "escalate" if escalate else "auto",
                "escalate":   escalate,
                "reasons":    [f"Low retrieval similarity ({ret_score:.2f})"]
                              if escalate else [],
                "top_retrieval": ret_score,
            })
        return results

    FALLBACK_REPLY = ("We're sorry for the inconvenience. Please DM us your details "
                      "and we'll help you right away.")

    def predict_one(self, text: str) -> dict:
        return self.predict([text])[0]


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    test_msgs = [
        "my driver cancelled twice and im still being charged",
        "I left my wallet in the uber",
        "the driver was rude and threatening",
        "app crashes when I try to request a ride",
        "great driver, very smooth ride!",
    ]

    print("\n=== Baseline 1 (Trivial) ===")
    b1 = TrivialBaseline()
    for msg, res in zip(test_msgs, b1.predict(test_msgs)):
        print(f"  {res['intent']:25s}  {msg[:60]}")

    print("\n=== Baseline 2 (Cosine NN) ===")
    b2 = CosineNNBaseline()
    for msg, res in zip(test_msgs, b2.predict(test_msgs)):
        print(f"  {res['intent']:25s} ({res['confidence']:.2f})  [{res['action']}]  {msg[:60]}")
        print(f"    → Reply: {res['reply'][:80]}")
