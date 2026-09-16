"""
build_golden_eval.py — Build a stratified golden evaluation set (150-250 examples).

Sampling strategy:
  - 60 "easy" examples (high retrieval score, clear intent)
  - 80 "medium" examples (moderate, borderline cases)
  - 40 "hard" examples (ambiguous, low confidence, escalation-worthy)
  - 20 "positive_feedback" examples

Each example is pre-populated with the agent's prediction so a human annotator
only needs to fill in: ground_truth_intent, ground_truth_escalate, reference_reply,
human_quality_score.

Run:
  python src/build_golden_eval.py
  # Then open eval/golden_eval_draft.json and fill in labels
  # Save final version to eval/golden_eval.json
"""

import json
import os
import sys
import random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

PAIRS_JSON     = "data/uber_pairs_full.json"
CLUSTERS_JSON  = "data/clusters.json"
OUT_DRAFT      = "eval/golden_eval_draft.json"
OUT_SCHEMA     = "eval/golden_eval_schema.json"
SEED           = 42

random.seed(SEED)


def load_clustered_pairs() -> list[dict]:
    """Load pairs with cluster/intent assignments from intent_discovery output."""
    if os.path.exists(CLUSTERS_JSON):
        with open(CLUSTERS_JSON) as f:
            return json.load(f)
    # Fall back to raw pairs without intent labels
    with open(PAIRS_JSON) as f:
        pairs = json.load(f)
    # Tag everything as "unlabelled" for human annotation
    return [{**p, "intent": "unlabelled", "cluster_id": -1} for p in pairs]


def sample_stratified(pairs: list[dict]) -> list[dict]:
    """Sample across difficulty strata."""
    by_intent = defaultdict(list)
    for p in pairs:
        intent = p.get("intent", "unknown")
        if intent not in ("-1", "unknown", -1):
            by_intent[intent].append(p)

    # Sort by retrieval proxy (use customer_msg length as crude difficulty proxy)
    for intent in by_intent:
        by_intent[intent].sort(key=lambda x: len(x["customer_msg"]))

    sampled = []
    intents  = [k for k in by_intent if k not in ("positive_feedback", "unknown")]
    pos_pairs = by_intent.get("positive_feedback", [])

    # Easy: shortest messages (clear, simple)
    easy_per_intent = max(1, 60 // max(len(intents), 1))
    for intent, ps in by_intent.items():
        if intent == "positive_feedback":
            continue
        sampled.extend(random.sample(ps[:len(ps)//3 + 1],
                                     min(easy_per_intent, len(ps)//3 + 1)))

    # Medium: middle band
    medium_per_intent = max(1, 80 // max(len(intents), 1))
    for intent, ps in by_intent.items():
        if intent == "positive_feedback":
            continue
        mid = ps[len(ps)//3: 2*len(ps)//3]
        sampled.extend(random.sample(mid, min(medium_per_intent, len(mid))))

    # Hard: longest messages + billing/safety-heavy
    hard_intents = ["payment_billing", "safety_incident", "driver_behavior", "account_access"]
    hard_per = max(1, 40 // max(len(hard_intents), 1))
    for intent in hard_intents:
        ps = by_intent.get(intent, [])
        tail = ps[2*len(ps)//3:]
        sampled.extend(random.sample(tail, min(hard_per, len(tail))))

    # Positive feedback
    sampled.extend(random.sample(pos_pairs, min(20, len(pos_pairs))))

    # Deduplicate by thread_id
    seen, unique = set(), []
    for p in sampled:
        tid = p.get("thread_id", p.get("customer_msg", ""))
        if tid not in seen:
            seen.add(tid)
            unique.append(p)

    return unique[:250]  # cap at 250


def build_draft(pairs: list[dict]) -> list[dict]:
    """Wrap each pair in the golden eval schema."""
    draft = []
    for p in pairs:
        draft.append({
            # ── Read-only context ───────────────────────────────
            "thread_id":         p.get("thread_id", ""),
            "customer_msg":      p["customer_msg"],
            "customer_raw":      p.get("customer_raw", ""),
            "actual_agent_reply": p.get("agent_reply", ""),

            # ── Labels to fill in (HUMAN ANNOTATION REQUIRED) ──
            # Predicted intent from clustering — VERIFY and correct this
            "ground_truth_intent":   p.get("intent", "TODO"),
            # Should this be escalated? true/false
            "ground_truth_escalate": None,           # TODO
            # Why escalate (or null if auto)
            "escalation_reason":     None,           # TODO
            # Ideal reply you'd want the agent to send (can be same as actual_agent_reply)
            "reference_reply":       p.get("agent_reply", "TODO"),  # REVIEW
            # Quality of actual_agent_reply (1=very poor, 5=excellent)
            "human_quality_score":   None,           # TODO (1-5)
            # Optional: stratum tag for analysis
            "_stratum":              "auto",
        })
    return draft


def write_schema():
    """Write a schema file documenting expected fields."""
    schema = {
        "description": "Golden evaluation set for Uber Support AI Agent",
        "fields": {
            "thread_id":             "Unique conversation ID from TWCS",
            "customer_msg":          "Cleaned customer message (no @mentions/URLs)",
            "customer_raw":          "Original raw tweet text",
            "actual_agent_reply":    "What Uber actually replied in the dataset",
            "ground_truth_intent":   "Correct intent label (from your defined taxonomy)",
            "ground_truth_escalate": "bool — should a human agent handle this?",
            "escalation_reason":     "String reason for escalation (null if auto)",
            "reference_reply":       "Ideal reply you'd want the agent to send (can match actual_agent_reply)",
            "human_quality_score":   "int 1-5 — quality of actual_agent_reply (1=terrible, 5=excellent)",
            "_stratum":              "easy/medium/hard/positive_feedback — for analysis",
        },
        "intent_taxonomy": {
            "trip_issue":         "Wrong route, bad GPS, navigation, driver went wrong way",
            "payment_billing":    "Overcharge, refund, promo not applied, double charge",
            "driver_behavior":    "Rude, unprofessional, aggressive driver",
            "safety_incident":    "Accident, assault, harassment, threatening — ALWAYS ESCALATE",
            "account_access":     "Login issues, account suspended, password reset",
            "app_technical":      "App crash, surge pricing bug, booking failure",
            "ride_cancellation":  "Driver cancelled, can't find driver, excessive surge",
            "lost_item":          "Left item in car, need to contact driver",
            "wait_time":          "Driver taking too long, ETA wrong",
            "eats_order":         "Uber Eats: wrong order, late delivery, missing items",
            "general_complaint":  "Vague frustration, unclear intent",
            "positive_feedback":  "Praise, compliments — auto-handle with thank-you",
        },
    }
    with open(OUT_SCHEMA, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Schema saved → {OUT_SCHEMA}")


if __name__ == "__main__":
    os.makedirs("eval", exist_ok=True)
    pairs   = load_clustered_pairs()
    sampled = sample_stratified(pairs)
    draft   = build_draft(sampled)

    with open(OUT_DRAFT, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)

    write_schema()
    print(f"\nGolden eval draft: {len(draft)} examples → {OUT_DRAFT}")
    print("Next steps:")
    print("  1. Open eval/golden_eval_draft.json")
    print("  2. Verify/correct 'ground_truth_intent' for each example")
    print("  3. Fill in 'ground_truth_escalate' (true/false)")
    print("  4. Set 'human_quality_score' (1-5) for actual_agent_reply quality")
    print("  5. Optionally improve 'reference_reply'")
    print("  6. Save completed file as eval/golden_eval.json")
    print("  7. Run: python src/evaluate.py --golden eval/golden_eval.json")
