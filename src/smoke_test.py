"""
smoke_test.py — End-to-end smoke test for the full pipeline (no LLM needed).
Tests: SetFit classifier + FAISS retrieval + escalation router.
Run: python src/smoke_test.py
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from intent_classifier import IntentClassifier
from rag_indexer import RAGRetriever
from escalation_router import route

print("=" * 70)
print("UBER SUPPORT AGENT — SMOKE TEST (no LLM required)")
print("=" * 70)

# ── 1. Load models ────────────────────────────────────────────────
print("\n[1/3] Loading models...")
clf = IntentClassifier()
ret = RAGRetriever()
print("  ✓ SetFit classifier and FAISS retriever loaded.")

# ── 2. Test cases: (message, expected_intent, should_escalate) ───
test_cases = [
    ("my driver cancelled twice and im still charged a fee",    "ride_cancellation", False),
    ("I was assaulted by my driver, this is an emergency",      "safety_incident",   True),
    ("app crashes every time I try to request a ride",          "app_technical",     False),
    ("I left my backpack in the car, please help",              "lost_item",         False),
    ("driver went completely the wrong route and overcharged",   "trip_issue",        False),
    ("I was charged twice for the same trip, this is fraud",    "payment_billing",   True),
    ("can't log into my uber account, keeps saying error",      "account_access",    True),
    ("driver was rude and made threatening comments",           "driver_behavior",   True),
    ("uber eats delivered completely wrong order",              "eats_order",        False),
    ("amazing driver, very professional and on time!",          "positive_feedback", False),
    ("been waiting 40 min, driver not moving on map",           "wait_time",         False),
    ("surge is 5x normal price, can't afford it",              "ride_cancellation", False),
]

print(f"\n[2/3] Intent Classification ({len(test_cases)} test cases)...")
print(f"\n  {'Message':<48} {'Expected':<22} {'Predicted':<22} {'Conf':>5}  {'OK?'}")
print("  " + "-" * 105)

correct = 0
for msg, expected, _ in test_cases:
    r = clf.predict_one(msg)
    ok = "✓" if r["intent"] == expected else "✗"
    if r["intent"] == expected:
        correct += 1
    print(f"  {msg[:46]:<48} {expected:<22} {r['intent']:<22} {r['confidence']:>4.0%}  {ok}")

accuracy = correct / len(test_cases)
print(f"\n  Spot-check accuracy: {correct}/{len(test_cases)} = {accuracy:.0%}")

# ── 3. Retrieval + escalation routing test ───────────────────────
print(f"\n[3/3] Full Pipeline (Intent → RAG → Escalation)...")
print(f"\n  {'Message':<48} {'Intent':<22} {'Action':<10} {'RAG':>5}  Reasons")
print("  " + "-" * 115)

for msg, expected, should_escalate in test_cases:
    r = clf.predict_one(msg)
    retrieved = ret.retrieve(msg, top_k=3)
    top_score = retrieved[0]["retrieval_score"] if retrieved else 0.0
    routing = route(
        intent=r["intent"],
        confidence=r["confidence"],
        customer_msg=msg,
        top_retrieval_score=top_score,
    )
    action = routing["action"].upper()
    reasons = "; ".join(routing["reasons"])[:45] if routing["reasons"] else "—"
    esc_ok = "✓" if routing["escalate"] == should_escalate else "!"
    print(f"  {msg[:46]:<48} {r['intent']:<22} {action:<10} {top_score:>4.2f}  {reasons} {esc_ok}")

print("\n" + "=" * 70)
print("Smoke test complete. Review ✗ predictions and ! routing decisions above.")
print("If intent accuracy < 70%, rerun: python src/intent_classifier.py --train")
print("=" * 70)
