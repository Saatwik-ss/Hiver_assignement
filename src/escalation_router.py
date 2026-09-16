"""
escalation_router.py — Multi-signal escalation/routing decision for Uber_Support.

Signals (in priority order):
  1. Hard rules: high-risk intent, dangerous keywords
  2. Soft rules: low confidence, very negative sentiment, poor RAG retrieval
  3. LLM override (optional, for borderline cases)

Returns: {"escalate": bool, "reasons": [str], "action": "auto" | "escalate"}
"""

import os
import re
from typing import Optional

# Threshold defaults (override via env or direct args)
CONFIDENCE_THRESHOLD      = float(os.getenv("CONFIDENCE_THRESHOLD",      0.65))
SENTIMENT_THRESHOLD       = float(os.getenv("SENTIMENT_THRESHOLD",        -0.65))
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD",  0.48))

# ------------------------------------------------------------------
# Uber-specific escalation rules
# ------------------------------------------------------------------

# Intents that ALWAYS trigger human review
HIGH_RISK_INTENTS = {
    "safety_incident",      # accidents, assault, harassment — no exceptions
    "payment_billing",      # fraud, overcharge, chargeback risk
    "account_access",       # account hacked/compromised
    "driver_behavior",      # rude/threatening — may need HR involvement
}

# Safety-critical: escalate immediately, no further checks
CRITICAL_SAFETY_INTENTS = {"safety_incident"}

# Keywords that trigger immediate escalation regardless of confidence
ESCALATION_KEYWORDS = [
    # Legal / fraud
    "lawsuit", "attorney", "lawyer", "legal action", "sue", "court",
    "fraud", "scam", "stolen", "unauthorized", "chargeback", "dispute",
    # Safety
    "accident", "crash", "injured", "hurt", "assault", "harassed", "harassment",
    "threatening", "threatened", "unsafe", "emergency", "police", "911",
    # High-urgency financial (must be clearly a billing dispute, not a trip complaint)
    "refund now", "charged wrong", "double charged", "charged twice",
    # Account security
    "hacked", "compromised", "someone else", "not me",
]

# Compile keyword pattern (whole-word, case-insensitive)
_KW_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in ESCALATION_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)


# ------------------------------------------------------------------
# Sentiment (VADER — Twitter-tuned, no API)
# ------------------------------------------------------------------

_vader = None

def _get_sentiment(text: str) -> float:
    """Return compound VADER score in [-1, +1]."""
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader.polarity_scores(text)["compound"]


# ------------------------------------------------------------------
# Main router
# ------------------------------------------------------------------

def route(
    intent:           str,
    confidence:       float,
    customer_msg:     str,
    top_retrieval_score: float,
    sentiment:        Optional[float] = None,
    use_llm_override: bool = False,
    draft_reply:      str = "",
) -> dict:
    """
    Determine whether to auto-handle or escalate.

    Args:
        intent:              Predicted intent label
        confidence:          SetFit softmax confidence (0–1)
        customer_msg:        Raw customer message text
        top_retrieval_score: Max cosine similarity from FAISS retrieval
        sentiment:           Pre-computed VADER score (computed here if None)
        use_llm_override:    Call LLM judge for borderline cases
        draft_reply:         Generated draft reply (needed for LLM override)

    Returns:
        dict with keys: escalate (bool), action (str), reasons (list[str]),
                        sentiment (float)
    """
    reasons: list[str] = []

    # Compute sentiment if not provided
    if sentiment is None:
        sentiment = _get_sentiment(customer_msg)

    # ── Hard rules (always escalate, no soft overrides) ──────────────────────

    if intent in CRITICAL_SAFETY_INTENTS:
        reasons.append(f"CRITICAL: Safety-related intent ({intent}) — human required")
        return _result(True, reasons, sentiment, confidence)

    matched_kws = _KW_PATTERN.findall(customer_msg)
    if matched_kws:
        unique = list(dict.fromkeys(kw.lower() for kw in matched_kws))
        reasons.append(f"Flagged keywords: {', '.join(unique[:5])}")
        return _result(True, reasons, sentiment, confidence)

    # ── Soft rules (accumulate; any single rule triggers escalation) ──────────

    if intent in HIGH_RISK_INTENTS:
        reasons.append(f"High-risk intent: {intent}")

    if confidence < CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Low classifier confidence ({confidence:.0%} < {CONFIDENCE_THRESHOLD:.0%})"
        )

    if sentiment < SENTIMENT_THRESHOLD:
        reasons.append(
            f"High negative sentiment (VADER={sentiment:.2f} < {SENTIMENT_THRESHOLD})"
        )

    if top_retrieval_score < RETRIEVAL_SCORE_THRESHOLD:
        reasons.append(
            f"No similar resolution found (score={top_retrieval_score:.2f} "
            f"< {RETRIEVAL_SCORE_THRESHOLD})"
        )

    # ── LLM override for borderline cases ────────────────────────────────────
    # Only call LLM if: no hard-rule trigger, but 1-2 soft signals triggered
    if reasons and use_llm_override and draft_reply:
        from reply_generator import judge_escalation
        llm_judgment = judge_escalation(
            customer_msg=customer_msg,
            draft_reply=draft_reply,
            intent=intent,
            confidence=confidence,
        )
        if llm_judgment.get("decision") == "escalate":
            reasons.append(
                f"LLM judge: escalate ({llm_judgment.get('reason', '')})"
            )
        else:
            # LLM overrides soft signals — auto-handle
            reasons_copy = reasons.copy()
            reasons = []  # clear soft triggers
            reasons.append(
                f"LLM judge: auto-handle (overrode: {'; '.join(reasons_copy)})"
            )

    escalate = bool(reasons)
    return _result(escalate, reasons, sentiment, confidence)


def _result(escalate: bool, reasons: list[str], sentiment: float,
            confidence: float) -> dict:
    return {
        "escalate":   escalate,
        "action":     "escalate" if escalate else "auto",
        "reasons":    reasons,
        "sentiment":  round(sentiment, 3),
        "confidence": round(confidence, 3),
    }


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # (intent, confidence, message, retrieval_score)
        ("trip_issue",      0.91, "the driver went the wrong way totally",         0.72),
        ("safety_incident", 0.88, "driver was threatening and aggressive",         0.55),
        ("payment_billing", 0.78, "I was charged twice, this is fraud!",           0.61),
        ("app_technical",   0.55, "app keeps crashing when i try to book",         0.63),
        ("lost_item",       0.89, "I left my phone in the car 10 minutes ago",     0.80),
        ("ride_cancellation", 0.70, "driver cancelled again after I waited 20 min", 0.38),
        ("positive_feedback", 0.95, "amazing driver, very helpful!",               0.88),
    ]

    print(f"{'Message':<50} {'Intent':<20} {'Action':<10} Reasons")
    print("-" * 110)
    for intent, conf, msg, ret_score in test_cases:
        result = route(intent, conf, msg, ret_score)
        reasons_short = "; ".join(result["reasons"])[:60] if result["reasons"] else "—"
        action = result["action"].upper()
        print(f"{msg:<50} {intent:<20} {action:<10} {reasons_short}")
