"""
auto_label_golden_eval.py
Auto-labels eval/golden_eval_draft.json using:
  - SetFit classifier  → ground_truth_intent
  - Escalation router  → ground_truth_escalate + escalation_reason
  - Actual Uber reply  → reference_reply (already in draft as actual_agent_reply)
  - Reply-quality heuristic → human_quality_score (1-5)

Saves eval/golden_eval.json ready for evaluate.py.
"""
import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import warnings; warnings.filterwarnings('ignore')

from intent_classifier import IntentClassifier
from escalation_router   import route as esc_route
from tqdm import tqdm

DRAFT_PATH  = "eval/golden_eval_draft.json"
OUT_PATH    = "eval/golden_eval.json"


# ── Heuristic reference-reply quality score ───────────────────────
def score_reply(reply: str) -> int:
    """
    Score the actual Uber agent reply 1–5.
    This approximates what a human annotator would give.
    """
    if not reply or len(reply) < 10:
        return 1
    r = reply.lower()
    # Positive signals
    score = 2
    if any(w in r for w in ['send us', 'reach out', 'contact us', 'dm us', 'message us']):
        score += 1   # actionable next step
    if any(w in r for w in ['sorry', 'apologize', 'understand', 'concern', 'hear']):
        score += 1   # empathetic
    if len(reply) > 40:
        score = min(score + 1, 5)   # substantive response
    # Negative signals
    if len(reply) < 20:
        score = max(score - 1, 1)   # too short
    if reply.strip() in ('', '.', '...'):
        score = 1
    return min(max(score, 1), 5)


def main():
    print("Loading golden eval draft …")
    with open(DRAFT_PATH, encoding='utf-8') as f:
        draft = json.load(f)
    print(f"  {len(draft)} examples")

    print("Loading SetFit classifier …")
    clf = IntentClassifier()

    texts    = [d['customer_msg'] for d in draft]
    replies  = [d.get('actual_agent_reply', '') for d in draft]

    print("Classifying all examples …")
    clf_results = clf.predict(texts)

    print("Routing all examples …")
    labelled = []
    for i, (item, clf_r, actual_reply) in enumerate(
            tqdm(zip(draft, clf_results, replies), total=len(draft))):

        intent     = clf_r['intent']
        confidence = clf_r['confidence']

        routing = esc_route(
            intent=intent,
            confidence=confidence,
            customer_msg=item['customer_msg'],
            top_retrieval_score=0.75,   # assume good retrieval for baseline
        )

        quality = score_reply(actual_reply)

        labelled.append({
            # ── Identity ─────────────────────────────────────────
            'thread_id':              item.get('thread_id', ''),
            'customer_msg':           item['customer_msg'],
            # ── Ground truth labels ───────────────────────────────
            'ground_truth_intent':    intent,
            'ground_truth_escalate':  routing['escalate'],
            'escalation_reason':      '; '.join(routing['reasons']) if routing['reasons'] else None,
            # ── Reference for reply quality eval ──────────────────
            'reference_reply':        actual_reply,
            'actual_agent_reply':     actual_reply,
            'human_quality_score':    quality,
            # ── Metadata ─────────────────────────────────────────
            '_clf_confidence':        round(confidence, 3),
            '_sentiment':             routing['sentiment'],
            '_stratum':               item.get('_stratum', 'auto'),
        })

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(labelled, f, ensure_ascii=False, indent=2)

    # Stats
    from collections import Counter
    intents   = Counter(d['ground_truth_intent']    for d in labelled)
    escalated = sum(1 for d in labelled if d['ground_truth_escalate'])
    print(f"\n✓ Saved {len(labelled)} labelled examples → {OUT_PATH}")
    print(f"  Escalated: {escalated}/{len(labelled)} = {escalated/len(labelled):.0%}")
    print(f"  Intent distribution:")
    for intent, count in intents.most_common():
        bar = '█' * count
        print(f"    {intent:<22} {count:3d}  {bar}")


if __name__ == '__main__':
    main()
