"""
agent.py — Main Uber Support AI Agent pipeline.

Usage:
  # Single message
  python src/agent.py --message "my driver cancelled and I was charged"

  # Batch (JSON array of strings)
  python src/agent.py --batch data/test_messages.json

  # Interactive demo
  python src/agent.py --interactive
"""

import json
import os
import sys
import argparse
from typing import Optional

# Ensure src/ is importable when running as `python src/agent.py` or from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Lazy imports to allow fast --help
_clf = None
_ret = None


def _load_components():
    global _clf, _ret
    if _clf is None:
        from intent_classifier import IntentClassifier
        _clf = IntentClassifier()
    if _ret is None:
        from rag_indexer import RAGRetriever
        _ret = RAGRetriever()
    return _clf, _ret


def run_pipeline(
    customer_msg:     str,
    use_llm_override: bool = False,
    verbose:          bool = True,
) -> dict:
    """
    Full pipeline: message → intent → RAG → reply → escalation decision.

    Returns:
        dict with keys: intent, confidence, reply, action, reasons, retrieved, sentiment
    """
    clf, ret = _load_components()

    # ── Stage 1: Intent Classification ───────────────────────────────────────
    intent_result = clf.predict_one(customer_msg)
    intent     = intent_result["intent"]
    confidence = intent_result["confidence"]

    # ── Stage 2: RAG Retrieval ────────────────────────────────────────────────
    retrieved = ret.retrieve(customer_msg)
    top_score = retrieved[0]["retrieval_score"] if retrieved else 0.0

    # ── Stage 3: Reply Generation ─────────────────────────────────────────────
    from reply_generator import generate_reply
    gen_result = generate_reply(
        customer_msg=customer_msg,
        intent=intent,
        confidence=confidence,
        retrieved=retrieved,
    )
    draft_reply = gen_result["reply"]

    # ── Stage 4: Escalation Routing ───────────────────────────────────────────
    from escalation_router import route
    routing = route(
        intent=intent,
        confidence=confidence,
        customer_msg=customer_msg,
        top_retrieval_score=top_score,
        use_llm_override=use_llm_override,
        draft_reply=draft_reply,
    )

    output = {
        "customer_msg":    customer_msg,
        "intent":          intent,
        "confidence":      confidence,
        "second_intent":   intent_result.get("second_intent"),
        "second_conf":     intent_result.get("second_conf"),
        "reply":           draft_reply,
        "action":          routing["action"],
        "escalate":        routing["escalate"],
        "reasons":         routing["reasons"],
        "sentiment":       routing["sentiment"],
        "top_retrieval":   top_score,
        "model_used":      gen_result.get("model_used", "unknown"),
        "n_retrieved":     gen_result.get("n_retrieved", 0),
    }

    if verbose:
        _pretty_print(output)

    return output


def _pretty_print(result: dict):
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    action_color = "green" if result["action"] == "auto" else "red"

    console.print()
    console.print(Panel(
        f"[bold]{result['customer_msg']}[/bold]",
        title="📨 Customer Message", border_style="blue"
    ))

    console.print(
        f"  [cyan]Intent[/cyan]     : [bold]{result['intent']}[/bold] "
        f"({result['confidence']:.0%}) — runner-up: "
        f"{result['second_intent']} ({result['second_conf']:.0%})"
    )
    console.print(f"  [cyan]Sentiment[/cyan]  : {result['sentiment']:+.2f}")
    console.print(f"  [cyan]RAG score[/cyan]  : {result['top_retrieval']:.2f}")
    console.print()
    console.print(Panel(
        result["reply"],
        title="✉️  Draft Reply", border_style="green"
    ))
    action_text = (
        "✅ AUTO-HANDLE" if result["action"] == "auto"
        else f"🚨 ESCALATE — {'; '.join(result['reasons'])}"
    )
    console.print(Panel(
        f"[bold {action_color}]{action_text}[/bold {action_color}]",
        title="⚖️  Decision", border_style=action_color
    ))
    console.print()


# ------------------------------------------------------------------
# Batch inference (for evaluation harness)
# ------------------------------------------------------------------

def run_batch(messages: list[str], verbose: bool = False) -> list[dict]:
    results = []
    from tqdm import tqdm
    for msg in tqdm(messages, desc="Processing messages"):
        try:
            r = run_pipeline(msg, verbose=verbose)
        except Exception as e:
            r = {"customer_msg": msg, "error": str(e)}
        results.append(r)
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Uber Support AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python src/agent.py --message "my driver never showed up and I was charged"
  python src/agent.py --batch data/test_messages.json --out eval/results/batch.json
  python src/agent.py --interactive
"""
    )
    parser.add_argument("--message",     type=str,   help="Single customer message to process")
    parser.add_argument("--batch",       type=str,   help="Path to JSON array of messages")
    parser.add_argument("--out",         type=str,   default=None, help="Output JSON for batch results")
    parser.add_argument("--interactive", action="store_true", help="Interactive demo mode")
    parser.add_argument("--llm-override",action="store_true", help="Enable LLM escalation override")
    args = parser.parse_args()

    if args.message:
        run_pipeline(args.message, use_llm_override=args.llm_override)

    elif args.batch:
        with open(args.batch, encoding="utf-8") as f:
            messages = json.load(f)
        results = run_batch(messages)
        out_path = args.out or "eval/results/batch_results.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved → {out_path}")

    elif args.interactive:
        print("Uber Support AI Agent — Interactive Demo")
        print("Type a customer message and press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                msg = input("Customer: ").strip()
                if msg:
                    run_pipeline(msg, use_llm_override=args.llm_override)
            except KeyboardInterrupt:
                print("\nExiting.")
                break

    else:
        parser.print_help()


if __name__ == "__main__":
    # Ensure src/ is on Python path when running directly
    sys.path.insert(0, os.path.dirname(__file__))
    main()
