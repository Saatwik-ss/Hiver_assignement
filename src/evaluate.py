"""
evaluate.py — Automated evaluation harness for the Uber Support AI Agent.

Runs three systems (Baseline 1, Baseline 2, Our Agent) against the golden eval set
and produces a full results table with:
  - Intent accuracy + macro-F1
  - Escalation precision, recall, F1
  - ROUGE-L, BERTScore
  - G-Eval (LLM-as-Judge) weighted scores
  - Human-LLM correlation (Spearman) for the judge

Usage:
  python src/evaluate.py --golden eval/golden_eval.json --out eval/results/
  python src/evaluate.py --golden eval/golden_eval.json --skip-bert --skip-judge
"""

import json
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

GOLDEN_PATH = os.getenv("GOLDEN_EVAL_PATH", "eval/golden_eval.json")
RESULTS_DIR = "eval/results"


# ------------------------------------------------------------------
# Golden eval loader
# ------------------------------------------------------------------

def load_golden(path: str = GOLDEN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} golden examples from {path}")
    return data


# ------------------------------------------------------------------
# Metric computations
# ------------------------------------------------------------------

def intent_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "macro_f1":  round(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0), 4),
    }


def escalation_metrics(esc_true: list[bool], esc_pred: list[bool]) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score
    return {
        "precision": round(precision_score(esc_true, esc_pred, zero_division=0), 4),
        "recall":    round(recall_score(esc_true, esc_pred, zero_division=0), 4),
        "f1":        round(f1_score(esc_true, esc_pred, zero_division=0), 4),
    }


def rouge_l_score(generated: list[str], reference: list[str]) -> float:
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, gen)["rougeL"].fmeasure
        for gen, ref in zip(generated, reference)
    ]
    return round(float(np.mean(scores)), 4)


def bert_score_avg(generated: list[str], reference: list[str]) -> float:
    from bert_score import score as bscore
    P, R, F = bscore(generated, reference, lang="en", verbose=False, batch_size=16)
    return round(float(F.mean()), 4)


def geval_batch(golden: list[dict], generated_replies: list[str]) -> list[dict]:
    """Run G-Eval judge on each example; returns list of score dicts."""
    from reply_generator import judge_reply
    scores = []
    for item, gen_reply in tqdm(zip(golden, generated_replies),
                                total=len(golden), desc="G-Eval judging"):
        s = judge_reply(
            customer_msg=item["customer_msg"],
            generated_reply=gen_reply,
            reference_reply=item.get("reference_reply", ""),
            intent=item["ground_truth_intent"],
        )
        scores.append(s)
    return scores


def geval_summary(scores: list[dict]) -> dict:
    """Aggregate G-Eval scores across examples."""
    valid = [s for s in scores if "weighted_score" in s]
    if not valid:
        return {}
    dims = ["relevance", "groundedness", "empathy", "actionability", "conciseness"]
    result = {
        "weighted_score": round(np.mean([s["weighted_score"] for s in valid]), 3),
    }
    for dim in dims:
        vals = [s[dim] for s in valid if dim in s]
        result[dim] = round(np.mean(vals), 3) if vals else None
    return result


def spearman_correlation(human_scores: list[float], llm_scores: list[float]) -> float:
    """Compute Spearman ρ between human and LLM judge scores."""
    from scipy.stats import spearmanr
    rho, p = spearmanr(human_scores, llm_scores)
    return round(float(rho), 4)


# ------------------------------------------------------------------
# System runners
# ------------------------------------------------------------------

def run_system_baseline1(golden: list[dict]) -> list[dict]:
    from baselines import TrivialBaseline
    b1 = TrivialBaseline()
    msgs    = [g["customer_msg"] for g in golden]
    results = b1.predict(msgs)
    return results


def run_system_baseline2(golden: list[dict]) -> list[dict]:
    from baselines import CosineNNBaseline
    b2 = CosineNNBaseline()
    msgs    = [g["customer_msg"] for g in golden]
    results = b2.predict(msgs)
    return results


def run_system_agent(golden: list[dict]) -> list[dict]:
    from agent import run_pipeline
    results = []
    for item in tqdm(golden, desc="Running agent"):
        try:
            r = run_pipeline(item["customer_msg"], verbose=False)
        except Exception as e:
            r = {"intent": "unknown", "confidence": 0.0,
                 "reply": "", "action": "auto", "escalate": False,
                 "reasons": [], "sentiment": 0.0, "top_retrieval": 0.0,
                 "error": str(e)}
        results.append(r)
    return results


# ------------------------------------------------------------------
# Full evaluation loop
# ------------------------------------------------------------------

def evaluate(golden: list[dict], system_results: list[dict],
             system_name: str, skip_bert: bool = False,
             skip_judge: bool = False) -> dict:
    """Evaluate one system against the golden set."""
    y_true       = [g["ground_truth_intent"]   for g in golden]
    esc_true     = [g["ground_truth_escalate"] for g in golden]
    refs         = [g.get("reference_reply", "") for g in golden]
    human_scores = [g.get("human_quality_score") for g in golden]

    y_pred   = [r.get("intent",  "unknown") for r in system_results]
    esc_pred = [r.get("escalate", False)    for r in system_results]
    gen      = [r.get("reply",   "")        for r in system_results]

    metrics: dict = {
        "system": system_name,
        "n":      len(golden),
    }

    # Intent metrics
    metrics["intent"] = intent_metrics(y_true, y_pred)

    # Escalation metrics
    metrics["escalation"] = escalation_metrics(esc_true, esc_pred)

    # Reply quality
    metrics["rouge_l"]    = rouge_l_score(gen, refs)
    if not skip_bert:
        metrics["bertscore"] = bert_score_avg(gen, refs)

    # G-Eval (LLM-as-Judge)
    if not skip_judge:
        judge_scores = geval_batch(golden, gen)
        metrics["geval"] = geval_summary(judge_scores)
        # Human-LLM correlation (only if human scores exist)
        valid_pairs = [
            (h, s["weighted_score"])
            for h, s in zip(human_scores, judge_scores)
            if h is not None and "weighted_score" in s
        ]
        if len(valid_pairs) >= 10:
            hs, ls = zip(*valid_pairs)
            metrics["human_llm_spearman"] = spearman_correlation(list(hs), list(ls))

    return metrics


# ------------------------------------------------------------------
# Results display
# ------------------------------------------------------------------

def print_results_table(all_metrics: list[dict]):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Evaluation Results — Uber Support Agent")

    table.add_column("System",          style="cyan",  no_wrap=True)
    table.add_column("Intent Acc",      justify="right")
    table.add_column("Intent F1",       justify="right")
    table.add_column("Esc Precision",   justify="right")
    table.add_column("Esc Recall",      justify="right")
    table.add_column("Esc F1",          justify="right")
    table.add_column("ROUGE-L",         justify="right")
    table.add_column("BERTScore",       justify="right")
    table.add_column("G-Eval",          justify="right")
    table.add_column("H-LLM ρ",         justify="right")

    for m in all_metrics:
        intent  = m.get("intent", {})
        esc     = m.get("escalation", {})
        geval   = m.get("geval", {})
        table.add_row(
            m["system"],
            f"{intent.get('accuracy', '-'):.1%}" if intent else "-",
            f"{intent.get('macro_f1', '-'):.3f}"  if intent else "-",
            f"{esc.get('precision', '-'):.3f}"     if esc    else "-",
            f"{esc.get('recall', '-'):.3f}"        if esc    else "-",
            f"{esc.get('f1', '-'):.3f}"            if esc    else "-",
            f"{m.get('rouge_l', '-'):.3f}",
            f"{m.get('bertscore', '-'):.3f}" if "bertscore" in m else "-",
            f"{geval.get('weighted_score', '-'):.2f}" if geval else "-",
            f"{m.get('human_llm_spearman', '-'):.3f}"
            if "human_llm_spearman" in m else "-",
        )

    console.print(table)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluation harness")
    parser.add_argument("--golden",      default=GOLDEN_PATH, help="Golden eval JSON")
    parser.add_argument("--out",         default=RESULTS_DIR,  help="Output directory")
    parser.add_argument("--skip-bert",   action="store_true",  help="Skip BERTScore (slow)")
    parser.add_argument("--skip-judge",  action="store_true",  help="Skip G-Eval judge (API calls)")
    parser.add_argument("--system",      default="all",
                        choices=["all", "b1", "b2", "agent"],
                        help="Which system to evaluate")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    os.makedirs(args.out, exist_ok=True)
    all_metrics = []

    systems = {
        "b1":    ("Baseline 1 (Trivial)",    run_system_baseline1),
        "b2":    ("Baseline 2 (Cosine NN)",  run_system_baseline2),
        "agent": ("Our Agent (SetFit+RAG)",  run_system_agent),
    }

    to_run = systems.keys() if args.system == "all" else [args.system]

    for key in to_run:
        name, runner = systems[key]
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print('='*60)

        results = runner(golden)

        # Save raw predictions
        raw_path = os.path.join(args.out, f"{key}_predictions.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        metrics = evaluate(
            golden=golden,
            system_results=results,
            system_name=name,
            skip_bert=args.skip_bert,
            skip_judge=args.skip_judge,
        )
        all_metrics.append(metrics)

        # Save metrics
        metrics_path = os.path.join(args.out, f"{key}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics saved → {metrics_path}")

    print_results_table(all_metrics)

    # Save combined summary
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nFull summary → {summary_path}")


if __name__ == "__main__":
    main()
