"""
reply_generator.py — Draft customer support replies using Groq (primary) or Ollama (fallback).

Groq models (free tier):
  - llama-3.1-8b-instant     : fast generation, good quality
  - llama-3.3-70b-versatile  : best quality, used for judging

Ollama fallback:
  - llama3.2:3b              : local, zero cost, offline-capable
"""

import os
import json
from typing import Optional

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_GEN_MODEL     = os.getenv("GROQ_GENERATION_MODEL", "llama-3.1-8b-instant")
GROQ_JUDGE_MODEL   = os.getenv("GROQ_JUDGE_MODEL",      "llama-3.3-70b-versatile")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL",           "llama3.2:3b")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",        "http://localhost:11434")

# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

REPLY_SYSTEM = """You are a customer support agent for Uber. You help customers \
resolve issues with rides, payments, driver behaviour, and account problems.
Write concise, empathetic replies in Uber's friendly but professional tone."""

REPLY_PROMPT = """\
## Customer Message
{customer_msg}

## Detected Intent
{intent} (confidence: {confidence:.0%})

## Relevant Historical Resolutions
Below are {n_examples} examples of how Uber has resolved similar issues:
{examples_block}

## Your Task
Draft a helpful Twitter reply that:
1. Directly addresses the customer's specific issue
2. Matches the resolution patterns shown above (do NOT invent policies)
3. Is concise — ideally under 280 characters
4. Is empathetic but not over-apologetic
5. Ends with a clear next step when appropriate

Reply (just the text, no preamble):"""


JUDGE_SYSTEM = """You are an expert evaluator of customer support quality. \
Rate replies on 5 dimensions using 1–5 Likert scales. Output valid JSON only."""

JUDGE_PROMPT = """\
Evaluate this Uber customer support reply.

## Customer Message
{customer_msg}

## Agent Reply (to evaluate)
{generated_reply}

## Reference Reply (actual Uber response, for calibration)
{reference_reply}

## Detected Intent
{intent}

Rate the agent reply on each dimension (1=very poor, 5=excellent):

1. RELEVANCE (0.30 weight): Does it address the actual issue?
2. GROUNDEDNESS (0.25 weight): Is it grounded in real Uber policies/processes?
3. EMPATHY (0.20 weight): Does it match Uber's empathetic, professional tone?
4. ACTIONABILITY (0.15 weight): Does it give the customer something concrete to do?
5. CONCISENESS (0.10 weight): Is it Twitter-appropriate and concise?

Think step by step, then output JSON exactly in this format:
{{
  "reasoning": "...",
  "relevance": X,
  "groundedness": X,
  "empathy": X,
  "actionability": X,
  "conciseness": X,
  "weighted_score": X.X
}}"""

ESCALATION_JUDGE_PROMPT = """\
Should this customer message be auto-handled or escalated to a human agent?

## Customer Message
{customer_msg}

## Proposed Reply
{draft_reply}

## Intent: {intent} (confidence: {confidence:.0%})

Escalate if:
- Billing dispute, fraud, or legal threat
- Safety incident (accident, harassment, assault)
- Account suspension requiring human review
- Proposed reply seems factually wrong or unhelpful
- Customer is highly distressed

Output JSON:
{{"decision": "auto" | "escalate", "reason": "...", "confidence": 0.0-1.0}}"""


# ------------------------------------------------------------------
# Groq client
# ------------------------------------------------------------------

def _groq_chat(system: str, user: str, model: str,
               max_tokens: int = 400, temperature: float = 0.3) -> str:
    """Call Groq chat completion API; returns reply text."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ------------------------------------------------------------------
# Ollama client (fallback)
# ------------------------------------------------------------------

def _ollama_chat(system: str, user: str, model: str = OLLAMA_MODEL,
                 max_tokens: int = 400) -> str:
    """Call local Ollama; returns reply text."""
    import ollama
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options={"num_predict": max_tokens, "temperature": 0.3},
    )
    return response["message"]["content"].strip()


# ------------------------------------------------------------------
# Unified LLM caller with Groq → Ollama fallback
# ------------------------------------------------------------------

def _template_reply(intent: str, retrieved: list[dict]) -> str:
    """Rule-based template reply used when no LLM is available."""
    # Use the top retrieved reply as a grounded template
    if retrieved:
        top = retrieved[0]["agent_reply"]
        if top and len(top) > 10:
            return top
    templates = {
        "safety_incident":   "Your safety is our top priority. We're escalating this immediately. Please stay safe — our team will contact you shortly.",
        "payment_billing":   "We're sorry for the billing issue. Please send us a note at uber.com/help and our team will review and refund if applicable.",
        "account_access":    "We can help with your account access. Please reach out at uber.com/help with your registered email so we can assist.",
        "driver_behavior":   "We take driver conduct very seriously. Please send us the trip details at uber.com/help and our team will follow up.",
        "trip_issue":        "Sorry for the trouble with your trip. Please send us a note at uber.com/help and our team will look into this right away.",
        "ride_cancellation": "We understand your frustration with the cancellation. Please contact us at uber.com/help and we'll make this right.",
        "app_technical":     "Sorry for the technical issue! Please try force-closing and reopening the app, or contact us at uber.com/help if the issue continues.",
        "lost_item":         "We'll help you recover your item! Please send us a note at uber.com/help with the trip details and we'll connect you with your driver.",
        "wait_time":         "We're sorry for the long wait. Please reach out at uber.com/help if you're still experiencing issues and our team will assist.",
        "eats_order":        "Sorry about your Eats order! Please contact us at uber.com/help with your order details and we'll sort this out for you.",
        "general_complaint": "We're sorry to hear about your experience. Please send us a note at uber.com/help and our team will look into this.",
        "positive_feedback": "Thank you so much for the kind words! We're glad you had a great experience. We'll pass your feedback along to the driver.",
    }
    return templates.get(intent, "Thank you for reaching out. Please send us a note at uber.com/help and our team will assist you shortly.")


def _llm_call(system: str, user: str, model: str,
              judge: bool = False, max_tokens: int = 400,
              _intent: str = "", _retrieved: list = None) -> str:
    """Try Groq first; fall back to Ollama; then template."""
    if GROQ_API_KEY and GROQ_API_KEY != "gsk_your_key_here":
        try:
            return _groq_chat(system, user, model=model, max_tokens=max_tokens)
        except Exception as e:
            print(f"[Groq error: {e}] — falling back to Ollama")

    # Ollama fallback
    try:
        return _ollama_chat(system, user, model=OLLAMA_MODEL, max_tokens=max_tokens)
    except Exception:
        pass   # Ollama not running

    # Template fallback — no LLM available (evaluation mode)
    if not judge:
        return _template_reply(_intent, _retrieved or [])
    return '{"relevance":3,"empathy":3,"accuracy":3,"actionability":3}'


# ------------------------------------------------------------------
# Reply generation
# ------------------------------------------------------------------

def format_examples(retrieved: list[dict]) -> str:
    """Format retrieved RAG examples for the prompt."""
    blocks = []
    for i, r in enumerate(retrieved, 1):
        cust  = r["customer_msg"][:120]
        reply = r["agent_reply"][:200]
        score = r.get("retrieval_score", 0.0)
        blocks.append(f"Example {i} (similarity={score:.2f}):\n"
                      f"  Customer : {cust}\n"
                      f"  Uber     : {reply}")
    return "\n\n".join(blocks)


def generate_reply(customer_msg: str, intent: str, confidence: float,
                   retrieved: list[dict]) -> dict:
    """Generate a grounded reply using RAG context."""
    examples_block = format_examples(retrieved)
    user_prompt = REPLY_PROMPT.format(
        customer_msg=customer_msg,
        intent=intent,
        confidence=confidence,
        n_examples=len(retrieved),
        examples_block=examples_block,
    )

    reply = _llm_call(
        system=REPLY_SYSTEM,
        user=user_prompt,
        model=GROQ_GEN_MODEL,
        max_tokens=180,
        _intent=intent,
        _retrieved=retrieved,
    )

    # Truncate to Twitter limit if LLM overruns
    if len(reply) > 500:
        reply = reply[:497] + "..."

    return {
        "reply":            reply,
        "model_used":       GROQ_GEN_MODEL,
        "n_retrieved":      len(retrieved),
        "top_retrieval_score": retrieved[0]["retrieval_score"] if retrieved else 0.0,
    }


# ------------------------------------------------------------------
# LLM-as-Judge (G-Eval)
# ------------------------------------------------------------------

def judge_reply(customer_msg: str, generated_reply: str, reference_reply: str,
                intent: str) -> dict:
    """G-Eval: rate generated reply on 5 dimensions using the judge model."""
    user_prompt = JUDGE_PROMPT.format(
        customer_msg=customer_msg,
        generated_reply=generated_reply,
        reference_reply=reference_reply,
        intent=intent,
    )

    raw = _llm_call(
        system=JUDGE_SYSTEM,
        user=user_prompt,
        model=GROQ_JUDGE_MODEL,
        max_tokens=400,
    )

    # Parse JSON from LLM output (handle code fences)
    import re
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return {"error": "No JSON found in judge output", "raw": raw}

    try:
        scores = json.loads(json_match.group())
        # Compute weighted score if not already done by LLM
        if "weighted_score" not in scores:
            weights = {"relevance": 0.30, "groundedness": 0.25,
                       "empathy": 0.20, "actionability": 0.15, "conciseness": 0.10}
            scores["weighted_score"] = sum(
                scores.get(k, 3) * w for k, w in weights.items()
            )
        return scores
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": raw}


def judge_escalation(customer_msg: str, draft_reply: str,
                     intent: str, confidence: float) -> dict:
    """Ask LLM to judge escalation decision (secondary layer)."""
    user_prompt = ESCALATION_JUDGE_PROMPT.format(
        customer_msg=customer_msg,
        draft_reply=draft_reply,
        intent=intent,
        confidence=confidence,
    )
    raw = _llm_call(
        system="You are a customer support quality controller. Output JSON only.",
        user=user_prompt,
        model=GROQ_JUDGE_MODEL,
        max_tokens=150,
    )
    import re
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return {"decision": "escalate", "reason": "judge parse failed", "confidence": 0.5}
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {"decision": "escalate", "reason": "judge parse failed", "confidence": 0.5}
