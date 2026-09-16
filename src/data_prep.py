"""
data_prep.py — Extract and clean Uber_Support conversation pairs from TWCS.

Outputs:
  data/uber_pairs.csv  — (thread_id, customer_msg, agent_reply, full_context, timestamp)
"""

import csv
import json
import re
import os
from collections import defaultdict
from tqdm import tqdm

BRAND = "Uber_Support"
TWCS_PATH = os.getenv("DATA_PATH", "twcs.csv")
OUT_CSV = "data/uber_pairs.csv"
OUT_JSON = "data/uber_pairs_full.json"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def is_english(text: str) -> bool:
    """Rough English filter: >70% ASCII printable chars."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / max(len(text), 1) > 0.70


def clean_text(text: str) -> str:
    """Remove @mentions, URLs, excess whitespace from a tweet."""
    text = re.sub(r"@\w+", "", text)          # strip @mentions
    text = re.sub(r"http\S+", "", text)        # strip URLs
    text = re.sub(r"\s+", " ", text).strip()   # normalise whitespace
    return text


def build_thread(row_lookup: dict, root_id: str, max_depth: int = 10) -> list[dict]:
    """Walk a conversation chain from root to leaf in chronological order."""
    thread, current, visited = [], root_id, set()
    while current and current not in visited and len(thread) < max_depth:
        visited.add(current)
        row = row_lookup.get(current)
        if row is None:
            break
        thread.append(row)
        current = row.get("in_response_to_tweet_id", "").strip()
    return list(reversed(thread))  # chronological (oldest first)


def flatten_thread(thread: list[dict]) -> str:
    """Produce a readable string of the conversation."""
    parts = []
    for row in thread:
        role = "Customer" if row["inbound"] == "True" else "Agent"
        text = clean_text(row["text"])
        if text:
            parts.append(f"{role}: {text}")
    return " | ".join(parts)


# ------------------------------------------------------------------
# Main extraction
# ------------------------------------------------------------------

def extract_uber_pairs(twcs_path: str = TWCS_PATH) -> list[dict]:
    print(f"Loading {twcs_path} …")
    row_lookup: dict[str, dict] = {}
    brand_reply_rows: list[dict] = []

    with open(twcs_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Reading rows", unit="rows"):
            row_lookup[row["tweet_id"]] = row
            if row["author_id"] == BRAND:
                brand_reply_rows.append(row)

    print(f"  Total rows: {len(row_lookup):,}")
    print(f"  {BRAND} reply rows: {len(brand_reply_rows):,}")

    pairs = []
    skipped_no_src = 0
    skipped_not_inbound = 0
    skipped_non_english = 0

    for agent_row in tqdm(brand_reply_rows, desc="Building pairs"):
        src_id = agent_row.get("in_response_to_tweet_id", "").strip()
        if not src_id or src_id not in row_lookup:
            skipped_no_src += 1
            continue

        src_row = row_lookup[src_id]
        if src_row["inbound"] != "True":
            skipped_not_inbound += 1
            continue

        customer_raw = src_row["text"]
        agent_raw = agent_row["text"]

        if not is_english(customer_raw):
            skipped_non_english += 1
            continue

        # Build full thread for context
        thread = build_thread(row_lookup, src_id, max_depth=8)
        full_context = flatten_thread(thread) if thread else clean_text(customer_raw)

        pairs.append({
            "thread_id":     src_id,
            "agent_tweet_id": agent_row["tweet_id"],
            "customer_msg":  clean_text(customer_raw),
            "customer_raw":  customer_raw,
            "agent_reply":   clean_text(agent_raw),
            "agent_raw":     agent_raw,
            "full_context":  full_context,
            "timestamp":     src_row.get("created_at", ""),
        })

    print(f"\nResults:")
    print(f"  Valid pairs extracted : {len(pairs):,}")
    print(f"  Skipped (no source)   : {skipped_no_src:,}")
    print(f"  Skipped (not inbound) : {skipped_not_inbound:,}")
    print(f"  Skipped (non-English) : {skipped_non_english:,}")
    return pairs


def save_pairs(pairs: list[dict], csv_path: str = OUT_CSV, json_path: str = OUT_JSON):
    os.makedirs("data", exist_ok=True)

    # CSV (for quick inspection)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pairs[0].keys())
        writer.writeheader()
        writer.writerows(pairs)
    print(f"Saved CSV  → {csv_path}")

    # JSON (for pipeline use)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON → {json_path}")


if __name__ == "__main__":
    pairs = extract_uber_pairs()
    if pairs:
        save_pairs(pairs)
        # Quick stats
        cust_lens = [len(p["customer_msg"]) for p in pairs]
        agent_lens = [len(p["agent_reply"]) for p in pairs]
        print(f"\nCustomer msg  — avg: {sum(cust_lens)/len(cust_lens):.0f}, "
              f"min: {min(cust_lens)}, max: {max(cust_lens)}")
        print(f"Agent reply   — avg: {sum(agent_lens)/len(agent_lens):.0f}, "
              f"min: {min(agent_lens)}, max: {max(agent_lens)}")
