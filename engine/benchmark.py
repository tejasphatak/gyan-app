#!/usr/bin/env python3
"""
Gyan Benchmark — test engine against NQ/TriviaQA/HotPotQA.

Same methodology as saqt/benchmark.mjs but using the Python Qdrant engine directly.
Measures exact match, F1, latency. Supports RLHF (learn from mistakes).

Usage:
  python3 benchmark.py --samples 50 --rlhf
  python3 benchmark.py --samples 100 --qdrant-path ./data/qdrant_88k
"""

import os
import sys
import re
import json
import time
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from gyan_engine import GyanEngine


# ─── Dataset Fetchers ──────────────────────────────────────────

def fetch_dataset(name: str, n: int) -> list[dict]:
    """Fetch benchmark dataset from HuggingFace."""
    import urllib.request

    configs = {
        "NaturalQuestions": {
            "url": f"https://datasets-server.huggingface.co/rows?dataset=google-research-datasets/nq_open&config=nq_open&split=validation&offset=0&length={n}",
            "parse": lambda r: {
                "question": r["row"]["question"],
                "gold_answers": r["row"]["answer"] if isinstance(r["row"]["answer"], list)
                                else [r["row"]["answer"]],
            },
        },
        "TriviaQA": {
            "url": f"https://datasets-server.huggingface.co/rows?dataset=mandarjoshi/trivia_qa&config=unfiltered.nocontext&split=validation&offset=0&length={n}",
            "parse": lambda r: {
                "question": r["row"]["question"],
                "gold_answers": list(filter(None, [
                    *(r["row"]["answer"].get("aliases", []) if r["row"]["answer"] else []),
                    r["row"]["answer"].get("value") if r["row"]["answer"] else None,
                ])),
            },
        },
        "HotPotQA": {
            "url": f"https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation&offset=0&length={n}",
            "parse": lambda r: {
                "question": r["row"]["question"],
                "gold_answers": [r["row"]["answer"]],
            },
        },
    }

    cfg = configs[name]
    print(f"  Fetching {name} ({n} samples)...")
    req = urllib.request.Request(cfg["url"])
    req.add_header("User-Agent", "gyan-benchmark/1.0")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return [cfg["parse"](r) for r in data["rows"]]
    except Exception as e:
        print(f"  Failed to fetch {name}: {e}")
        return []


# ─── Metrics ───────────────────────────────────────────────────

STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "of", "in", "on",
              "at", "to", "for", "and", "or", "but", "not", "with", "from",
              "by", "as", "it", "its", "this", "that", "these", "those"}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return ' '.join(words).strip()


def exact_match(predicted: str, golds: list[str]) -> int:
    norm_pred = normalize(predicted)
    return 1 if any(normalize(g) == norm_pred for g in golds) else 0


def f1_score(predicted: str, golds: list[str]) -> float:
    pred_tokens = normalize(predicted).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in golds:
        gold_tokens = normalize(gold).split()
        if not gold_tokens:
            continue
        common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


# ─── Runner ────────────────────────────────────────────────────

def run_benchmark(engine: GyanEngine, items: list[dict], label: str,
                  rlhf: bool = False, context: str = "") -> list[dict]:
    results = []
    for i, item in enumerate(items):
        result = engine.query(item["question"], context=context)
        em = exact_match(result["answer"], item["gold_answers"])
        f1 = f1_score(result["answer"], item["gold_answers"])

        results.append({
            "question": item["question"],
            "gold_answers": item["gold_answers"],
            "predicted": result["answer"],
            "exact_match": em,
            "f1": f1,
            "confidence": result["confidence"],
            "hops": result["hops"],
            "timeMs": result["timeMs"],
        })

        # RLHF: teach correct answer on miss
        if rlhf and em == 0 and item["gold_answers"] and result["answer"]:
            engine.learn(
                item["question"],
                item["gold_answers"][0],
                source="benchmark-rlhf",
            )

        if (i + 1) % 10 == 0 or i + 1 == len(items):
            print(f"  [{label}] {i+1}/{len(items)}")

    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    em = sum(r["exact_match"] for r in results) / max(n, 1)
    f1 = sum(r["f1"] for r in results) / max(n, 1)
    avg_ms = sum(r["timeMs"] for r in results) / max(n, 1)
    latencies = sorted(r["timeMs"] for r in results)
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
    return {"n": n, "em": em, "f1": f1, "avgMs": avg_ms, "p50": p50, "p95": p95}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--db", default=os.path.expanduser(
        "~/webmind-research/trained_model/saqt.db"
    ))
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--rlhf", action="store_true",
                        help="Learn from wrong answers")
    parser.add_argument("--context", default="",
                        help="Context prompt for queries")
    parser.add_argument("--output", default=os.path.expanduser(
        "~/gyan-app/data/benchmarks"
    ))
    args = parser.parse_args()

    engine = GyanEngine(qdrant_path=args.qdrant_path)
    stats = engine.stats()
    if stats["points"] == 0:
        print("Loading KB...")
        engine.load_from_sqlite(args.db)

    print(f"\nGyan Benchmark — {args.samples} samples/dataset")
    print(f"KB: {engine.stats()['points']} pairs")
    print(f"RLHF: {'on' if args.rlhf else 'off'}\n")

    # Fetch datasets
    print("Fetching datasets...")
    datasets = {}
    for name in ["NaturalQuestions", "TriviaQA", "HotPotQA"]:
        items = fetch_dataset(name, args.samples)
        if items:
            datasets[name] = items

    if not datasets:
        print("No datasets fetched. Aborting.")
        sys.exit(1)

    # Run benchmarks
    all_results = {}
    all_summaries = {}
    context = args.context or (
        "You are being evaluated for factual accuracy. Give precise, direct answers."
        if args.rlhf else ""
    )

    for name, items in datasets.items():
        print(f"\nRunning {name}...")
        results = run_benchmark(engine, items, name, rlhf=args.rlhf, context=context)
        all_results[name] = results
        all_summaries[name] = summarize(results)

    # Overall
    all_flat = [r for results in all_results.values() for r in results]
    all_summaries["OVERALL"] = summarize(all_flat)

    # Print results
    print("\n" + "=" * 72)
    print("GYAN BENCHMARK RESULTS")
    print("=" * 72)

    header = f"{'Dataset':<22}{'N':>5}{'EM':>8}{'F1':>8}{'AvgMs':>8}{'P50':>8}{'P95':>8}"
    print(header)
    print("-" * 72)
    for name, s in all_summaries.items():
        row = (f"{name:<22}{s['n']:>5}"
               f"{s['em']*100:>7.1f}%{s['f1']*100:>7.1f}%"
               f"{s['avgMs']:>8.0f}{s['p50']:>8}{s['p95']:>8}")
        print(row)
    print("=" * 72)

    # Save
    os.makedirs(args.output, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_file = os.path.join(args.output, f"gyan-bench-{ts}.json")
    with open(out_file, "w") as f:
        json.dump({
            "meta": {
                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "samples": args.samples,
                "kb_size": engine.stats()["points"],
                "rlhf": args.rlhf,
            },
            "summaries": all_summaries,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
