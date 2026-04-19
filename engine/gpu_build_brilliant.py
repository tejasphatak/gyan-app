#!/usr/bin/env python3
"""
Build the "Brilliant Stack" KB on GPU.

Downloads and encodes:
  1. OASST2 (~160K) — conversational assistant answers
  2. ELI5 (~272K) — explanatory long-form
  3. NaturalQuestions (~88K) — factual grounding
  4. MS MARCO QnA (~1M) — broad factual coverage
  5. TriviaQA (~650K) — trivia/general knowledge

Exports as numpy (embeddings.npy + metadata.json) for portability.

Usage (on RunPod GPU):
  pip install sentence-transformers datasets
  python3 gpu_build_brilliant.py --output /workspace/brilliant_model
"""

import os
import sys
import time
import json
import argparse
import numpy as np


def load_oasst2(max_pairs=160000):
    """Open Assistant 2 — high-quality human-written answers."""
    from datasets import load_dataset
    print(f"Loading OASST2 (max {max_pairs})...")
    try:
        ds = load_dataset("OpenAssistant/oasst2", split="train")
        pairs = []
        # OASST2 is a tree structure — extract Q&A pairs from root→reply
        messages_by_parent = {}
        roots = []
        for row in ds:
            msg_id = row.get("message_id", "")
            parent_id = row.get("parent_id")
            text = row.get("text", "")
            role = row.get("role", "")
            if parent_id is None:
                roots.append({"id": msg_id, "text": text, "role": role})
            else:
                if parent_id not in messages_by_parent:
                    messages_by_parent[parent_id] = []
                messages_by_parent[parent_id].append({"id": msg_id, "text": text, "role": role})

        for root in roots:
            if root["role"] == "prompter" and root["id"] in messages_by_parent:
                replies = messages_by_parent[root["id"]]
                # Take the first assistant reply
                for reply in replies:
                    if reply["role"] == "assistant" and len(reply["text"]) > 20:
                        pairs.append({
                            "question": root["text"][:500],
                            "answer": reply["text"][:1000],
                            "source": "oasst2",
                        })
                        break
            if len(pairs) >= max_pairs:
                break
        print(f"  OASST2: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  OASST2 failed: {e}")
        return []


def load_eli5(max_pairs=272000):
    """ELI5 — Explain Like I'm 5."""
    from datasets import load_dataset
    print(f"Loading ELI5 (max {max_pairs})...")
    try:
        ds = load_dataset("eli5_category", split="train", trust_remote_code=True)
        pairs = []
        for row in ds:
            q = row.get("title", "")
            answers = row.get("answers", {})
            texts = answers.get("text", []) if isinstance(answers, dict) else []
            if texts and q:
                # Take the highest-scored answer
                scores = answers.get("score", [0] * len(texts))
                best_idx = scores.index(max(scores)) if scores else 0
                best_answer = texts[best_idx] if best_idx < len(texts) else texts[0]
                if len(best_answer) > 30:
                    pairs.append({
                        "question": q[:500],
                        "answer": best_answer[:1000],
                        "source": "eli5",
                    })
            if len(pairs) >= max_pairs:
                break
        print(f"  ELI5: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  ELI5 failed: {e}")
        return []


def load_natural_questions(max_pairs=88000):
    """NaturalQuestions open — factual grounding."""
    from datasets import load_dataset
    print(f"Loading NaturalQuestions (max {max_pairs})...")
    try:
        ds = load_dataset("google-research-datasets/nq_open", split="train")
        pairs = []
        for row in ds:
            q = row.get("question", "")
            answers = row.get("answer", [])
            if q and answers:
                a = answers[0] if isinstance(answers, list) else str(answers)
                pairs.append({
                    "question": q,
                    "answer": a,
                    "source": "nq-train",
                })
            if len(pairs) >= max_pairs:
                break
        print(f"  NQ: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  NQ failed: {e}")
        return []


def load_msmarco_qna(max_pairs=1000000):
    """MS MARCO QnA — broad factual coverage."""
    from datasets import load_dataset
    print(f"Loading MS MARCO QnA (max {max_pairs})...")
    try:
        ds = load_dataset("ms_marco", "v2.1", split="train", trust_remote_code=True)
        pairs = []
        for row in ds:
            q = row.get("query", "")
            answers = row.get("answers", [])
            if q and answers and answers[0] and answers[0] != "No Answer Present.":
                pairs.append({
                    "question": q[:500],
                    "answer": answers[0][:1000],
                    "source": "msmarco",
                })
            if len(pairs) >= max_pairs:
                break
        print(f"  MS MARCO: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  MS MARCO failed: {e}")
        return []


def load_triviaqa(max_pairs=650000):
    """TriviaQA — trivia and general knowledge."""
    from datasets import load_dataset
    print(f"Loading TriviaQA (max {max_pairs})...")
    try:
        ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext",
                         split="train", trust_remote_code=True)
        pairs = []
        for row in ds:
            q = row.get("question", "")
            answer = row.get("answer", {})
            value = answer.get("value", "") if isinstance(answer, dict) else str(answer)
            aliases = answer.get("aliases", []) if isinstance(answer, dict) else []
            if q and value:
                # Use value + first alias for richer answer
                full_answer = value
                if aliases:
                    alt = [a for a in aliases if a.lower() != value.lower()]
                    if alt:
                        full_answer = f"{value} (also known as: {', '.join(alt[:3])})"
                pairs.append({
                    "question": q[:500],
                    "answer": full_answer[:500],
                    "source": "triviaqa",
                })
            if len(pairs) >= max_pairs:
                break
        print(f"  TriviaQA: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  TriviaQA failed: {e}")
        return []


def encode_and_export(pairs, output_dir, batch_size=512):
    """Encode all pairs and export as numpy."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    total = len(pairs)
    print(f"\nEncoding {total} pairs...")

    questions = [p["question"] for p in pairs]
    t0 = time.time()
    embeddings = model.encode(
        questions, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"Encoded in {elapsed:.1f}s ({total/elapsed:.0f} pairs/s)")

    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as float16
    emb_path = os.path.join(output_dir, "embeddings.npy")
    np.save(emb_path, embeddings.astype(np.float16))

    # Save metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    metadata = [
        {
            "id": i + 1,
            "question": p["question"],
            "answer": p["answer"],
            "source": p["source"],
            "weight": 1.0,
        }
        for i, p in enumerate(pairs)
    ]
    with open(meta_path, "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

    # Stats
    emb_size = os.path.getsize(emb_path)
    meta_size = os.path.getsize(meta_path)
    print(f"\nExported to {output_dir}:")
    print(f"  embeddings.npy: {emb_size/1024/1024:.1f}MB ({total} x 384 x fp16)")
    print(f"  metadata.json: {meta_size/1024/1024:.1f}MB")
    print(f"  Total: {(emb_size+meta_size)/1024/1024:.1f}MB")

    # Summary by source
    from collections import Counter
    sources = Counter(p["source"] for p in pairs)
    print(f"\nDataset breakdown:")
    for src, count in sources.most_common():
        print(f"  {src}: {count:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip datasets: oasst2, eli5, nq, msmarco, triviaqa")
    args = parser.parse_args()

    # Load all datasets
    all_pairs = []

    loaders = [
        ("oasst2", load_oasst2),
        ("eli5", load_eli5),
        ("nq", load_natural_questions),
        ("triviaqa", load_triviaqa),
        ("msmarco", load_msmarco_qna),
    ]

    for name, loader in loaders:
        if name in args.skip:
            print(f"Skipping {name}")
            continue
        pairs = loader()
        all_pairs.extend(pairs)

    if not all_pairs:
        print("No data loaded!")
        sys.exit(1)

    print(f"\nTotal: {len(all_pairs):,} pairs")

    # Deduplicate by question (keep first occurrence)
    seen = set()
    unique = []
    for p in all_pairs:
        key = p["question"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    print(f"After dedup: {len(unique):,} pairs (removed {len(all_pairs)-len(unique):,} dupes)")

    encode_and_export(unique, args.output, args.batch_size)


if __name__ == "__main__":
    main()
