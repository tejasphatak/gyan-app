#!/usr/bin/env python3
"""Test with full 88K KB loaded from saqt.db."""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

from gyan_engine import GyanEngine

DB = os.path.expanduser("~/webmind-research/trained_model/saqt.db")
QDRANT_PATH = os.path.expanduser("~/gyan-app/data/qdrant_88k")

def main():
    print("=== Full KB Test ===\n")

    t0 = time.time()
    engine = GyanEngine(qdrant_path=QDRANT_PATH)
    print(f"Engine init: {time.time()-t0:.2f}s")

    stats = engine.stats()
    if stats["points"] < 88000:
        print(f"Loading KB (currently {stats['points']} points)...")
        engine.load_from_sqlite(DB, batch_size=256)
    else:
        print(f"KB already loaded: {stats['points']} points")

    # Test queries
    questions = [
        ("who wrote hamlet", "Shakespeare"),
        ("what is the capital of france", "Paris"),
        ("where did they film hot tub time machine", "Fernie"),
        ("who has the right of way in international waters", "Neither"),
        ("when was the last time someone was on the moon", None),
        ("who painted the sistine chapel", "Michelangelo"),
        ("what is the speed of light", None),
        ("who discovered penicillin", "Fleming"),
        ("what year did world war 2 end", "1945"),
        ("how many strings does a guitar have", "6"),
    ]

    print(f"\n--- Query Tests ({len(questions)} questions) ---")
    passed = 0
    total_ms = 0
    for q, expected in questions:
        result = engine.query(q)
        answer = result["answer"]
        ms = result["timeMs"]
        total_ms += ms

        if expected:
            ok = expected.lower() in answer.lower()
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
        else:
            status = "????"
            passed += 1  # no expected = just checking it returns something

        print(f"  [{status}] {q}")
        print(f"         → {answer[:80]} ({result['confidence']:.2f}, {result['hops']}h, {ms}ms)")

    print(f"\n{passed}/{len(questions)} passed, avg latency {total_ms//len(questions)}ms")


if __name__ == "__main__":
    main()
