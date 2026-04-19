#!/usr/bin/env python3
"""Quick smoke test for GyanEngine — in-memory Qdrant, small sample."""

import sys
import time
import os
sys.path.insert(0, os.path.dirname(__file__))

from gyan_engine import GyanEngine

def test():
    print("=== Gyan Engine Smoke Test ===\n")

    # 1. Init engine (in-memory)
    t0 = time.time()
    engine = GyanEngine()
    print(f"Engine init: {time.time()-t0:.2f}s")

    # 2. Teach it some facts
    facts = [
        ("Who wrote Hamlet?", "William Shakespeare"),
        ("What is the capital of France?", "Paris"),
        ("When was the Declaration of Independence signed?", "August 2, 1776"),
        ("What is the speed of light?", "299,792,458 meters per second"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("Who discovered penicillin?", "Alexander Fleming"),
        ("What is the chemical symbol for gold?", "Au"),
        ("How many chromosomes do humans have?", "46"),
        ("What year did World War II end?", "1945"),
    ]

    print(f"\nTeaching {len(facts)} facts...")
    for q, a in facts:
        result = engine.learn(q, a, source="test")
        assert result["action"] == "learned", f"Failed to learn: {q}"
    print(f"Stats: {engine.stats()}")

    # 3. Query — exact matches
    print("\n--- Exact Match Tests ---")
    tests = [
        ("Who wrote Hamlet?", "Shakespeare"),
        ("capital of France", "Paris"),
        ("speed of light", "299"),
        ("largest planet", "Jupiter"),
        ("chemical symbol gold", "Au"),
    ]

    passed = 0
    for q, expected_substr in tests:
        result = engine.query(q)
        answer = result["answer"]
        ok = expected_substr.lower() in answer.lower()
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] Q: {q}")
        print(f"         A: {answer} ({result['confidence']:.2f}, {result['hops']} hops, {result['timeMs']}ms)")

    # 4. Query — paraphrased
    print("\n--- Paraphrase Tests ---")
    paraphrase_tests = [
        ("Who is the author of the play Hamlet?", "Shakespeare"),
        ("What's France's capital city?", "Paris"),
        ("Which planet is the biggest?", "Jupiter"),
        ("When did WWII finish?", "1945"),
    ]
    for q, expected_substr in paraphrase_tests:
        result = engine.query(q)
        answer = result["answer"]
        ok = expected_substr.lower() in answer.lower()
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] Q: {q}")
        print(f"         A: {answer} ({result['confidence']:.2f}, {result['hops']} hops, {result['timeMs']}ms)")

    # 5. Test learn + dedup
    print("\n--- Learn Dedup Test ---")
    r1 = engine.learn("Who wrote Hamlet?", "Shakespeare", source="test")
    assert r1["action"] == "boosted", f"Expected boost, got {r1['action']}"
    print(f"  [PASS] Duplicate learn → boosted (weight={r1['weight']:.2f})")
    passed += 1

    # 6. Convergence test
    print("\n--- Convergence Test ---")
    result = engine.query("Who wrote Hamlet?")
    print(f"  Hops: {result['hops']}, Converged: {result.get('converged', False)}")
    print(f"  Q_sim: {result.get('q_similarity', 0):.3f}, A_sim: {result.get('a_similarity', 0):.3f}")

    total = len(tests) + len(paraphrase_tests) + 1
    print(f"\n=== Results: {passed}/{total} passed ===")
    return passed == total


if __name__ == "__main__":
    success = test()
    sys.exit(0 if success else 1)
