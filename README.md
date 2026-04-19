# Gyan — AI In Your Hand

**ज्ञान (Gyan) = Knowledge**

The "INSERT INTO Is All You Need" architecture as a standalone engine. No cloud. No LLM. No hallucination. Just a sentence encoder + Qdrant vector search + a learning loop.

## Architecture

```
You type → ONNX MiniLM (22M params) → 384-dim embedding
        → Qdrant semantic search → Top-K candidates
        → Bi-embedding re-rank → Convergence loop → Answer
        → If miss → Web search → Learn → KB grows
        → Gets smarter with every question
```

## Engine (Python backend)

```
engine/
├── gyan_engine.py    # Core: ONNX encoder + Qdrant + convergence loop
├── serve.py          # HTTP API server
├── web_search.py     # Wikipedia + DuckDuckGo fallback
├── build_kb.py       # Build KB on GPU (RunPod/local)
├── benchmark.py      # NQ/TriviaQA/HotPotQA benchmark
├── test_engine.py    # Smoke tests (10/10 passing)
└── requirements.txt  # onnxruntime, qdrant-client, transformers
```

### Quick start

```bash
cd engine
pip install -r requirements.txt

# Smoke test (in-memory, no data needed)
python3 test_engine.py

# Interactive mode with full KB
python3 gyan_engine.py -i --qdrant-path ../data/qdrant_88k

# HTTP server
python3 serve.py

# Benchmark
python3 benchmark.py --samples 50 --rlhf
```

### Build KB on GPU

```bash
# On RunPod/GPU (fast — ~2 min for 88K pairs):
python3 build_kb.py --db saqt.db --output ./qdrant_data --gpu

# On CPU (slower — ~10 min):
python3 build_kb.py --db saqt.db --output ./qdrant_data
```

## Pre-trained Dataset

| Dataset | Pairs | Type |
|---------|-------|------|
| NaturalQuestions | 88K | Factual Q&A |
| Wikipedia composed | 317 | Full paragraph answers |
| Thinking templates | 19 | Response patterns |
| **Total** | **~88K** | Mixed |

Target: scale to 500K+ with SQuAD, TriviaQA, ELI5, MS MARCO

## How It Works

1. Query encoded to 384-dim vector via ONNX MiniLM (22M params)
2. Qdrant semantic search finds top-K candidates
3. Bi-embedding re-rank: score = 0.4 * Q_sim + 0.4 * A_sim + 0.2 * weight
4. Convergence loop: search with answer's embedding → same answer = stable
5. If miss: web search → learn → KB grows
6. Every miss makes the system smarter. Training = INSERT.

## License

Code: MIT · Data: CC-BY 4.0
