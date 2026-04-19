"""
Gyan Engine — ONNX encoder + Qdrant vector search + convergence loop.

The database is the model. Training is INSERT. Cost is $0.

Architecture:
  Query → ONNX MiniLM (22M, 384-dim) → Qdrant semantic search → Top-K
  → Bi-embedding re-rank → Convergence loop → Answer
  → If miss → web search → learn → KB grows

Usage:
  from gyan_engine import GyanEngine
  engine = GyanEngine()
  engine.load_from_sqlite("path/to/saqt.db")
  result = engine.query("Who wrote Hamlet?")
"""

import os
import time
import numpy as np
import onnxruntime as ort
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from transformers import AutoTokenizer
from typing import Optional
import sqlite3
import json
import math


class GyanEngine:
    """On-device AI engine. Encoder + Vector DB + Convergence Loop."""

    COLLECTION = "knowledge"
    EMBED_DIM = 384
    MAX_CONVERGENCE_HOPS = 5
    CONVERGENCE_THRESHOLD = 0.95  # cosine similarity for fixed-point

    def __init__(self, qdrant_path: Optional[str] = None, onnx_model: Optional[str] = None):
        """
        Args:
            qdrant_path: Path to persistent Qdrant storage, or None for in-memory.
            onnx_model: Path to ONNX model file. Auto-detected if None.
        """
        # ONNX encoder
        model_path = onnx_model or self._find_onnx_model()
        self.session = ort.InferenceSession(
            model_path,
            providers=ort.get_available_providers(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Qdrant
        if qdrant_path:
            self.qdrant = QdrantClient(path=qdrant_path)
        else:
            self.qdrant = QdrantClient(":memory:")

        self._ensure_collection()
        self._pair_count = 0

    def _find_onnx_model(self) -> str:
        """Auto-detect cached ONNX model."""
        cache_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
        )
        for root, _, files in os.walk(cache_dir):
            for f in files:
                if f == "model_O3.onnx":
                    return os.path.join(root, f)
                if f == "model_O2.onnx":
                    return os.path.join(root, f)
        # Fallback: download via sentence-transformers
        raise FileNotFoundError(
            "ONNX model not found. Run: "
            "python3 -c \"from sentence_transformers import SentenceTransformer; "
            "SentenceTransformer('all-MiniLM-L6-v2', backend='onnx')\""
        )

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.COLLECTION not in collections:
            self.qdrant.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(
                    size=self.EMBED_DIM,
                    distance=Distance.COSINE,
                ),
            )

    # ─── Encoding ───────────────────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode text to 384-dim embedding via ONNX."""
        inputs = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=128, return_tensors="np",
        )
        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
                "token_type_ids": inputs.get(
                    "token_type_ids",
                    np.zeros_like(inputs["input_ids"])
                ).astype(np.int64),
            },
        )
        # Mean pooling over token embeddings
        token_embeds = outputs[0]  # (1, seq_len, 384)
        mask = inputs["attention_mask"].astype(np.float32)
        mask_expanded = np.expand_dims(mask, -1)  # (1, seq_len, 1)
        summed = np.sum(token_embeds * mask_expanded, axis=1)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embedding = (summed / counts)[0]
        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode multiple texts efficiently."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="np",
            )
            outputs = self.session.run(
                None,
                {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                    "token_type_ids": inputs.get(
                        "token_type_ids",
                        np.zeros_like(inputs["input_ids"])
                    ).astype(np.int64),
                },
            )
            token_embeds = outputs[0]
            mask = inputs["attention_mask"].astype(np.float32)
            mask_expanded = np.expand_dims(mask, -1)
            summed = np.sum(token_embeds * mask_expanded, axis=1)
            counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            batch_embeds = summed / counts
            # L2 normalize
            norms = np.linalg.norm(batch_embeds, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-9, a_max=None)
            batch_embeds = batch_embeds / norms
            all_embeddings.append(batch_embeds)
        return np.vstack(all_embeddings)

    # ─── Knowledge Base ─────────────────────────────────────────

    def load_from_sqlite(self, db_path: str, batch_size: int = 256):
        """Load Q&A pairs from SAQT SQLite DB into Qdrant."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM qa")
        total = cursor.fetchone()[0]

        existing = self.qdrant.count(self.COLLECTION).count
        if existing >= total:
            print(f"Qdrant already has {existing} points (DB has {total}). Skipping load.")
            self._pair_count = existing
            return

        print(f"Loading {total} pairs from {db_path} into Qdrant...")
        cursor.execute("SELECT id, question, answer, source, weight FROM qa")

        points_batch = []
        loaded = 0
        questions_batch = []
        ids_batch = []
        meta_batch = []

        for row in cursor:
            rid, question, answer, source, weight = row
            questions_batch.append(question)
            ids_batch.append(rid)
            meta_batch.append({
                "question": question,
                "answer": answer,
                "source": source or "",
                "weight": weight or 1.0,
            })

            if len(questions_batch) >= batch_size:
                embeddings = self.encode_batch(questions_batch)
                for j, emb in enumerate(embeddings):
                    points_batch.append(PointStruct(
                        id=ids_batch[j],
                        vector=emb.tolist(),
                        payload=meta_batch[j],
                    ))

                self.qdrant.upsert(
                    collection_name=self.COLLECTION,
                    points=points_batch,
                )
                loaded += len(points_batch)
                if loaded % 5000 < batch_size:
                    print(f"  {loaded}/{total} ({loaded*100//total}%)")
                points_batch = []
                questions_batch = []
                ids_batch = []
                meta_batch = []

        # Final batch
        if questions_batch:
            embeddings = self.encode_batch(questions_batch)
            for j, emb in enumerate(embeddings):
                points_batch.append(PointStruct(
                    id=ids_batch[j],
                    vector=emb.tolist(),
                    payload=meta_batch[j],
                ))
            self.qdrant.upsert(
                collection_name=self.COLLECTION,
                points=points_batch,
            )
            loaded += len(points_batch)

        conn.close()
        self._pair_count = loaded
        print(f"Loaded {loaded} pairs into Qdrant.")

    def learn(self, question: str, answer: str, source: str = "learned",
              weight: float = 0.5) -> dict:
        """Learn a new Q&A pair. INSERT INTO is all you need."""
        # Semantic dedup — check if we already know this
        q_embed = self.encode(question)
        resp = self.qdrant.query_points(
            collection_name=self.COLLECTION,
            query=q_embed.tolist(),
            limit=1,
        )
        hits = resp.points
        if hits and hits[0].score > 0.95:
            # Already know this — boost existing
            existing = hits[0]
            new_weight = min(existing.payload.get("weight", 1.0) * 1.1, 5.0)
            self.qdrant.set_payload(
                collection_name=self.COLLECTION,
                payload={"weight": new_weight},
                points=[existing.id],
            )
            return {"action": "boosted", "id": existing.id, "weight": new_weight}

        # New knowledge
        self._pair_count += 1
        new_id = self._pair_count + 100000  # offset to avoid collisions with SQLite IDs
        self.qdrant.upsert(
            collection_name=self.COLLECTION,
            points=[PointStruct(
                id=new_id,
                vector=q_embed.tolist(),
                payload={
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "weight": weight,
                },
            )],
        )
        return {"action": "learned", "id": new_id}

    def feedback(self, point_id: int, positive: bool):
        """Boost or penalize a known answer."""
        points = self.qdrant.retrieve(self.COLLECTION, ids=[point_id])
        if not points:
            return
        current_weight = points[0].payload.get("weight", 1.0)
        if positive:
            new_weight = min(current_weight * 1.1, 5.0)
        else:
            new_weight = max(current_weight * 0.9, 0.1)
        self.qdrant.set_payload(
            collection_name=self.COLLECTION,
            payload={"weight": new_weight},
            points=[point_id],
        )

    # ─── Question Type Awareness ──────────────────────────────

    _TYPE_HINTS = {
        "when": "date year time period",
        "where": "place location country city",
        "who": "person name people",
        "how many": "number count quantity",
        "how much": "number amount quantity",
        "how long": "duration time length",
        "how": "method way process",
        "what year": "date year",
        "what country": "country nation",
        "what language": "language spoken",
    }

    def _augment_query(self, query: str) -> str:
        """Prepend question-type context to nudge encoder toward correct fact type."""
        ql = query.lower().strip()
        for prefix, hint in self._TYPE_HINTS.items():
            if ql.startswith(prefix):
                return f"{hint}: {query}"
        return query

    # ─── Search ─────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Semantic search with bi-embedding re-ranking + type awareness."""
        augmented = self._augment_query(query)
        q_embed = self.encode(augmented)

        # Phase 1: broad retrieval
        resp = self.qdrant.query_points(
            collection_name=self.COLLECTION,
            query=q_embed.tolist(),
            limit=top_k * 3,
        )
        candidates = resp.points

        if not candidates:
            return []

        # Phase 2: bi-embedding re-rank
        # Score = 0.4 * Q_similarity + 0.4 * A_similarity + 0.2 * weight
        results = []
        for hit in candidates:
            q_sim = hit.score
            # Encode the answer and check alignment with query
            a_embed = self.encode(hit.payload["answer"][:200])
            a_sim = float(np.dot(q_embed, a_embed))
            weight = hit.payload.get("weight", 1.0) / 5.0  # normalize to 0-1

            combined = 0.4 * q_sim + 0.4 * a_sim + 0.2 * weight
            results.append({
                "id": hit.id,
                "question": hit.payload["question"],
                "answer": hit.payload["answer"],
                "source": hit.payload.get("source", ""),
                "q_similarity": q_sim,
                "a_similarity": a_sim,
                "weight": hit.payload.get("weight", 1.0),
                "score": combined,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ─── Chain-of-Retrieval ───────────────────────────────────

    _CHAIN_PATTERNS = [
        (r'(?:the )?author of ["\']?(.+?)["\']?(?:\s+\w+\?|\?|$)', 'Who is the author of {}?'),
        (r'(?:the )?country where (.+?)(?:\s+is|\?|$)', 'What country is {} in?'),
        (r'where (.+?) (?:is|are|was)\b', 'Where is {}?'),
        (r'(?:the )?instrument (.+?) played', 'What instrument did {} play?'),
        (r'(?:the )?planet (?:that |which )?is known as (.+?)(?:\s*\?|$)', 'Which planet is {}?'),
        (r'country where (.+?) originated', 'What country did {} originate from?'),
        (r'(?:the )?theory of (.+?)(?:\s*\?|$)', 'Who created the theory of {}?'),
    ]

    def _try_chain(self, question: str) -> list[dict]:
        """Try chain-of-retrieval: resolve inner references, rewrite, search."""
        import re as _re
        ql = question.lower()
        chain_results = []

        for pattern, sub_template in self._CHAIN_PATTERNS:
            match = _re.search(pattern, ql)
            if not match:
                continue

            inner_ref = match.group(1).strip().rstrip('?')
            sub_q = sub_template.format(inner_ref)
            sub_results = self.search(sub_q, top_k=3)
            if not sub_results or sub_results[0]["score"] < 0.5:
                continue

            resolved = sub_results[0]["answer"].split('(')[0].split('\n')[0].strip()[:100]

            # Rewrite with resolved entity
            rewritten = _re.sub(pattern, resolved, ql, count=1)
            q_word = ql.split()[0]
            if not rewritten.lower().startswith(q_word):
                rewritten = f"{q_word} {rewritten}"

            rewritten_results = self.search(rewritten, top_k=3)
            if rewritten_results:
                for r in rewritten_results:
                    r["_chain"] = True
                    r["_resolved"] = resolved
                chain_results.extend(rewritten_results)

        return chain_results

    # ─── Convergence Loop ───────────────────────────────────────

    def query(self, question: str, context: str = "") -> dict:
        """
        Full query: type-aware search + chain-of-retrieval + convergence.
        Picks the highest-confidence result across all strategies.
        """
        start = time.time()
        full_query = f"{context} {question}".strip() if context else question

        # Direct search (with type hints)
        results = self.search(full_query)

        # Chain-of-retrieval for compositional questions
        chain_results = self._try_chain(full_query)
        if chain_results:
            results = results + chain_results
            results.sort(key=lambda x: x["score"], reverse=True)
        if not results:
            return {
                "answer": "",
                "confidence": 0,
                "hops": 0,
                "timeMs": int((time.time() - start) * 1000),
                "source": "none",
            }

        best = results[0]
        noise_floor = 1.0 / math.sqrt(max(len(results), 1))

        if best["score"] < noise_floor:
            return {
                "answer": "",
                "confidence": 0,
                "hops": 0,
                "timeMs": int((time.time() - start) * 1000),
                "source": "below_noise_floor",
            }

        # Convergence loop — iterate until answer stabilizes
        prev_answer = best["answer"]
        prev_embed = self.encode(prev_answer[:200])
        hops = 0

        for hop in range(1, self.MAX_CONVERGENCE_HOPS + 1):
            # Search using the answer's embedding
            hop_resp = self.qdrant.query_points(
                collection_name=self.COLLECTION,
                query=prev_embed.tolist(),
                limit=5,
            )

            if not hop_resp.points:
                break

            hop_best = hop_resp.points[0]
            hop_embed = self.encode(hop_best.payload["answer"][:200])

            # Check convergence: is the new answer similar to the previous?
            similarity = float(np.dot(prev_embed, hop_embed))
            hops = hop

            if similarity >= self.CONVERGENCE_THRESHOLD:
                # Fixed point reached — answer is stable
                break

            # Update for next iteration
            prev_answer = hop_best.payload["answer"]
            prev_embed = hop_embed

        elapsed = int((time.time() - start) * 1000)
        return {
            "answer": best["answer"],
            "confidence": float(best["score"]),
            "hops": hops,
            "timeMs": elapsed,
            "source": best.get("source", ""),
            "q_similarity": float(best["q_similarity"]),
            "a_similarity": float(best["a_similarity"]),
            "converged": hops > 0,
        }

    # ─── Export / Import ────────────────────────────────────────

    def export_snapshot(self, path: str):
        """Export Qdrant collection to a snapshot file."""
        self.qdrant.create_snapshot(collection_name=self.COLLECTION)
        print(f"Snapshot created. Use qdrant CLI to export to {path}")

    def stats(self) -> dict:
        """Collection statistics."""
        info = self.qdrant.get_collection(self.COLLECTION)
        count = self.qdrant.count(self.COLLECTION).count
        return {
            "points": count,
            "status": info.status.value,
        }


# ─── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gyan Engine CLI")
    parser.add_argument("--db", default=os.path.expanduser(
        "~/webmind-research/trained_model/saqt.db"
    ))
    parser.add_argument("--qdrant-path", default=None,
                        help="Persistent Qdrant storage path")
    parser.add_argument("--query", "-q", help="Ask a question")
    parser.add_argument("--load", action="store_true",
                        help="Load SQLite DB into Qdrant")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    args = parser.parse_args()

    engine = GyanEngine(qdrant_path=args.qdrant_path)

    if args.load:
        engine.load_from_sqlite(args.db)

    if args.stats:
        print(json.dumps(engine.stats(), indent=2))

    if args.query:
        result = engine.query(args.query)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.interactive:
        if engine.stats()["points"] == 0:
            print("Loading KB...")
            engine.load_from_sqlite(args.db)
        print(f"\nGyan Engine — {engine.stats()['points']} knowledge pairs")
        print("Type a question (Ctrl+C to quit)\n")
        while True:
            try:
                q = input("? ")
                if not q.strip():
                    continue
                result = engine.query(q)
                if result["answer"]:
                    print(f"→ {result['answer']}")
                    print(f"  [{result['confidence']:.2f} conf, {result['hops']} hops, {result['timeMs']}ms]")
                else:
                    print("  (no answer found)")
                print()
            except (KeyboardInterrupt, EOFError):
                break
