#!/usr/bin/env python3
"""
Build Gyan knowledge base on GPU.

Encodes all Q&A pairs from SQLite → Qdrant persistent storage.
Run this on RunPod/GPU, then transfer the Qdrant snapshot to the device.

Usage:
  # On GPU (RunPod):
  python3 build_kb.py --db saqt.db --output ./qdrant_data --gpu

  # Test locally (CPU):
  python3 build_kb.py --db ~/webmind-research/trained_model/saqt.db --output ./qdrant_data
"""

import os
import sys
import time
import argparse
import sqlite3
import json
import numpy as np


def build_with_onnx(db_path: str, output_path: str, batch_size: int = 256):
    """Build KB using ONNX encoder (works on CPU and GPU)."""
    from gyan_engine import GyanEngine

    engine = GyanEngine(qdrant_path=output_path)
    engine.load_from_sqlite(db_path, batch_size=batch_size)

    stats = engine.stats()
    print(f"\nDone! {stats['points']} points in Qdrant at {output_path}")
    return stats


def build_with_torch(db_path: str, output_path: str, batch_size: int = 512):
    """Build KB using PyTorch sentence-transformers (faster on GPU)."""
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    client = QdrantClient(path=output_path)

    # Create collection
    collections = [c.name for c in client.get_collections().collections]
    if "knowledge" in collections:
        client.delete_collection("knowledge")
    client.create_collection(
        collection_name="knowledge",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Load from SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM qa")
    total = cursor.fetchone()[0]
    print(f"Encoding {total} pairs...")

    cursor.execute("SELECT id, question, answer, source, weight FROM qa")
    batch_questions = []
    batch_meta = []
    batch_ids = []
    loaded = 0
    t0 = time.time()

    for row in cursor:
        rid, question, answer, source, weight = row
        batch_questions.append(question)
        batch_ids.append(rid)
        batch_meta.append({
            "question": question,
            "answer": answer,
            "source": source or "",
            "weight": weight or 1.0,
        })

        if len(batch_questions) >= batch_size:
            embeddings = model.encode(
                batch_questions, batch_size=batch_size,
                show_progress_bar=False, normalize_embeddings=True,
            )
            points = [
                PointStruct(
                    id=batch_ids[j],
                    vector=embeddings[j].tolist(),
                    payload=batch_meta[j],
                )
                for j in range(len(batch_questions))
            ]
            client.upsert(collection_name="knowledge", points=points)
            loaded += len(points)

            elapsed = time.time() - t0
            rate = loaded / elapsed
            eta = (total - loaded) / rate if rate > 0 else 0
            print(f"  {loaded}/{total} ({loaded*100//total}%) — {rate:.0f} pairs/s — ETA {eta:.0f}s")

            batch_questions = []
            batch_ids = []
            batch_meta = []

    # Final batch
    if batch_questions:
        embeddings = model.encode(
            batch_questions, batch_size=batch_size,
            show_progress_bar=False, normalize_embeddings=True,
        )
        points = [
            PointStruct(
                id=batch_ids[j],
                vector=embeddings[j].tolist(),
                payload=batch_meta[j],
            )
            for j in range(len(batch_questions))
        ]
        client.upsert(collection_name="knowledge", points=points)
        loaded += len(points)

    conn.close()
    elapsed = time.time() - t0
    print(f"\nDone! {loaded} points encoded in {elapsed:.1f}s ({loaded/elapsed:.0f} pairs/s)")
    print(f"Qdrant data at: {output_path}")

    info = client.get_collection("knowledge")
    print(f"Collection stats: {info.points_count} points, status={info.status.value}")


def export_to_numpy(db_path: str, output_dir: str, batch_size: int = 512):
    """Export embeddings + metadata as numpy arrays for maximum portability."""
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, question, answer, source, weight FROM qa ORDER BY id")
    rows = cursor.fetchall()
    conn.close()

    total = len(rows)
    print(f"Encoding {total} pairs...")

    questions = [r[1] for r in rows]
    t0 = time.time()
    embeddings = model.encode(
        questions, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"Encoded in {elapsed:.1f}s ({total/elapsed:.0f} pairs/s)")

    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings as float16 to save space
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings.astype(np.float16))

    # Save metadata as JSON
    metadata = [
        {
            "id": r[0],
            "question": r[1],
            "answer": r[2],
            "source": r[3] or "",
            "weight": r[4] or 1.0,
        }
        for r in rows
    ]
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

    emb_size = os.path.getsize(os.path.join(output_dir, "embeddings.npy"))
    meta_size = os.path.getsize(os.path.join(output_dir, "metadata.json"))
    print(f"\nExported to {output_dir}:")
    print(f"  embeddings.npy: {emb_size/1024/1024:.1f}MB ({total} × 384 × fp16)")
    print(f"  metadata.json: {meta_size/1024/1024:.1f}MB")


def main():
    parser = argparse.ArgumentParser(description="Build Gyan KB on GPU")
    parser.add_argument("--db", required=True, help="Path to saqt.db")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--gpu", action="store_true",
                        help="Use PyTorch + GPU (faster)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--format", choices=["qdrant", "numpy"], default="qdrant",
                        help="Output format: qdrant (persistent) or numpy (portable)")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: DB not found at {args.db}")
        sys.exit(1)

    if args.format == "numpy":
        export_to_numpy(args.db, args.output, args.batch_size)
    elif args.gpu:
        build_with_torch(args.db, args.output, args.batch_size)
    else:
        build_with_onnx(args.db, args.output, args.batch_size)


if __name__ == "__main__":
    main()
