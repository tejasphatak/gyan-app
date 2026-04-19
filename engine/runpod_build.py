#!/usr/bin/env python3
"""
Spin up a RunPod GPU, build the Gyan KB, download results.

This script:
1. Creates a RunPod pod with GPU
2. Uploads saqt.db + build_kb.py
3. Runs encoding on GPU (~2 min for 88K pairs)
4. Downloads embeddings.npy + metadata.json
5. Terminates the pod

Usage:
  python3 runpod_build.py --api-key KEY --db ~/webmind-research/trained_model/saqt.db
"""

import os
import sys
import json
import time
import argparse
import subprocess

def get_api_key():
    """Load RunPod API key from secrets."""
    secrets_file = os.path.expanduser("~/.claude/secrets/runpod.json")
    if os.path.exists(secrets_file):
        with open(secrets_file) as f:
            return json.load(f)["api_key"]
    return os.environ.get("RUNPOD_AI_API_KEY")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--db", default=os.path.expanduser(
        "~/webmind-research/trained_model/saqt.db"
    ))
    parser.add_argument("--output", default=os.path.expanduser(
        "~/gyan-app/data/gpu_built"
    ))
    parser.add_argument("--gpu", default="NVIDIA RTX A4000",
                        help="GPU type (default: A4000, cheapest option)")
    args = parser.parse_args()

    api_key = args.api_key or get_api_key()
    if not api_key:
        print("Error: No RunPod API key found")
        sys.exit(1)

    os.environ["RUNPOD_AI_API_KEY"] = api_key

    try:
        import runpod
        runpod.api_key = api_key
    except ImportError:
        print("Installing runpod...")
        subprocess.run([sys.executable, "-m", "pip", "install", "runpod"], check=True)
        import runpod
        runpod.api_key = api_key

    # Check available GPUs and pricing
    print("Checking GPU availability...")
    gpus = runpod.get_gpus()
    for g in gpus:
        if "A4000" in g.get("id", "") or "RTX" in g.get("id", ""):
            print(f"  {g['id']}: ${g.get('securePrice', '?')}/hr")

    print(f"\nCreating pod with {args.gpu}...")
    print("This will cost ~$0.20-0.50 for the build.\n")

    # Create pod
    pod = runpod.create_pod(
        name="gyan-kb-build",
        image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        gpu_type_id="NVIDIA RTX A4000",
        cloud_type="SECURE",
        gpu_count=1,
        volume_in_gb=10,
        container_disk_in_gb=20,
        min_vcpu_count=4,
        min_memory_in_gb=16,
    )

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    print("Waiting for pod to be ready...")

    # Wait for pod to be running
    for i in range(60):
        status = runpod.get_pod(pod_id)
        state = status.get("desiredStatus", "unknown")
        runtime = status.get("runtime", {})
        if runtime and runtime.get("uptimeInSeconds", 0) > 5:
            print(f"Pod ready! IP: {runtime.get('ports', [])}")
            break
        print(f"  Status: {state}... ({i*5}s)")
        time.sleep(5)
    else:
        print("Timeout waiting for pod. Check RunPod dashboard.")
        sys.exit(1)

    # Get SSH connection info
    ssh_info = runtime.get("ports", [])
    print(f"SSH ports: {ssh_info}")
    print("\n--- Manual steps (if SSH automated fails) ---")
    print(f"1. scp {args.db} runpod:{pod_id}:/workspace/saqt.db")
    print(f"2. scp build_kb.py runpod:{pod_id}:/workspace/build_kb.py")
    print(f"3. ssh runpod:{pod_id} 'cd /workspace && pip install qdrant-client sentence-transformers && python3 build_kb.py --db saqt.db --output ./kb_out --gpu --format numpy'")
    print(f"4. scp runpod:{pod_id}:/workspace/kb_out/* {args.output}/")
    print(f"5. runpodctl stop pod {pod_id}")

    print(f"\nTo terminate: python3 -c \"import runpod; runpod.api_key='{api_key}'; runpod.terminate_pod('{pod_id}')\"")


if __name__ == "__main__":
    main()
