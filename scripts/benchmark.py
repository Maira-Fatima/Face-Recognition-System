"""
Benchmark script for comparing face recognition engines (ArcFace, AdaFace, EdgeFace).

Measures, per engine:
    - Average detect + embed time (ms), after discarding the first (warmup) call
    - GPU memory delta (MB) from loading the model
    - Genuine similarity (same person, different photos)
    - Impostor similarity (different people)

Uses app.engine.get_engine() directly, so results reflect the real
production code path rather than a standalone/ad hoc script.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --test-dir test_images
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import pynvml
from tabulate import tabulate

# Make sure the project root (parent of scripts/) is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import get_engine, ENGINES  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_gpu_mem_mb(handle) -> float:
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def collect_images(root: str) -> dict:
    """Returns {person_name: [image_paths]} for every subfolder under root."""
    people = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Test image folder '{root}' not found. "
            f"Create it with one subfolder per person, e.g. {root}/alice/img1.jpg"
        )
    for person in sorted(os.listdir(root)):
        p_dir = os.path.join(root, person)
        if not os.path.isdir(p_dir):
            continue
        files = [
            os.path.join(p_dir, f)
            for f in sorted(os.listdir(p_dir))
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if files:
            people[person] = files
    return people


def benchmark_engine(engine_name: str, people: dict, handle) -> dict:
    print(f"\n=== {engine_name} ===")
    torch.cuda.empty_cache()
    mem_before = get_gpu_mem_mb(handle)

    engine = get_engine(engine_name)
    mem_after_load = get_gpu_mem_mb(handle)

    embeddings = {}
    detect_times, embed_times = [], []
    warmed_up = False

    for person, paths in people.items():
        embeddings[person] = []
        for path in paths:
            image = cv2.imread(path)
            if image is None:
                print(f"  {person}/{os.path.basename(path)}: could not read image, skipping")
                continue

            embedding, detect_ms, embed_ms = engine.embed(image)

            if not warmed_up:
                # Discard the first call across the whole run: it includes
                # one-time CUDA/cuDNN algorithm-search warmup and would
                # otherwise skew this engine's numbers.
                warmed_up = True
                print(f"  {person}/{os.path.basename(path)}: warmup call (discarded)")
                continue

            if embedding is None:
                print(f"  {person}/{os.path.basename(path)}: no face detected")
                continue

            embeddings[person].append(embedding)
            detect_times.append(detect_ms)
            embed_times.append(embed_ms)
            print(
                f"  {person}/{os.path.basename(path)}: "
                f"detect={detect_ms:.2f}ms embed={embed_ms:.2f}ms"
            )

    mem_peak = get_gpu_mem_mb(handle)

    # Genuine (same person) vs impostor (different people) similarity
    genuine_sims, impostor_sims = [], []
    people_list = list(embeddings.items())
    for i, (_, embs_a) in enumerate(people_list):
        for j in range(len(embs_a)):
            for k in range(j + 1, len(embs_a)):
                genuine_sims.append(cosine_sim(embs_a[j], embs_a[k]))
        for _, embs_b in people_list[i + 1:]:
            for ea in embs_a:
                for eb in embs_b:
                    impostor_sims.append(cosine_sim(ea, eb))

    del engine
    torch.cuda.empty_cache()

    return {
        "engine": engine_name,
        "avg_detect_ms": np.mean(detect_times) if detect_times else None,
        "avg_embed_ms": np.mean(embed_times) if embed_times else None,
        "model_mem_mb": mem_after_load - mem_before,
        "peak_mem_mb": mem_peak - mem_before,
        "genuine_sims": genuine_sims,
        "impostor_sims": impostor_sims,
    }


def print_results_table(all_results: dict):
    rows = []
    for name, r in all_results.items():
        avg_detect = f"{r['avg_detect_ms']:.2f} ms" if r["avg_detect_ms"] is not None else "N/A"
        avg_embed = f"{r['avg_embed_ms']:.2f} ms" if r["avg_embed_ms"] is not None else "N/A"
        genuine = f"{np.mean(r['genuine_sims']):.4f}" if r["genuine_sims"] else "N/A (need 2+ imgs/person)"
        impostor = f"{np.mean(r['impostor_sims']):.4f}" if r["impostor_sims"] else "N/A (need 2+ people)"
        rows.append([
            r["engine"],
            avg_detect,
            avg_embed,
            f"{r['model_mem_mb']:.1f} MB",
            genuine,
            impostor,
        ])

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(tabulate(
        rows,
        headers=["Engine", "Avg Detect", "Avg Embed", "GPU Mem (load)", "Genuine Sim", "Impostor Sim"],
    ))
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark face recognition engines.")
    parser.add_argument(
        "--test-dir", default="test_images",
        help="Folder containing one subfolder per person (default: test_images)",
    )
    parser.add_argument(
        "--engines", nargs="+", default=list(ENGINES.keys()),
        help=f"Which engines to benchmark (default: all of {list(ENGINES.keys())})",
    )
    args = parser.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    people = collect_images(args.test_dir)
    print("Found test images:", {k: len(v) for k, v in people.items()})

    if not people:
        print(f"No images found under '{args.test_dir}'. Add subfolders with images per person and re-run.")
        return

    all_results = {}
    for engine_name in args.engines:
        if engine_name not in ENGINES:
            print(f"Skipping unknown engine '{engine_name}' (available: {list(ENGINES.keys())})")
            continue
        all_results[engine_name] = benchmark_engine(engine_name, people, handle)

    print_results_table(all_results)


if __name__ == "__main__":
    main()
