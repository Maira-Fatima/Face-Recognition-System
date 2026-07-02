"""
Full-pipeline evaluation script for comparing ArcFace, AdaFace, and EdgeFace
through the ACTUAL production storage layer (FAISS + SQLite via FaceStore),
not just raw in-memory embeddings.

For each engine, this script:
    1. Loads the engine (measures GPU memory used by loading the model).
    2. ENROLLS each known person's photos into that engine's FaceStore
       (writes to SQLite + that engine's own FAISS index on disk),
       measuring insert time.
    3. Runs held-out QUERY (probe) photos of the SAME known people through
       /search-equivalent logic, checking whether the correct person is
       returned above MATCH_THRESHOLD (True Accept) or not (False Reject).
    4. Runs photos of UNKNOWN people (never enrolled) through search too,
       checking whether the system correctly rejects them (True Reject)
       or wrongly matches them to someone (False Accept).
    5. Reports GPU memory, average enroll/search time, and accuracy
       (TAR / FRR / FAR) per engine.

Expected folder structure:
    test_images/
        person_a/
            img1.jpg   <- enrolled
            img2.jpg   <- enrolled
            img3.jpg   <- held out, used as genuine probe
        person_b/
            img1.jpg   <- enrolled
            img2.jpg   <- held out, used as genuine probe
        _unknown/                (optional but recommended)
            stranger1.jpg        <- never enrolled, used as impostor probe
            stranger2.jpg

For each person folder with N images, the LAST image (alphabetically) is
held out as the probe; the rest are enrolled. Folders starting with "_"
(e.g. "_unknown") are never enrolled — every image inside is used purely
as an impostor probe.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --test-dir test_images --engines arcface adaface
"""

import os
import sys
import shutil
import argparse
import time
import numpy as np
import cv2
import torch
import pynvml
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import get_engine, ENGINES  # noqa: E402
from app.storage import FaceStore  # noqa: E402
from app.config import settings  # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def get_gpu_mem_mb(handle) -> float:
    if handle is None:
        return 0.0
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)


def collect_dataset(root: str):
    """
    Returns (known, unknown):
        known   = {person_name: {"enroll": [paths], "probe": path}}
        unknown = [paths]  (impostor probes, never enrolled)
    """
    known = {}
    unknown = []

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Test image folder '{root}' not found.")

    for entry in sorted(os.listdir(root)):
        entry_path = os.path.join(root, entry)
        if not os.path.isdir(entry_path):
            continue

        files = sorted(
            os.path.join(entry_path, f)
            for f in os.listdir(entry_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not files:
            continue

        if entry.startswith("_"):
            unknown.extend(files)
            continue

        if len(files) < 2:
            print(f"  [skip] '{entry}' has only 1 image; need at least 2 "
                  f"(1+ to enroll, 1 held out as probe). Skipping this person.")
            continue

        known[entry] = {"enroll": files[:-1], "probe": files[-1]}

    return known, unknown


def reset_engine_data(engine_name: str):
    """Delete this engine's FAISS index so re-runs start from a clean state.
    SQLite rows are left in place (harmless leftovers, tagged by engine),
    but the FAISS index controls what's actually searchable per engine.
    """
    faiss_path = os.path.join(settings.DATA_DIR, f"{engine_name}_faiss.bin")
    if os.path.exists(faiss_path):
        os.remove(faiss_path)


def evaluate_engine(engine_name: str, known: dict, unknown: list, handle) -> dict:
    print(f"\n=== {engine_name} ===")
    torch.cuda.empty_cache()
    mem_before = get_gpu_mem_mb(handle)

    engine = get_engine(engine_name)
    mem_after_load = get_gpu_mem_mb(handle)

    reset_engine_data(engine_name)
    store = FaceStore(engine_name=engine_name)

    # --- Warmup: run once on the first available enroll image, discard result ---
    first_person = next(iter(known.values()))
    warmup_img = cv2.imread(first_person["enroll"][0])
    if warmup_img is not None:
        _ = engine.embed(warmup_img)

    # --- Enrollment ---
    enroll_times = []
    for person, data in known.items():
        for path in data["enroll"]:
            image = cv2.imread(path)
            if image is None:
                print(f"  [enroll] could not read {path}, skipping")
                continue

            embedding, detect_ms, embed_ms = engine.embed(image)
            if embedding is None:
                print(f"  [enroll] {person}/{os.path.basename(path)}: no face detected, skipping")
                continue

            start = time.perf_counter()
            store.insert(name=person, embedding=embedding.reshape(1, -1), image_path=path)
            insert_ms = (time.perf_counter() - start) * 1000.0
            enroll_times.append(detect_ms + embed_ms + insert_ms)
            print(f"  [enroll] {person}/{os.path.basename(path)}: "
                  f"detect={detect_ms:.2f}ms embed={embed_ms:.2f}ms store={insert_ms:.2f}ms")

    # --- Genuine probes (known people, held-out photo) ---
    search_times = []
    true_accepts, false_rejects = 0, 0
    for person, data in known.items():
        path = data["probe"]
        image = cv2.imread(path)
        if image is None:
            continue

        embedding, detect_ms, embed_ms = engine.embed(image)
        if embedding is None:
            print(f"  [probe] {person}/{os.path.basename(path)}: no face detected, skipping")
            continue

        start = time.perf_counter()
        matches = store.search(embedding.reshape(1, -1), top_k=1)
        search_ms = (time.perf_counter() - start) * 1000.0
        search_times.append(detect_ms + embed_ms + search_ms)

        best = matches[0] if matches else None
        if best and best["name"] == person and best["similarity"] >= settings.MATCH_THRESHOLD:
            true_accepts += 1
            result = "TRUE ACCEPT"
        else:
            false_rejects += 1
            result = "FALSE REJECT"
        sim = f"{best['similarity']:.4f}" if best else "N/A"
        print(f"  [probe/genuine] {person}/{os.path.basename(path)}: "
              f"matched='{best['name'] if best else None}' sim={sim} -> {result}")

    # --- Impostor probes (unknown people, never enrolled) ---
    true_rejects, false_accepts = 0, 0
    for path in unknown:
        image = cv2.imread(path)
        if image is None:
            continue

        embedding, detect_ms, embed_ms = engine.embed(image)
        if embedding is None:
            print(f"  [probe/unknown] {os.path.basename(path)}: no face detected, skipping")
            continue

        start = time.perf_counter()
        matches = store.search(embedding.reshape(1, -1), top_k=1)
        search_ms = (time.perf_counter() - start) * 1000.0
        search_times.append(detect_ms + embed_ms + search_ms)

        best = matches[0] if matches else None
        if best and best["similarity"] >= settings.MATCH_THRESHOLD:
            false_accepts += 1
            result = "FALSE ACCEPT"
        else:
            true_rejects += 1
            result = "TRUE REJECT"
        sim = f"{best['similarity']:.4f}" if best else "N/A"
        print(f"  [probe/unknown] {os.path.basename(path)}: "
              f"matched='{best['name'] if best else None}' sim={sim} -> {result}")

    mem_peak = get_gpu_mem_mb(handle)
    del engine
    torch.cuda.empty_cache()

    n_genuine = true_accepts + false_rejects
    n_impostor = true_rejects + false_accepts

    return {
        "engine": engine_name,
        "avg_enroll_ms": np.mean(enroll_times) if enroll_times else None,
        "avg_search_ms": np.mean(search_times) if search_times else None,
        "model_mem_mb": mem_after_load - mem_before,
        "peak_mem_mb": mem_peak - mem_before,
        "tar": true_accepts / n_genuine if n_genuine else None,   # True Accept Rate
        "frr": false_rejects / n_genuine if n_genuine else None,  # False Reject Rate
        "far": false_accepts / n_impostor if n_impostor else None,  # False Accept Rate
        "trr": true_rejects / n_impostor if n_impostor else None,   # True Reject Rate
        "n_genuine": n_genuine,
        "n_impostor": n_impostor,
    }


def print_results_table(all_results: dict):
    rows = []
    for name, r in all_results.items():
        enroll = f"{r['avg_enroll_ms']:.2f} ms" if r["avg_enroll_ms"] is not None else "N/A"
        search = f"{r['avg_search_ms']:.2f} ms" if r["avg_search_ms"] is not None else "N/A"
        tar = f"{r['tar']*100:.1f}% ({r['n_genuine']})" if r["tar"] is not None else "N/A"
        far = f"{r['far']*100:.1f}% ({r['n_impostor']})" if r["far"] is not None else "N/A"
        rows.append([
            r["engine"], enroll, search, f"{r['model_mem_mb']:.1f} MB", tar, far,
        ])

    print("\n" + "=" * 90)
    print("FULL-PIPELINE EVALUATION RESULTS (via real engine.py + FAISS + SQLite)")
    print("=" * 90)
    print(tabulate(
        rows,
        headers=["Engine", "Avg Enroll", "Avg Search", "GPU Mem (load)",
                 "True Accept Rate (n)", "False Accept Rate (n)"],
    ))
    print("\nTrue Accept Rate: genuine probe correctly matched to the right person, above threshold.")
    print("False Accept Rate: an unknown/impostor probe incorrectly matched to someone, above threshold.")
    print(f"Match threshold in use: {settings.MATCH_THRESHOLD}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Full-pipeline face recognition evaluation.")
    parser.add_argument("--test-dir", default="test_images",
                         help="Folder with one subfolder per known person, plus optional '_unknown' folder.")
    parser.add_argument("--engines", nargs="+", default=list(ENGINES.keys()),
                         help=f"Which engines to evaluate (default: all of {list(ENGINES.keys())})")
    args = parser.parse_args()

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        print("No GPU detected (pynvml init failed) -- running on CPU, GPU memory will show as 0.")
        handle = None

    known, unknown = collect_dataset(args.test_dir)
    print("Known people (enroll/probe split):",
          {k: f"{len(v['enroll'])} enroll + 1 probe" for k, v in known.items()})
    print(f"Unknown/impostor probes: {len(unknown)}")

    if not known:
        print(f"No usable person folders found under '{args.test_dir}' "
              f"(each needs 2+ images). Add more photos and re-run.")
        return

    all_results = {}
    for engine_name in args.engines:
        if engine_name not in ENGINES:
            print(f"Skipping unknown engine '{engine_name}'")
            continue
        all_results[engine_name] = evaluate_engine(engine_name, known, unknown, handle)

    print_results_table(all_results)


if __name__ == "__main__":
    main()
