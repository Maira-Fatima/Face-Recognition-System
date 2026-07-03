"""
No-label threshold helper for the scene gallery.

Since scene entries have no location name, we can't compute classic
TAR/FAR directly. Instead, this script finds the most similar PAIRS of
photos already sitting in the gallery and prints them ranked by cosine
similarity, along with their saved image paths so you can open the two
files and visually confirm: is this actually the same building/place?

How to use it:
    1. Run scripts/populate_scene_gallery.py first to fill the gallery.
    2. Run this script.
    3. Open the top-ranked pairs (highest similarity first) and check by
       eye whether they're genuinely the same location.
       - If pairs that ARE the same place score high (e.g. > 0.80) and
         pairs that are clearly different places score notably lower,
         set settings.SCENE_MATCH_THRESHOLD somewhere in that gap.
       - If your dataset has few/no true duplicate locations, instead
         test with a deliberately altered copy of one existing photo
         (crop/rotate/different angle) as your query -- see the
         --self-test flag below.

Usage:
    python scripts/inspect_scene_similarities.py
    python scripts/inspect_scene_similarities.py --top-n 20
    python scripts/inspect_scene_similarities.py --self-test path/to/query.jpg
"""

import os
import sys
import argparse
import sqlite3

import numpy as np
import faiss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings          # noqa: E402
from app.engine import get_scene_engine   # noqa: E402
import cv2                                # noqa: E402


def load_all_embeddings():
    index = faiss.read_index(settings.SCENE_FAISS_INDEX_PATH)
    n = index.ntotal
    if n == 0:
        return None, None, []
    # IndexIDMap wraps IndexFlatIP; reconstruct_n gives back stored vectors
    xb = np.zeros((n, settings.SCENE_EMBEDDING_DIM), dtype=np.float32)
    ids = []
    conn = sqlite3.connect(settings.SCENE_DB_PATH)
    cursor = conn.execute("SELECT id, image_path FROM scenes ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    id_to_path = {r[0]: r[1] for r in rows}

    # Rebuild ordered by the FAISS-assigned ids (which are the sqlite ids)
    all_ids = list(id_to_path.keys())
    for i, rid in enumerate(all_ids):
        xb[i] = index.reconstruct(rid)
        ids.append(rid)

    return xb, ids, id_to_path


def show_top_pairs(top_n: int):
    xb, ids, id_to_path = load_all_embeddings()
    if xb is None:
        print("Scene gallery is empty. Run scripts/populate_scene_gallery.py first.")
        return

    sims = xb @ xb.T
    np.fill_diagonal(sims, -1)

    pairs = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((ids[i], ids[j], float(sims[i][j])))
    pairs.sort(key=lambda x: -x[2])

    print(f"Top {top_n} most similar pairs in the gallery:\n")
    for rid_a, rid_b, score in pairs[:top_n]:
        print(f"{score:.4f}   {id_to_path[rid_a]}   <->   {id_to_path[rid_b]}")


def self_test(query_path: str, top_k: int = 5):
    scene_engine = get_scene_engine()
    image = cv2.imread(query_path)
    if image is None:
        print(f"Could not read {query_path}")
        return

    embedding, mask_ms, embed_ms = scene_engine.embed(image)
    index = faiss.read_index(settings.SCENE_FAISS_INDEX_PATH)
    if index.ntotal == 0:
        print("Scene gallery is empty. Run scripts/populate_scene_gallery.py first.")
        return

    scores, ids = index.search(embedding.reshape(1, -1), top_k)

    conn = sqlite3.connect(settings.SCENE_DB_PATH)
    print(f"Query: {query_path}  (mask={mask_ms:.0f}ms, embed={embed_ms:.0f}ms)\n")
    print(f"Top {top_k} matches:")
    for score, rid in zip(scores[0], ids[0]):
        if rid == -1:
            continue
        row = conn.execute("SELECT image_path FROM scenes WHERE id=?", (int(rid),)).fetchone()
        print(f"{score:.4f}   {row[0]}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--self-test", type=str, default=None,
                         help="Path to a query image to test against the existing gallery")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.self_test)
    else:
        show_top_pairs(args.top_n)
