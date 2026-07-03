"""
Bulk-inserts a flat folder of photos (e.g. scene_dataset/) into the scene
gallery -- one entry per photo, no location labels required or used.

For each image:
    1. Runs the default face engine's detector just to confirm a person
       is present (mirrors the mentor's requirement + the /insert rule).
       Images with no detectable face are skipped.
    2. Runs SceneEngine: masks out the person(s) with YOLOv8-seg, embeds
       the remaining background with DINOv2.
    3. Saves the ORIGINAL (unmasked) photo into data/scene_images/ and
       stores [embedding -> saved photo path] in the scene FAISS index +
       SQLite table.

Uses app.engine / app.storage directly (the real production code path),
same philosophy as scripts/benchmark.py.

Usage:
    python scripts/populate_scene_gallery.py --source-dir scene_dataset
    python scripts/populate_scene_gallery.py --source-dir scene_dataset --face-engine arcface
"""

import os
import sys
import argparse
import shutil
import uuid

import cv2
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings          # noqa: E402
from app.engine import get_engine, get_scene_engine  # noqa: E402
from app.storage import SceneStore        # noqa: E402

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="Flat folder of location photos")
    parser.add_argument("--face-engine", default=settings.DEFAULT_ENGINE,
                         help="Which face engine to use just to confirm a person is present")
    args = parser.parse_args()

    print(f"Loading face engine '{args.face_engine}' (for the person-presence gate)...")
    face_engine = get_engine(args.face_engine)

    print("Loading scene engine (YOLOv8-seg + DINOv2)...")
    scene_engine = get_scene_engine()
    scene_store = SceneStore()

    files = sorted(
        f for f in os.listdir(args.source_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    print(f"Found {len(files)} images in {args.source_dir}\n")

    rows = []
    inserted, skipped_no_face, skipped_error = 0, 0, 0

    for fname in files:
        fpath = os.path.join(args.source_dir, fname)
        image = cv2.imread(fpath)
        if image is None:
            rows.append([fname, "ERROR", "could not read image"])
            skipped_error += 1
            continue

        face_embedding, _, _ = face_engine.embed(image)
        if face_embedding is None:
            rows.append([fname, "SKIPPED", "no face detected"])
            skipped_no_face += 1
            continue

        try:
            scene_embedding, mask_ms, embed_ms = scene_engine.embed(image)
        except Exception as e:
            rows.append([fname, "ERROR", str(e)])
            skipped_error += 1
            continue

        # Save the ORIGINAL (unmasked) photo -- this is what gets shown
        # back to the user on a future match.
        saved_name = f"{uuid.uuid4().hex}.jpg"
        saved_path = os.path.join(settings.SCENE_IMAGE_DIR, saved_name)
        shutil.copy(fpath, saved_path) if fpath.lower().endswith(".jpg") else cv2.imwrite(saved_path, image)

        scene_id = scene_store.insert(embedding=scene_embedding, image_path=saved_path)
        rows.append([fname, "INSERTED", f"scene_id={scene_id}  mask={mask_ms:.0f}ms  embed={embed_ms:.0f}ms"])
        inserted += 1

    print(tabulate(rows, headers=["file", "status", "detail"], tablefmt="github"))
    print(f"\nDone. Inserted: {inserted}  |  Skipped (no face): {skipped_no_face}  |  Errors: {skipped_error}")
    print(f"Total scenes now in gallery: {scene_store.get_total_scenes()}")


if __name__ == "__main__":
    main()
