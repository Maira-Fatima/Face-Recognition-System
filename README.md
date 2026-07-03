# Face Recognition System

An end-to-end face recognition service built with FastAPI, FAISS, and SQLite — plus a scene/location recognition extension that recognizes *where* a photo was taken, even when a different person is standing in front of the camera.

## Part 1 — Face Recognition

A pluggable-engine architecture for enrolling and searching faces, used to benchmark three recognition models side by side.

**Engines:** ArcFace (via `insightface`, `buffalo_l` pack, SCRFD detector), AdaFace and EdgeFace (via `uniface`, RetinaFace detector). All three implement a shared `BaseEngine.embed(image) -> (embedding, detect_ms, embed_ms)` interface, each with its own FAISS `IndexFlatIP` index and SQLite table, so the same enrolled photo is searchable under any engine and results are directly comparable.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Engine status, total faces stored per engine |
| `POST /insert` | Enrolls a photo under a name, across **all three** engines at once |
| `POST /search` | Searches with **one** chosen engine (`arcface` / `adaface` / `edgeface`) |
| `POST /search_all` | Searches the same photo across all three engines, side by side |

### Benchmarking

- `scripts/benchmark.py` — detect/embed timing and GPU memory per engine, plus genuine/impostor similarity, using the real production code path.
- `scripts/evaluate.py` — full enroll → search pipeline evaluation (TAR / FRR / FAR) against `test images/`.

Current results: **90% TAR / 0% FAR** across ArcFace, AdaFace, and EdgeFace on GPU (Tesla T4).

## Part 2 — Scene / Location Recognition

Extends the same photo pipeline to also answer: *"has this location been seen before, regardless of who's standing in it?"*

This is a **retrieval task, not classification** — there are no location names or labels anywhere in the system. Given a new photo, it either finds a close match to a previously stored photo of the same place, or it doesn't. No "trained" model in the fine-tuning sense either — like the face engines, it uses frozen pretrained models and does nearest-neighbour search.

**Pipeline:** a face must first be detected (enforced the same way as `/insert`, so a person must be present) → YOLOv8-seg masks out every person in the frame → DINOv2 embeds the remaining background → the embedding is stored/searched via cosine similarity in its own FAISS index, exactly like the face engines.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /insert` | Now also stores a scene embedding + the original photo, gated on a face being detected |
| `POST /search_scene` | Requires a face in the query photo, matches on background only, returns the original stored photo (via `/scene_images/<file>`) — never a location name |

### Populating the scene gallery

```bash
python scripts/populate_scene_gallery.py --source-dir path/to/scene_dataset
```

Bulk-loads a flat folder of photos (no location subfolders needed), skipping anything with no detectable face.

### Tuning the match threshold

Since there are no location labels to compute classic TAR/FAR against, use:

```bash
python scripts/inspect_scene_similarities.py --top-n 15
```

This prints the most similar photo pairs already in the gallery, ranked by cosine similarity, so you can eyeball real numbers and set `SCENE_MATCH_THRESHOLD` in `app/config.py` accordingly.

## Setup

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `/docs` for the interactive Swagger UI. First run will auto-download `yolov8n-seg.pt` and the DINOv2 weights.

## Project structure

```
app/
  main.py        FastAPI app — all endpoints
  engine.py       Face engines (ArcFace/AdaFace/EdgeFace) + SceneEngine
  storage.py      FaceStore + SceneStore (FAISS + SQLite)
  config.py       Paths, thresholds, model names
scripts/
  benchmark.py                     Face engine timing/GPU/accuracy benchmark
  evaluate.py                      Full-pipeline face TAR/FRR/FAR evaluation
  populate_scene_gallery.py        Bulk-insert a folder of scene photos
  inspect_scene_similarities.py    Threshold-tuning helper for scene matching
data/                              FAISS indexes, SQLite db, saved scene photos (gitignored)
test images/                       Sample enrollment/probe photos for face evaluation
```
