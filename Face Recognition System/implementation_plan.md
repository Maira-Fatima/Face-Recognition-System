# Face Recognition System Implementation Plan

## Goal Description

Build a comprehensive end-to-end Face Recognition service using FastAPI, FAISS, and SQLite. The project will feature a pluggable architecture to evaluate and benchmark three models: ArcFace, AdaFace, and EdgeFace. The core objective is to provide a robust API for face insertion and search while enabling detailed benchmarking of accuracy, processing time, and GPU consumption across models.

## User Review Required

- **`uniface` package**: The prompt mentions `uniface` as a pip-installable package for AdaFace and EdgeFace. While I will add it to the requirements, please confirm if this is available on the public PyPI registry or if it requires a specific GitHub repository link in the requirements file (e.g., `git+https://github.com/...`).
- **Environment Setup**: The initial GPU check failed because `torch` is not installed yet. I will create a `requirements.txt` file. Please ensure you install it within your preferred virtual environment (`pip install -r requirements.txt`) before we run the FastAPI service.

## Open Questions

- None at this moment, the requirements are very detailed. We will proceed with the phased execution as requested.

## Proposed Changes (Phases 1 & 2)

We will start with Step 1 and Step 2 as requested, pausing for verification before moving to Step 3.

### Project Structure & Dependencies

#### [NEW] `requirements.txt`
Dependencies including `fastapi`, `uvicorn`, `python-multipart`, `torch`, `faiss-cpu` (or `faiss-gpu` depending on final GPU check after install), `insightface`, `uniface`, `onnxruntime`, `sqlite3` (built-in), `opencv-python`, `tabulate`, `matplotlib`, `pynvml`.

### Application Core

#### [NEW] `app/config.py`
Central configuration defining paths for data/models, default engine settings, and the similarity match threshold.

#### [NEW] `app/engine.py`
- `BaseEngine`: Abstract base class defining the `embed(image) -> (embedding, detect_ms, embed_ms)` interface.
- `ArcFaceEngine`: Implementation using `insightface` (`buffalo_l`). It will use SCRFD for detection and ArcFace for embedding. Returns L2-normalized 512-d vectors.

#### [NEW] `app/storage.py`
- `FaceStore`: Manages a FAISS `IndexFlatIP` (wrapped in `IndexIDMap`) for cosine similarity, paired with a SQLite database for metadata (id, name, image_path, engine, inserted_at). It will persist the index to disk after every insert.

#### [NEW] `app/main.py`
FastAPI application exposing:
- `GET /health`: Engine status and total faces stored.
- `POST /insert`: Form data (name, file) -> detect, embed, store, return ID and timing.
- `POST /search`: Form data (file, top_k) -> detect, embed, search FAISS, return ranked matches and timing.

## Verification Plan

### Phase 1 & 2 Verification
1. Install dependencies from `requirements.txt`.
2. Run a script to definitively confirm GPU availability (CUDA).
3. Start the FastAPI server locally (`uvicorn app.main:app --reload`).
4. Test the `/health` endpoint to ensure the ArcFace model loads successfully.
5. Perform test `/insert` calls with sample images.
6. Perform test `/search` calls to verify FAISS similarity ranking and response formatting.

*Once Phase 1 & 2 are verified and approved by you, we will proceed to subsequent phases (AdaFace/EdgeFace integration, bulk insert script, and evaluation script).*
