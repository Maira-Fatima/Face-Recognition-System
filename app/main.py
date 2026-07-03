import os
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import time

from app.config import settings
from app.engine import get_engine, ENGINES, get_scene_engine
from app.storage import FaceStore, SceneStore

app = FastAPI(title="Face Recognition API")

# Load ALL engines and stores at startup, so /insert can write to every
# engine's FAISS index, and /search can choose any engine per-request.
# Kept in dicts keyed by engine name, e.g. engines["arcface"].
engines: dict = {}
stores: dict = {}

# Scene/location recognition -- a single shared engine + store, since
# (unlike faces) there's only one embedding model in play here.
scene_engine = None
scene_store: SceneStore = None

# Serve saved scene photos back over HTTP so /search_scene results are
# directly viewable, e.g. http://localhost:8000/scene_images/<file>.jpg
app.mount("/scene_images", StaticFiles(directory=settings.SCENE_IMAGE_DIR), name="scene_images")


@app.on_event("startup")
async def startup_event():
    global engines, stores, scene_engine, scene_store
    for engine_name in ENGINES.keys():
        print(f"Loading engine: {engine_name}...")
        engines[engine_name] = get_engine(engine_name)
        stores[engine_name] = FaceStore(engine_name=engine_name)

    print("Loading scene engine (YOLOv8-seg + DINOv2)...")
    scene_engine = get_scene_engine()
    scene_store = SceneStore()

    print(f"Startup complete. Loaded engines: {list(engines.keys())} + scene engine")


def _save_scene_image(image: np.ndarray) -> str:
    """Saves the original (unmasked) photo to disk so it can be returned
    later as 'the matching photo' on a scene search hit."""
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(settings.SCENE_IMAGE_DIR, filename)
    cv2.imwrite(filepath, image)
    return filepath


async def _process_image(file: UploadFile) -> np.ndarray:
    """Helper to read UploadFile into an OpenCV numpy array (BGR)."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")
    return image


@app.get("/health")
def health_check():
    if not engines or not stores:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    return {
        "status": "healthy",
        "engines_loaded": list(engines.keys()),
        "total_faces_stored": {
            name: store.get_total_faces() for name, store in stores.items()
        },
        "scene_engine_loaded": scene_engine is not None,
        "total_scenes_stored": scene_store.get_total_scenes() if scene_store else 0,
    }


@app.post("/insert")
async def insert_face(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Runs the uploaded photo through ALL THREE engines (ArcFace, AdaFace,
    EdgeFace), storing a separate embedding into each engine's own FAISS
    index + SQLite record. This lets the SAME enrolled photo be searchable
    later under any of the three engines.
    """
    try:
        image = await _process_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = {}
    any_success = False

    for engine_name, engine in engines.items():
        embedding, detect_ms, embed_ms = engine.embed(image)

        if embedding is None:
            results[engine_name] = {
                "success": False,
                "error": "No face detected",
            }
            continue

        record_id = stores[engine_name].insert(name=name, embedding=embedding)
        any_success = True
        results[engine_name] = {
            "success": True,
            "record_id": record_id,
            "detect_ms": round(detect_ms, 2),
            "embed_ms": round(embed_ms, 2),
        }

    if not any_success:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the image by any engine",
        )

    # --- Scene/location pipeline ---
    # Only runs because a face was already confirmed present above -- this
    # is how we enforce "there must be a person in the picture" for scene
    # enrollment without needing a separate face check inside SceneEngine.
    scene_result = None
    try:
        scene_embedding, mask_ms, scene_embed_ms = scene_engine.embed(image)
        saved_path = _save_scene_image(image)
        scene_id = scene_store.insert(
            embedding=scene_embedding, image_path=saved_path, person_name=name
        )
        scene_result = {
            "success": True,
            "scene_id": scene_id,
            "image_path": saved_path,
            "mask_ms": round(mask_ms, 2),
            "embed_ms": round(scene_embed_ms, 2),
        }
    except Exception as e:
        scene_result = {"success": False, "error": str(e)}

    return {
        "name": name,
        "results_by_engine": results,
        "scene": scene_result,
    }


@app.post("/search")
async def search_face(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    engine: str = Form(
        default=settings.DEFAULT_ENGINE,
        description=f"Which engine to search with. One of: {list(ENGINES.keys())}",
    ),
):
    """
    Searches using a SINGLE chosen engine (defaults to settings.DEFAULT_ENGINE
    if not specified). Pick "arcface", "adaface", or "edgeface" directly in
    Swagger to compare results without editing config.py or restarting.
    """
    if engine not in engines:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown engine '{engine}'. Available: {list(engines.keys())}",
        )

    start_time = time.perf_counter()

    try:
        image = await _process_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    active_engine = engines[engine]
    active_store = stores[engine]

    embedding, detect_ms, embed_ms = active_engine.embed(image)

    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    search_start = time.perf_counter()
    matches = active_store.search(embedding=embedding, top_k=top_k)
    search_ms = (time.perf_counter() - search_start) * 1000.0

    total_ms = (time.perf_counter() - start_time) * 1000.0

    best_match = None
    if matches and matches[0]["similarity"] >= settings.MATCH_THRESHOLD:
        best_match = matches[0]["name"]

    return {
        "engine_used": engine,
        "matches": matches,
        "best_match": best_match,
        "detect_ms": round(detect_ms, 2),
        "embed_ms": round(embed_ms, 2),
        "search_ms": round(search_ms, 2),
        "total_ms": round(total_ms, 2),
    }


@app.post("/search_all")
async def search_face_all_engines(
    file: UploadFile = File(...),
    top_k: int = Form(5),
):
    """
    Runs the SAME uploaded photo through all three engines and searches
    each engine's own FAISS index, returning a side-by-side comparison:
    best match, similarity (accuracy signal), and detect/embed timing
    for ArcFace, AdaFace, and EdgeFace in a single response.
    """
    try:
        image = await _process_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = {}

    for engine_name, active_engine in engines.items():
        active_store = stores[engine_name]

        embedding, detect_ms, embed_ms = active_engine.embed(image)

        if embedding is None:
            results[engine_name] = {
                "success": False,
                "error": "No face detected",
            }
            continue

        search_start = time.perf_counter()
        matches = active_store.search(embedding=embedding, top_k=top_k)
        search_ms = (time.perf_counter() - search_start) * 1000.0

        best_match = None
        best_similarity = None
        if matches:
            best_similarity = round(matches[0]["similarity"], 4)
            if matches[0]["similarity"] >= settings.MATCH_THRESHOLD:
                best_match = matches[0]["name"]

        results[engine_name] = {
            "success": True,
            "best_match": best_match,
            "best_similarity": best_similarity,
            "matches": matches,
            "detect_ms": round(detect_ms, 2),
            "embed_ms": round(embed_ms, 2),
            "search_ms": round(search_ms, 2),
            "total_ms": round(detect_ms + embed_ms + search_ms, 2),
        }

    return {
        "match_threshold": settings.MATCH_THRESHOLD,
        "results_by_engine": results,
    }


@app.post("/search_scene")
async def search_scene(
    file: UploadFile = File(...),
    top_k: int = Form(3),
    face_check_engine: str = Form(
        default=settings.DEFAULT_ENGINE,
        description="Which face engine to use just to confirm a person is present.",
    ),
):
    """
    Scene/location search: 'have I seen this background before, even with
    a different person in front of it?'

    A face must still be detected in the query photo (same rule as
    /insert) -- but the actual MATCH is done purely on the background,
    ignoring who the person is. On a hit, returns the ORIGINAL stored
    photo (via image_path / the /scene_images static URL), not a location
    name -- there is no location label anywhere in this system.
    """
    if face_check_engine not in engines:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown engine '{face_check_engine}'. Available: {list(engines.keys())}",
        )

    try:
        image = await _process_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Gate: require a person in the query photo too, same as /insert.
    face_embedding, detect_ms, _ = engines[face_check_engine].embed(image)
    if face_embedding is None:
        raise HTTPException(
            status_code=400,
            detail="No face detected in the query image -- a person must be present.",
        )

    start = time.perf_counter()
    scene_embedding, mask_ms, embed_ms = scene_engine.embed(image)
    matches = scene_store.search(embedding=scene_embedding, top_k=top_k)
    total_ms = (time.perf_counter() - start) * 1000.0

    # Attach a browsable URL for each match's stored photo.
    for m in matches:
        filename = os.path.basename(m["image_path"])
        m["image_url"] = f"/scene_images/{filename}"

    best_match = None
    if matches and matches[0]["similarity"] >= settings.SCENE_MATCH_THRESHOLD:
        best_match = matches[0]

    return {
        "location_recognized": best_match is not None,
        "best_match": best_match,
        "matches": matches,
        "match_threshold": settings.SCENE_MATCH_THRESHOLD,
        "face_detect_ms": round(detect_ms, 2),
        "mask_ms": round(mask_ms, 2),
        "embed_ms": round(embed_ms, 2),
        "total_ms": round(total_ms, 2),
    }
