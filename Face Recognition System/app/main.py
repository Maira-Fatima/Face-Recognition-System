from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import time

from app.config import settings
from app.engine import get_engine
from app.storage import FaceStore

app = FastAPI(title="Face Recognition API")

# Initialize global engine and storage based on default config
# We can make this dynamic per request if needed, but for Phase 1/2 we'll use ArcFace
engine = None
store = None

@app.on_event("startup")
async def startup_event():
    global engine, store
    print(f"Loading engine: {settings.DEFAULT_ENGINE}...")
    engine = get_engine(settings.DEFAULT_ENGINE)
    store = FaceStore(engine_name=settings.DEFAULT_ENGINE)
    print("Startup complete.")

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
    if not engine or not store:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    return {
        "status": "healthy",
        "engine": settings.DEFAULT_ENGINE,
        "total_faces_stored": store.get_total_faces()
    }

@app.post("/insert")
async def insert_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        image = await _process_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    embedding, detect_ms, embed_ms = engine.embed(image)
    
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")
        
    # Store in FAISS and SQLite
    record_id = store.insert(name=name, embedding=embedding)
    
    return {
        "record_id": record_id,
        "name": name,
        "detect_ms": round(detect_ms, 2),
        "embed_ms": round(embed_ms, 2),
        "total_inference_ms": round(detect_ms + embed_ms, 2)
    }

@app.post("/search")
async def search_face(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    start_time = time.perf_counter()
    
    try:
        image = await _process_image(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    embedding, detect_ms, embed_ms = engine.embed(image)
    
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")
        
    # Search FAISS
    search_start = time.perf_counter()
    matches = store.search(embedding=embedding, top_k=top_k)
    search_ms = (time.perf_counter() - search_start) * 1000.0
    
    total_ms = (time.perf_counter() - start_time) * 1000.0
    
    # Check best match against threshold
    best_match = None
    if matches and matches[0]["similarity"] >= settings.MATCH_THRESHOLD:
        best_match = matches[0]["name"]
        
    return {
        "matches": matches,
        "best_match": best_match,
        "detect_ms": round(detect_ms, 2),
        "embed_ms": round(embed_ms, 2),
        "search_ms": round(search_ms, 2),
        "total_ms": round(total_ms, 2)
    }
