import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = "/content/drive/MyDrive/face_recognition_data"
    DB_PATH: str = os.path.join(DATA_DIR, "facestore.sqlite")
    FAISS_INDEX_PATH: str = os.path.join(DATA_DIR, "faiss_index.bin")

    # Engine Settings
    DEFAULT_ENGINE: str = "arcface"
    MATCH_THRESHOLD: float = 0.5  # Cosine similarity threshold (0.0 to 1.0)
    
    # Model configs
    EMBEDDING_DIM: int = 512

    # --- Scene / Location Recognition Settings ---
    # A scene is only ever stored/searched if a face was detected first
    # (enforced in main.py) -- this mirrors the mentor's requirement that
    # a person must be present in the frame.
    SCENE_IMAGE_DIR: str = os.path.join(DATA_DIR, "scene_images")
    SCENE_DB_PATH: str = os.path.join(DATA_DIR, "facestore.sqlite")  # same sqlite file, separate table
    SCENE_FAISS_INDEX_PATH: str = os.path.join(DATA_DIR, "scene_faiss.bin")
    SCENE_EMBEDDING_DIM: int = 768          # dinov2_vitb14 output dim
    SCENE_MATCH_THRESHOLD: float = 0.75     # cosine similarity cutoff -- tune via scripts/evaluate_scene.py
    YOLO_SEG_MODEL: str = "yolov8n-seg.pt"  # auto-downloads on first run
    DINOV2_MODEL_NAME: str = "dinov2_vitb14"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.SCENE_IMAGE_DIR, exist_ok=True)
