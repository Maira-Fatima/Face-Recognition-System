import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    DB_PATH: str = os.path.join(DATA_DIR, "facestore.sqlite")
    FAISS_INDEX_PATH: str = os.path.join(DATA_DIR, "faiss_index.bin")

    # Engine Settings
    DEFAULT_ENGINE: str = "arcface"
    MATCH_THRESHOLD: float = 0.5  # Cosine similarity threshold (0.0 to 1.0)
    
    # Model configs
    EMBEDDING_DIM: int = 512

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
