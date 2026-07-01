import time
from abc import ABC, abstractmethod
import numpy as np
import cv2
import torch
from insightface.app import FaceAnalysis

class BaseEngine(ABC):
    """
    Abstract base class for all face recognition engines.
    Ensures a consistent interface for benchmarking different models.
    """
    @abstractmethod
    def embed(self, image: np.ndarray):
        """
        Detects a face and extracts an L2-normalized embedding.
        
        Args:
            image (np.ndarray): The input image (BGR format).
            
        Returns:
            tuple: (embedding, detect_ms, embed_ms)
                embedding: np.ndarray of shape (1, embedding_dim) or None if no face detected.
                detect_ms: float, time taken for detection in milliseconds.
                embed_ms: float, time taken for embedding in milliseconds.
        """
        pass

class ArcFaceEngine(BaseEngine):
    """
    ArcFace implementation using the 'buffalo_l' model pack from insightface.
    Uses SCRFD for detection and ArcFace for recognition.
    """
    def __init__(self):
        # Determine GPU availability
        # ctx_id=0 means GPU, ctx_id=-1 means CPU
        ctx_id = 0 if torch.cuda.is_available() else -1
        
        # Initialize the FaceAnalysis app with buffalo_l
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # To separate detection and recognition timing, we will interact with the models directly.
        # However, insightface FaceAnalysis `.get()` does both. We can simulate or measure
        # it by running them sequentially.
        # self.app.models contains 'detection' (SCRFD) and 'recognition' (ArcFace)
        self.det_model = self.app.models['detection']
        self.rec_model = self.app.models['recognition']
        
    def embed(self, image: np.ndarray):
        # 1. Detection
        start_det = time.perf_counter()
        # scrfd returns bboxes, kpss
        bboxes, kpss = self.det_model.detect(image, max_num=1, metric='default')
        detect_ms = (time.perf_counter() - start_det) * 1000.0
        
        if bboxes.shape[0] == 0:
            return None, detect_ms, 0.0
            
        # Get the first face detected
        bbox = bboxes[0, 0:4]
        det_score = bboxes[0, 4]
        kps = kpss[0] if kpss is not None else None
        
        # We need a Face object or we can just pass the bbox/kps directly if insightface supports it
        # insightface rec_model.get(image, face) expects a face object with bbox and kps.
        class FakeFace:
            def __init__(self, bbox, kps, det_score):
                self.bbox = bbox
                self.kps = kps
                self.det_score = det_score
        
        face = FakeFace(bbox, kps, det_score)
        
        # 2. Embedding
        start_embed = time.perf_counter()
        self.rec_model.get(image, face)
        embedding = face.embedding
        embed_ms = (time.perf_counter() - start_embed) * 1000.0
        
        # Insightface returns L2-normalized embeddings by default, but we enforce it just in case
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding, detect_ms, embed_ms

# Engine registry for pluggable architecture
ENGINES = {
    "arcface": ArcFaceEngine,
    # "adaface": AdaFaceEngine, # To be added in Phase 3
    # "edgeface": EdgeFaceEngine, # To be added in Phase 3
}

def get_engine(engine_name: str) -> BaseEngine:
    if engine_name not in ENGINES:
        raise ValueError(f"Engine {engine_name} not found. Available engines: {list(ENGINES.keys())}")
    # Instantiate the engine lazily
    return ENGINES[engine_name]()
