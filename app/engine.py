import time
from abc import ABC, abstractmethod
import numpy as np
import cv2
import torch
from insightface.app import FaceAnalysis
from uniface.detection import RetinaFace
from uniface.recognition import AdaFace, EdgeFace

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
        ctx_id = 0 if torch.cuda.is_available() else -1
        
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        self.det_model = self.app.models['detection']
        self.rec_model = self.app.models['recognition']
        
    def embed(self, image: np.ndarray):
        start_det = time.perf_counter()
        bboxes, kpss = self.det_model.detect(image, max_num=1, metric='default')
        detect_ms = (time.perf_counter() - start_det) * 1000.0
        
        if bboxes.shape[0] == 0:
            return None, detect_ms, 0.0
            
        bbox = bboxes[0, 0:4]
        det_score = bboxes[0, 4]
        kps = kpss[0] if kpss is not None else None
        
        class FakeFace:
            def __init__(self, bbox, kps, det_score):
                self.bbox = bbox
                self.kps = kps
                self.det_score = det_score
        
        face = FakeFace(bbox, kps, det_score)
        
        start_embed = time.perf_counter()
        self.rec_model.get(image, face)
        embedding = face.embedding
        embed_ms = (time.perf_counter() - start_embed) * 1000.0
        
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding, detect_ms, embed_ms


class _UniFaceEngineBase(BaseEngine):
    """
    Shared base for uniface-backed engines (AdaFace, EdgeFace).
    Uses RetinaFace for detection, and the given recognizer class for embedding.
    A single RetinaFace detector is shared across all uniface-based engines
    to avoid loading it multiple times if several engines are active.
    """
    _shared_detector = None

    def __init__(self, recognizer_cls):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        if _UniFaceEngineBase._shared_detector is None:
            _UniFaceEngineBase._shared_detector = RetinaFace(providers=providers)
        self.detector = _UniFaceEngineBase._shared_detector

        self.recognizer = recognizer_cls(providers=providers)

    def embed(self, image: np.ndarray):
        start_det = time.perf_counter()
        faces = self.detector.detect(image)
        detect_ms = (time.perf_counter() - start_det) * 1000.0

        if not faces:
            return None, detect_ms, 0.0

        start_embed = time.perf_counter()
        embedding = self.recognizer.get_normalized_embedding(image, faces[0].landmarks)
        embed_ms = (time.perf_counter() - start_embed) * 1000.0

        embedding = embedding / np.linalg.norm(embedding)

        return embedding, detect_ms, embed_ms


class AdaFaceEngine(_UniFaceEngineBase):
    """
    AdaFace implementation via the uniface library.
    Uses RetinaFace for detection and AdaFace for recognition.
    """
    def __init__(self):
        super().__init__(AdaFace)


class EdgeFaceEngine(_UniFaceEngineBase):
    """
    EdgeFace implementation via the uniface library.
    Uses RetinaFace for detection and EdgeFace for recognition.
    Optimized for lightweight/edge deployment.
    """
    def __init__(self):
        super().__init__(EdgeFace)


# Engine registry for pluggable architecture
ENGINES = {
    "arcface": ArcFaceEngine,
    "adaface": AdaFaceEngine,
    "edgeface": EdgeFaceEngine,
}

def get_engine(engine_name: str) -> BaseEngine:
    if engine_name not in ENGINES:
        raise ValueError(f"Engine {engine_name} not found. Available engines: {list(ENGINES.keys())}")
    return ENGINES[engine_name]()