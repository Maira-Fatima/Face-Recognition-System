import time
from abc import ABC, abstractmethod
import numpy as np
import cv2
import torch
from insightface.app import FaceAnalysis
from uniface.detection import RetinaFace
from uniface.recognition import AdaFace, EdgeFace

from app.config import settings

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


class SceneEngine:
    """
    Generates an embedding of the LOCATION/BACKGROUND in an image, with any
    people masked out first. This is intentionally separate from BaseEngine:
    it is not identifying a person, it is identifying a place, and the
    caller (main.py) is responsible for making sure a face was detected
    before this engine is ever invoked -- that's how we enforce "a person
    must be present in the photo" without baking a face-detector into a
    scene/background model.

    Pipeline: YOLOv8-seg (mask out every 'person' instance) -> DINOv2
    (embed the person-less background) -> L2-normalized 768-dim vector,
    stored/searched via cosine similarity, exactly like the face engines.
    """

    def __init__(self):
        from ultralytics import YOLO
        from torchvision import transforms

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Person segmentation model (COCO class 0 = 'person')
        self.segmenter = YOLO(settings.YOLO_SEG_MODEL)

        # Frozen, pretrained DINOv2 -- no fine-tuning, used purely for
        # embedding + nearest-neighbour retrieval (same philosophy as
        # ArcFace/AdaFace/EdgeFace being used pretrained, not trained here).
        self.model = torch.hub.load("facebookresearch/dinov2", settings.DINOV2_MODEL_NAME)
        self.model.to(self.device).eval()

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _mask_out_people(self, image_bgr: np.ndarray) -> np.ndarray:
        """Blacks out every detected person instance so only background remains."""
        results = self.segmenter(image_bgr, classes=[0], verbose=False)  # class 0 = person
        masked = image_bgr.copy()

        if results and results[0].masks is not None:
            h, w = masked.shape[:2]
            for mask in results[0].masks.data:
                mask_np = mask.cpu().numpy().astype(np.uint8)
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                masked[mask_resized] = 0

        return masked

    def embed(self, image_bgr: np.ndarray):
        """
        Masks out people, then embeds the remaining background with DINOv2.

        Args:
            image_bgr (np.ndarray): input image, BGR (same convention as
                the face engines / cv2.imdecode output).

        Returns:
            tuple: (embedding, mask_ms, embed_ms)
                embedding: np.ndarray of shape (768,), L2-normalized.
                mask_ms: float, time spent on person segmentation.
                embed_ms: float, time spent on DINOv2 embedding.
        """
        start_mask = time.perf_counter()
        masked_bgr = self._mask_out_people(image_bgr)
        mask_ms = (time.perf_counter() - start_mask) * 1000.0

        start_embed = time.perf_counter()
        from PIL import Image
        rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = self._transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        embed_ms = (time.perf_counter() - start_embed) * 1000.0

        embedding = embedding.cpu().numpy().flatten().astype(np.float32)
        return embedding, mask_ms, embed_ms


_scene_engine_singleton: "SceneEngine | None" = None


def get_scene_engine() -> SceneEngine:
    """Lazily builds and caches a single SceneEngine (mirrors get_engine() for face engines)."""
    global _scene_engine_singleton
    if _scene_engine_singleton is None:
        _scene_engine_singleton = SceneEngine()
    return _scene_engine_singleton