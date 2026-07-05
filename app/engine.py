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
        Detects a SINGLE face (the largest/most central one) and extracts
        an L2-normalized embedding. Used for /insert, where a photo is
        being enrolled under exactly one name.

        Args:
            image (np.ndarray): The input image (BGR format).
            
        Returns:
            tuple: (embedding, detect_ms, embed_ms)
                embedding: np.ndarray of shape (1, embedding_dim) or None if no face detected.
                detect_ms: float, time taken for detection in milliseconds.
                embed_ms: float, time taken for embedding in milliseconds.
        """
        pass

    @abstractmethod
    def embed_all(self, image: np.ndarray):
        """
        Detects EVERY face in the image (no cap) and returns one embedding
        per face. Used for /search, so a group photo doesn't get silently
        reduced to a single face before recognition runs -- every person
        present gets their own embedding searched independently.

        Args:
            image (np.ndarray): The input image (BGR format).

        Returns:
            tuple: (faces, detect_ms, embed_ms)
                faces: list of dicts, one per detected face:
                    {"embedding": np.ndarray, "bbox": [x1,y1,x2,y2], "confidence": float}
                detect_ms: float, total detection time in milliseconds.
                embed_ms: float, total embedding time in milliseconds (all faces).
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

    def embed_all(self, image: np.ndarray):
        start_det = time.perf_counter()
        # max_num=0 -> no cap, return every face the detector finds
        bboxes, kpss = self.det_model.detect(image, max_num=0, metric='default')
        detect_ms = (time.perf_counter() - start_det) * 1000.0

        faces_out = []

        class FakeFace:
            def __init__(self, bbox, kps, det_score):
                self.bbox = bbox
                self.kps = kps
                self.det_score = det_score

        start_embed = time.perf_counter()
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None

            face = FakeFace(bbox, kps, det_score)
            self.rec_model.get(image, face)
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            faces_out.append({
                "embedding": embedding,
                "bbox": [float(v) for v in bbox],
                "confidence": float(det_score),
            })
        embed_ms = (time.perf_counter() - start_embed) * 1000.0

        return faces_out, detect_ms, embed_ms


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

    def embed_all(self, image: np.ndarray):
        start_det = time.perf_counter()
        faces = self.detector.detect(image)  # max_num=0 by default -> every face
        detect_ms = (time.perf_counter() - start_det) * 1000.0

        faces_out = []
        start_embed = time.perf_counter()
        for face in faces:
            embedding = self.recognizer.get_normalized_embedding(image, face.landmarks)
            embedding = embedding / np.linalg.norm(embedding)
            bbox = face.bbox.tolist() if hasattr(face.bbox, "tolist") else list(face.bbox)
            faces_out.append({
                "embedding": embedding,
                "bbox": [float(v) for v in bbox],
                "confidence": float(face.confidence),
            })
        embed_ms = (time.perf_counter() - start_embed) * 1000.0

        return faces_out, detect_ms, embed_ms


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
            transforms.Resize(256),        # resize shorter side to 256, keeps aspect ratio
            transforms.CenterCrop(224),    # then crop the standard 224x224 square
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # A second, no-crop transform used only as one of several TTA views
        # during search (see embed_multi_view). Squashing distorts a full
        # building's proportions, which is why it is NOT used for the main
        # gallery embed() -- but for an already-partial/cropped query photo,
        # keeping the ENTIRE remaining content (instead of center-cropping
        # away more of it) can occasionally recover a match the aspect-
        # preserving crop would miss.
        self._transform_full_frame = transforms.Compose([
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

    def _embed_pil(self, pil_image, transform):
        """Runs one PIL image through DINOv2 with the given transform, returns a flat L2-normalized numpy embedding."""
        tensor = transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().astype(np.float32)

    def embed(self, image_bgr: np.ndarray):
        """
        Masks out people, then embeds the remaining background with DINOv2.
        Used for /insert and for building the gallery -- single, consistent
        aspect-preserving view per photo.

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
        embedding = self._embed_pil(pil_image, self._transform)
        embed_ms = (time.perf_counter() - start_embed) * 1000.0

        return embedding, mask_ms, embed_ms

    def embed_multi_view(self, image_bgr: np.ndarray):
        """
        Used for SEARCH only. Generates several embeddings of the SAME
        photo under different crops/framings, so a partially-cropped or
        tightly-framed query (e.g. half of a landmark cut off) still has
        a chance to match a gallery photo that shows the full building --
        rather than relying on a single fixed crop that might discard the
        one distinctive part of the building still visible.

        Views generated:
            "standard"   - aspect-preserving resize + center crop (same as embed())
            "full_frame" - resize the whole masked image to 224x224 without
                            cropping, so no additional content is thrown away
            "zoom"       - center-crop to 85% first, then aspect-preserving
                            resize+crop, i.e. a slightly tighter framing

        Returns:
            tuple: (views, mask_ms, embed_ms)
                views: list of {"variant": str, "embedding": np.ndarray}
                mask_ms: float, time spent on person segmentation (shared across views).
                embed_ms: float, TOTAL time spent embedding all views.
        """
        from PIL import Image

        start_mask = time.perf_counter()
        masked_bgr = self._mask_out_people(image_bgr)
        mask_ms = (time.perf_counter() - start_mask) * 1000.0

        rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        w, h = pil_image.size

        views = []
        start_embed = time.perf_counter()

        views.append({"variant": "standard", "embedding": self._embed_pil(pil_image, self._transform)})
        views.append({"variant": "full_frame", "embedding": self._embed_pil(pil_image, self._transform_full_frame)})

        # 85% center crop -> slightly zoomed-in framing
        crop_w, crop_h = int(w * 0.85), int(h * 0.85)
        left, top = (w - crop_w) // 2, (h - crop_h) // 2
        zoomed = pil_image.crop((left, top, left + crop_w, top + crop_h))
        views.append({"variant": "zoom", "embedding": self._embed_pil(zoomed, self._transform)})

        embed_ms = (time.perf_counter() - start_embed) * 1000.0

        return views, mask_ms, embed_ms


_scene_engine_singleton: "SceneEngine | None" = None


def get_scene_engine() -> SceneEngine:
    """Lazily builds and caches a single SceneEngine (mirrors get_engine() for face engines)."""
    global _scene_engine_singleton
    if _scene_engine_singleton is None:
        _scene_engine_singleton = SceneEngine()
    return _scene_engine_singleton