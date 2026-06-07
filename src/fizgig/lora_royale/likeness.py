"""Face-likeness scoring for LoRA Royale.

Given a reference (training) image and a set of rendered epoch images, score
how closely each render's face matches the reference using InsightFace ArcFace
embeddings (the buffalo_l recognition model — already a Fizgig dependency via
face_utils). Higher cosine similarity = more alike.

CPU-only (ctx_id=-1) so it never fights the DiT for VRAM. The recognition
module is loaded separately from face_utils' detector, which only enables
detection + genderage — here we need 'recognition' to get `normed_embedding`.
"""

import os
from typing import List, Optional, Tuple

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_app = None
_available = None


def available() -> bool:
    """True if InsightFace + OpenCV are importable."""
    global _available
    if _available is None:
        try:
            import insightface  # noqa: F401
            import cv2  # noqa: F401
            _available = True
        except ImportError:
            _available = False
    return _available


def _ensure_app():
    global _app
    if _app is not None:
        return _app
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1)
    _app = app
    return _app


def _detect_faces(bgr):
    """Detect faces, retrying with a padded border if none are found.

    RetinaFace (buffalo_l) fails on faces that FILL the frame — a close-up
    headshot is too large for its anchor scales, so it returns nothing. Padding
    a border around the image shrinks the face's fraction of the frame, bringing
    it back into the detector's working range. The padding shifts coordinates but
    not the face content, so the alignment/embedding is unchanged — it only helps
    detection. We only pad when the unpadded pass fails (no regression for normal
    framing, and small faces aren't shrunk needlessly)."""
    import cv2
    app = _ensure_app()
    faces = app.get(bgr)
    if faces:
        return faces
    for pad in (0.25, 0.5):
        h, w = bgr.shape[:2]
        py, px = int(h * pad), int(w * pad)
        padded = cv2.copyMakeBorder(bgr, py, py, px, px, cv2.BORDER_REPLICATE)
        faces = app.get(padded)
        if faces:
            return faces
    return []


def _largest_embedding(pil_image) -> Optional[np.ndarray]:
    """ArcFace embedding (L2-normalized, 512-d) of the largest face, or None."""
    import cv2
    arr = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    faces = _detect_faces(bgr)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        return None
    return np.asarray(emb, dtype=np.float32)


def embedding_for_path(path: str) -> Optional[np.ndarray]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return _largest_embedding(im)
    except Exception:
        return None


def embedding_for_image(pil_image) -> Optional[np.ndarray]:
    try:
        return _largest_embedding(pil_image)
    except Exception:
        return None


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity of two normed embeddings (dot product). NaN if either
    is missing (no face detected)."""
    if a is None or b is None:
        return float("nan")
    return float(np.dot(a, b))


def score_renders(ref_path: str, renders: List[Tuple]) -> List[Tuple[object, float]]:
    """`renders`: list of (label, PIL.Image). Returns [(label, score)] with
    score in [-1, 1] (NaN where the reference or a render has no detectable
    face). Reference is embedded once."""
    ref = embedding_for_path(ref_path)
    out = []
    for label, img in renders:
        out.append((label, cosine(ref, embedding_for_image(img))))
    return out
