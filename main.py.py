import base64
import binascii
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
import insightface

app = FastAPI(title="Face Verification API")

# Load model once on startup
face_app = insightface.app.FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0 if False else -1, det_size=(640, 640))
# ctx_id=-1 for CPU
# if you later use GPU, change to ctx_id=0


# -----------------------------
# Request / Response Schemas
# -----------------------------
class VerifyEmployeeFaceRequest(BaseModel):
    employee_id: int
    frames: List[str] = Field(..., min_length=1)
    registered_embeddings: List[List[float]] = Field(..., min_length=1)
    threshold: float = 0.58
    min_matched_frames: int = 4


class VerifyEmployeeFaceResponse(BaseModel):
    success: bool
    message: str
    confidence: float
    matched_frames: int
    total_frames: int
    quality_score: float


# -----------------------------
# Utility Functions
# -----------------------------
def decode_base64_image(data_url: str) -> np.ndarray:
    """
    Accepts:
    data:image/jpeg;base64,...
    or raw base64 string
    Returns OpenCV BGR image
    """
    try:
        if "," in data_url:
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url

        image_bytes = base64.b64decode(encoded)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        return image_bgr
    except (binascii.Error, ValueError, OSError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid base64 image: {str(e)}")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a.astype(np.float32))
    b = l2_normalize(b.astype(np.float32))
    return float(np.dot(a, b))


def average_best_similarity(
    live_embedding: np.ndarray,
    registered_embeddings: List[np.ndarray]
) -> float:
    """
    Compare one live embedding against all saved embeddings
    and return the highest similarity score.
    """
    scores = [cosine_similarity(live_embedding, reg) for reg in registered_embeddings]
    return max(scores) if scores else 0.0


def estimate_quality(face) -> float:
    """
    Simple quality estimate based on detection score.
    You can improve later with blur / angle checks.
    """
    det_score = float(getattr(face, "det_score", 0.0) or 0.0)
    return round(det_score, 4)


def extract_single_face_embedding(frame_bgr: np.ndarray) -> tuple[Optional[np.ndarray], float, str]:
    """
    Returns:
    (embedding, quality_score, message)
    Rules:
    - exactly 1 face required
    """
    faces = face_app.get(frame_bgr)

    if not faces:
        return None, 0.0, "No face detected."

    if len(faces) > 1:
        return None, 0.0, "Multiple faces detected."

    face = faces[0]
    embedding = np.array(face.embedding, dtype=np.float32)
    quality_score = estimate_quality(face)

    return embedding, quality_score, "OK"


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Employee Verification
# -----------------------------
@app.post("/verify-employee-face", response_model=VerifyEmployeeFaceResponse)
def verify_employee_face(payload: VerifyEmployeeFaceRequest):
    if not payload.registered_embeddings:
        raise HTTPException(status_code=422, detail="No registered embeddings provided.")

    registered_embeddings = [
        np.array(embedding, dtype=np.float32)
        for embedding in payload.registered_embeddings
        if embedding
    ]

    if not registered_embeddings:
        raise HTTPException(status_code=422, detail="Registered embeddings are empty or invalid.")

    matched_frames = 0
    total_frames = 0
    all_scores: List[float] = []
    quality_scores: List[float] = []

    for frame_data in payload.frames:
        frame_bgr = decode_base64_image(frame_data)
        live_embedding, quality_score, message = extract_single_face_embedding(frame_bgr)

        total_frames += 1

        if live_embedding is None:
            continue

        score = average_best_similarity(live_embedding, registered_embeddings)
        all_scores.append(score)
        quality_scores.append(quality_score)

        if score >= payload.threshold:
            matched_frames += 1

    avg_confidence = round(float(np.mean(all_scores)) if all_scores else 0.0, 4)
    avg_quality = round(float(np.mean(quality_scores)) if quality_scores else 0.0, 4)

    if matched_frames >= payload.min_matched_frames:
        return VerifyEmployeeFaceResponse(
            success=True,
            message="Face verified successfully.",
            confidence=avg_confidence,
            matched_frames=matched_frames,
            total_frames=total_frames,
            quality_score=avg_quality,
        )

    return VerifyEmployeeFaceResponse(
        success=False,
        message="Face verification failed. The live face does not match the logged-in employee.",
        confidence=avg_confidence,
        matched_frames=matched_frames,
        total_frames=total_frames,
        quality_score=avg_quality,
    )