from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from insightface.app import FaceAnalysis

app = FastAPI()

# Load InsightFace model
face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=-1)  # CPU


# -----------------------------
# Request models
# -----------------------------
class RegisterRequest(BaseModel):
    employee_id: int
    employee_no: str | None = None
    frames: list[str]


class VerifyEmployeeFaceRequest(BaseModel):
    employee_id: int
    frames: list[str]
    registered_embeddings: list[list[float]]
    threshold: float = 0.58
    min_matched_frames: int = 4


# -----------------------------
# Helper: decode base64 image
# -----------------------------
def decode_image(base64_str: str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    img_bytes = base64.b64decode(base64_str)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


# -----------------------------
# Helper: normalize vector
# -----------------------------
def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


# -----------------------------
# Helper: cosine similarity
# -----------------------------
def cosine_similarity(vec1, vec2):
    v1 = normalize_vector(vec1)
    v2 = normalize_vector(vec2)
    return float(np.dot(v1, v2))


# -----------------------------
# Helper: get best similarity
# -----------------------------
def best_similarity(live_embedding, registered_embeddings):
    if not registered_embeddings:
        return 0.0

    scores = [
        cosine_similarity(live_embedding, reg_embedding)
        for reg_embedding in registered_embeddings
    ]
    return max(scores)


# -----------------------------
# REGISTER FACE ENDPOINT
# -----------------------------
@app.post("/register-face")
def register_face(data: RegisterRequest):
    accepted_samples = []

    for frame in data.frames:
        try:
            img = decode_image(frame)
            faces = face_app.get(img)

            # No face
            if len(faces) == 0:
                continue

            # Multiple faces
            if len(faces) > 1:
                continue

            face = faces[0]

            # Basic quality filtering
            if float(face.det_score) < 0.6:
                continue

            embedding = face.embedding.tolist()

            # Convert image back to base64 for Laravel saving
            _, buffer = cv2.imencode(".jpg", img)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            accepted_samples.append({
                "image_base64": img_base64,
                "embedding": embedding,
                "det_score": float(face.det_score),
                "quality_score": float(face.det_score),
                "yaw": float(face.pose[1]) if face.pose is not None else 0.0,
                "pitch": float(face.pose[0]) if face.pose is not None else 0.0,
                "roll": float(face.pose[2]) if face.pose is not None else 0.0,
                "landmarks": face.kps.tolist() if face.kps is not None else [],
                "model_name": "insightface",
                "model_version": "buffalo_s"
            })

        except Exception as e:
            print("Error processing frame:", e)
            continue

    if len(accepted_samples) < 3:
        return {
            "success": False,
            "message": "Not enough valid face samples. Please try again."
        }

    return {
        "success": True,
        "accepted_samples": accepted_samples
    }


# -----------------------------
# VERIFY EMPLOYEE FACE ENDPOINT
# -----------------------------
@app.post("/verify-employee-face")
def verify_employee_face(data: VerifyEmployeeFaceRequest):
    try:
        if not data.frames:
            return {
                "success": False,
                "message": "No frames provided.",
                "confidence": 0.0,
                "matched_frames": 0,
                "quality_score": 0.0
            }

        if not data.registered_embeddings:
            return {
                "success": False,
                "message": "No registered embeddings provided.",
                "confidence": 0.0,
                "matched_frames": 0,
                "quality_score": 0.0
            }

        matched_frames = 0
        confidence_scores = []
        quality_scores = []

        registered_embeddings = [
            np.array(embedding, dtype=np.float32)
            for embedding in data.registered_embeddings
            if embedding
        ]

        for frame in data.frames:
            try:
                img = decode_image(frame)
                faces = face_app.get(img)

                # Reject frame if no face
                if len(faces) == 0:
                    continue

                # Reject frame if multiple faces
                if len(faces) > 1:
                    continue

                face = faces[0]

                # Detection quality check
                det_score = float(face.det_score)
                if det_score < 0.6:
                    continue

                live_embedding = np.array(face.embedding, dtype=np.float32)
                score = best_similarity(live_embedding, registered_embeddings)

                confidence_scores.append(score)
                quality_scores.append(det_score)

                if score >= data.threshold:
                    matched_frames += 1

            except Exception as e:
                print("Error verifying frame:", e)
                continue

        avg_confidence = round(float(np.mean(confidence_scores)), 4) if confidence_scores else 0.0
        avg_quality = round(float(np.mean(quality_scores)), 4) if quality_scores else 0.0

        if matched_frames >= data.min_matched_frames:
            return {
                "success": True,
                "message": "Face verified successfully.",
                "confidence": avg_confidence,
                "matched_frames": matched_frames,
                "quality_score": avg_quality
            }

        return {
            "success": False,
            "message": "Face verification failed. The live face does not match the logged-in employee.",
            "confidence": avg_confidence,
            "matched_frames": matched_frames,
            "quality_score": avg_quality
        }

    except Exception as e:
        print("Verification error:", e)
        return {
            "success": False,
            "message": f"Verification error: {str(e)}",
            "confidence": 0.0,
            "matched_frames": 0,
            "quality_score": 0.0
        }


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok"
    }
