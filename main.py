import os
from io import BytesIO
from typing import List

import cv2
import insightface
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel


app = FastAPI()

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://rastaconnect-frontend.onrender.com",
]


def get_allowed_origins() -> List[str]:
    raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw_origins:
        origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
        if origins:
            return origins
    return DEFAULT_ALLOWED_ORIGINS


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_app = None
model_init_error = None


@app.on_event("startup")
def startup_event() -> None:
    global face_app
    global model_init_error
    try:
        face_app = insightface.app.FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        model_init_error = None
    except Exception as error:
        face_app = None
        model_init_error = str(error)


class VerifyFaceRequest(BaseModel):
    embedding1: List[float]
    embedding2: List[float]


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/extract-embedding")
async def extract_embedding(image: UploadFile = File(...)) -> List[float]:
    if face_app is None:
        detail = "Face model is not loaded yet."
        if model_init_error:
            detail = f"{detail} Startup error: {model_init_error}"
        raise HTTPException(status_code=503, detail=detail)

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        pil_image = Image.open(BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image format")

    width, height = pil_image.size
    if width < 80 or height < 80:
        raise HTTPException(status_code=400, detail="Image too small for face detection")

    rgb_array = np.array(pil_image)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    try:
        faces = face_app.get(bgr_array)
    except Exception:
        raise HTTPException(status_code=500, detail="Face analysis failed")

    if not faces:
        raise HTTPException(status_code=400, detail="No face detected")

    embedding = np.array(faces[0].embedding, dtype=np.float32).flatten()
    if embedding.shape[0] != 512:
        raise HTTPException(status_code=400, detail="Invalid embedding size")

    return embedding.tolist()


@app.post("/verify-face")
async def verify_face(payload: VerifyFaceRequest) -> dict:
    emb1 = np.array(payload.embedding1, dtype=np.float32)
    emb2 = np.array(payload.embedding2, dtype=np.float32)

    if emb1.size == 0 or emb2.size == 0:
        raise HTTPException(status_code=400, detail="Embeddings must not be empty")

    if emb1.shape != emb2.shape:
        raise HTTPException(status_code=400, detail="Embeddings must have same dimensions")

    denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    if denom == 0.0:
        raise HTTPException(status_code=400, detail="Invalid embeddings")

    score = float(np.dot(emb1, emb2) / denom)
    return {"match": score > 0.65, "score": score}
