import io
import json
import os
import urllib.error
import urllib.request
import warnings
import jwt
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from PIL import Image
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Auth (shared JWT secret with the Node/Express server AND the clinical-api) ─
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is not set. Refusing to start.")


def require_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ─── Load CNN Model (ONNX — no torch/torchvision dependency at runtime) ──────
# alzheimer_cnn_model.onnx is ~94MB — too large for GitHub's web-upload UI
# (25MB limit) and excluded from git via .gitignore. Host it as a GitHub
# Release asset (or any direct-download URL) and set CNN_MODEL_URL so the
# server fetches it on first boot; otherwise just place the file manually.
_CNN_MODEL_PATH = os.environ.get("CNN_MODEL_PATH") or os.path.join(BASE_DIR, "alzheimer_cnn_model.onnx")
_CNN_MODEL_URL = os.environ.get("CNN_MODEL_URL")
# Opt-in: defer CNN loading to the first /predict-mri call instead of failing
# the whole app at startup if the model isn't resolved yet. Off by default —
# Render/HF deployments should keep failing fast on misconfiguration; Vercel's
# entrypoint sets this to true since its cold starts can't afford a hard fail.
_LAZY_CNN_LOAD = os.environ.get("LAZY_CNN_LOAD", "false").lower() == "true"

with open(os.path.join(BASE_DIR, "alzheimer_cnn_onnx_metadata.json")) as f:
    _cnn_metadata = json.load(f)

_cnn_cache = {}


def _load_cnn():
    if not os.path.exists(_CNN_MODEL_PATH):
        if not _CNN_MODEL_URL:
            raise RuntimeError(
                "alzheimer_cnn_model.onnx not found and CNN_MODEL_URL is not set. "
                "Either place the model file at mri-api/alzheimer_cnn_model.onnx, "
                "or set CNN_MODEL_URL to a direct-download link (e.g. a GitHub Release asset)."
            )
        print(f"alzheimer_cnn_model.onnx not found locally — downloading from: {_CNN_MODEL_URL}")
        try:
            req = urllib.request.Request(_CNN_MODEL_URL, headers={"User-Agent": "mindbridge-deploy/1.0"})
            with urllib.request.urlopen(req) as resp, open(_CNN_MODEL_PATH, "wb") as out:
                out.write(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Failed to download alzheimer_cnn_model.onnx from CNN_MODEL_URL "
                f"(HTTP {e.code} {e.reason}) — URL was: {_CNN_MODEL_URL}. "
                "Check that this is the asset's direct download link (not the release "
                "page), the tag/filename are exact, and the repo/asset is public."
            ) from e
        downloaded_size = os.path.getsize(_CNN_MODEL_PATH)
        print(f"Download complete ({downloaded_size} bytes).")

    session = ort.InferenceSession(_CNN_MODEL_PATH, providers=["CPUExecutionProvider"])
    img_size = _cnn_metadata["img_size"]
    mean = np.array(_cnn_metadata["normalize_mean"], dtype=np.float32)
    std = np.array(_cnn_metadata["normalize_std"], dtype=np.float32)

    def preprocess(img: Image.Image) -> np.ndarray:
        resized = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(resized).astype(np.float32) / 255.0  # HWC, 0-1
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # NCHW
        return arr

    return {
        "session": session,
        "preprocess": preprocess,
        "classes": _cnn_metadata["classes"],
        "img_size": img_size,
        "val_acc": _cnn_metadata.get("best_val_acc"),
    }


def _get_cnn(force: bool = True):
    """Returns the cached CNN bundle, loading it on first call. If `force` is
    False and lazy-loading is enabled, returns None instead of loading when
    it hasn't been loaded yet (used by /health to avoid triggering a load)."""
    if "data" not in _cnn_cache:
        if _LAZY_CNN_LOAD and not force:
            return None
        _cnn_cache["data"] = _load_cnn()
    return _cnn_cache["data"]


if not _LAZY_CNN_LOAD:
    _get_cnn()

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="MindBridge MRI Classification API", version="1.0.0")

_cors_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:5000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_MRI_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


# ─── MRI Image Classification Endpoint ───────────────────────────────────────
@app.post("/predict-mri")
async def predict_mri(file: UploadFile = File(...), _user: dict = Depends(require_auth)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        if len(contents) > MAX_MRI_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB).")
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Send a valid JPEG/PNG.")

    try:
        cnn = _get_cnn()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"MRI model unavailable: {e}")

    input_array = cnn["preprocess"](img)
    logits = cnn["session"].run(None, {"input": input_array})[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    pred_idx = int(np.argmax(probs))
    predicted_class = cnn["classes"][pred_idx]
    confidence = float(probs[pred_idx])

    class_probabilities = {
        cls: round(float(probs[i]), 4)
        for i, cls in enumerate(cnn["classes"])
    }

    if predicted_class == "NonDemented":
        severity = "None"
    elif predicted_class == "VeryMildDemented":
        severity = "Very Mild"
    elif predicted_class == "MildDemented":
        severity = "Mild"
    else:
        severity = "Moderate"

    return {
        "predicted_class": predicted_class,
        "severity": severity,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probabilities,
        "model": "ResNet50 CNN",
    }


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    # Doesn't force-load the CNN model when lazy-loading is enabled — a health
    # check shouldn't trigger a 94MB download + model build on its own.
    cnn = _get_cnn(force=not _LAZY_CNN_LOAD)
    cnn_info = (
        {
            "type": "ResNet50 CNN (ONNX)",
            "classes": cnn["classes"],
            "img_size": cnn["img_size"],
            "val_accuracy": cnn["val_acc"],
        }
        if cnn
        else {"type": "ResNet50 CNN (ONNX)", "status": "not loaded yet (lazy mode)"}
    )
    return {"status": "ok", "model": cnn_info}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8001)),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )
