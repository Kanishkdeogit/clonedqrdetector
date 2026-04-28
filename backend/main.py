"""
QR Code Authenticity Detection API
Detects whether a QR code is original (with micro-dots) or a photocopy/clone.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import io
import os
import cv2
import logging
from typing import Optional
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="QR Code Authenticity Detector API",
    description="Detects original vs. photocopied QR codes using micro-dot steganography analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ─── Model Definitions ────────────────────────────────────────────────────────

class MicroDotCNN_UltraSimple(nn.Module):
    """3-block lightweight CNN for micro-dot detection."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(64 * 28 * 28, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MicroDotCNN_FiveLayer(nn.Module):
    """5-block deep CNN for fine-grained micro-dot analysis."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   nn.BatchNorm2d(16),  nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 7 * 7, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_mobilenet_v3():
    """MobileNetV3-Small fine-tuned for 2-class QR detection."""
    m = models.mobilenet_v3_small(weights=None)
    num_ftrs = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(num_ftrs, 2)
    return m


# ─── Model Registry ───────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_CONFIGS = {
    "MicroDotCNN_UltraSimple": {
        "builder": MicroDotCNN_UltraSimple,
        "weight_file": "best_microdot_cnn_model.pth",
        "description": "Lightweight 3-block CNN — fast inference, catches obvious micro-dot patterns.",
    },
    "MicroDotCNN_FiveLayer": {
        "builder": MicroDotCNN_FiveLayer,
        "weight_file": "best_microdot_cnn_model2.pth",
        "description": "Deep 5-block CNN — higher capacity for subtle 5–10% intensity micro-dots.",
    },
    "MobileNetV3": {
        "builder": build_mobilenet_v3,
        "weight_file": "best_mobilenet_v3_model.pth",
        "description": "Transfer-learned MobileNetV3-Small — generalises well across print/scan conditions.",
    },
}

loaded_models: dict = {}


def load_models():
    """Load all available models from the models/ directory."""
    for name, cfg in MODEL_CONFIGS.items():
        weight_path = os.path.join(MODELS_DIR, cfg["weight_file"])
        if os.path.exists(weight_path):
            try:
                model = cfg["builder"]()
                state = torch.load(weight_path, map_location=DEVICE)
                model.load_state_dict(state)
                model.to(DEVICE)
                model.eval()
                loaded_models[name] = model
                logger.info(f"✓ Loaded model: {name}")
            except Exception as e:
                logger.warning(f"✗ Could not load {name}: {e}")
        else:
            logger.info(f"  Skipping {name} — weight file not found at {weight_path}")


# ─── Preprocessing ────────────────────────────────────────────────────────────

IMAGE_SIZE = 224

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode bytes → PIL RGB → normalised tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return val_transform(image).unsqueeze(0).to(DEVICE)


# ─── Micro-dot Analysis ───────────────────────────────────────────────────────

def analyze_microdots(image_bytes: bytes) -> dict:
    """
    Low-level pixel analysis to count pixels in the 5–10% black intensity
    range (229–242 on a 0-255 scale) — the covert micro-dot band.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": "Could not decode image for analysis"}

    white_mask = img > 245
    dot_mask   = (img >= 229) & (img <= 242)          # 5–10 % black intensity
    white_px   = int(white_mask.sum())
    dot_px     = int(dot_mask.sum())
    dot_ratio  = round(dot_px / white_px, 6) if white_px > 0 else 0.0

    return {
        "total_pixels":       int(img.size),
        "white_pixels":       white_px,
        "microdot_pixels":    dot_px,
        "microdot_ratio":     dot_ratio,
        "expected_min_dots":  150,
        "intensity_range":    "229–242 (5–10% black)",
    }


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    load_models()
    if not loaded_models:
        logger.warning("No pre-trained weights found. API will run in DEMO mode (random predictions).")


# ─── Response Schema ──────────────────────────────────────────────────────────

class ModelResult(BaseModel):
    model_name: str
    prediction: str          # "Authentic" | "Cloned/Photocopy"
    confidence: float
    authentic_prob: float
    cloned_prob: float
    description: str
    status: str              # "loaded" | "demo"


class PredictionResponse(BaseModel):
    overall_verdict: str
    overall_confidence: float
    authentic_votes: int
    cloned_votes: int
    model_results: list[ModelResult]
    microdot_analysis: dict
    filename: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

CLASSES = ["authentic", "fake"]   # ImageFolder alphabetical order

def run_inference(model: nn.Module, tensor: torch.Tensor) -> tuple[str, float, float, float]:
    with torch.no_grad():
        logits  = model(tensor)
        probs   = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())

    authentic_prob = float(probs[0])
    cloned_prob    = float(probs[1])
    prediction     = "Authentic" if pred_idx == 0 else "Cloned/Photocopy"
    confidence     = float(probs[pred_idx])
    return prediction, confidence, authentic_prob, cloned_prob


def demo_inference() -> tuple[str, float, float, float]:
    """Fallback when weights are unavailable."""
    import random
    authentic_prob = random.uniform(0.3, 0.7)
    cloned_prob    = 1 - authentic_prob
    pred_idx       = 0 if authentic_prob > 0.5 else 1
    prediction     = "Authentic" if pred_idx == 0 else "Cloned/Photocopy"
    confidence     = max(authentic_prob, cloned_prob)
    return prediction, confidence, authentic_prob, cloned_prob


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "QR Code Authenticity Detector",
        "status": "running",
        "models_loaded": list(loaded_models.keys()),
        "device": str(DEVICE),
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "models_available": list(loaded_models.keys())}


@app.get("/models", tags=["Models"])
def list_models():
    return {
        name: {
            "description": cfg["description"],
            "loaded": name in loaded_models,
            "weight_file": cfg["weight_file"],
        }
        for name, cfg in MODEL_CONFIGS.items()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a QR code image.
    Returns predictions from all 3 models plus a low-level micro-dot pixel analysis.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file received.")

    try:
        tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    microdot_info = analyze_microdots(image_bytes)

    results: list[ModelResult] = []
    demo_mode = len(loaded_models) == 0

    for name, cfg in MODEL_CONFIGS.items():
        try:
            if name in loaded_models:
                pred, conf, auth_p, clone_p = run_inference(loaded_models[name], tensor)
                status = "loaded"
            else:
                pred, conf, auth_p, clone_p = demo_inference()
                status = "demo"

            results.append(ModelResult(
                model_name     = name,
                prediction     = pred,
                confidence     = round(conf * 100, 2),
                authentic_prob = round(auth_p * 100, 2),
                cloned_prob    = round(clone_p * 100, 2),
                description    = cfg["description"],
                status         = status,
            ))
        except Exception as e:
            logger.error(f"Inference error for {name}: {e}")
            results.append(ModelResult(
                model_name     = name,
                prediction     = "Error",
                confidence     = 0.0,
                authentic_prob = 0.0,
                cloned_prob    = 0.0,
                description    = cfg["description"],
                status         = "error",
            ))

    # ── Ensemble majority vote ───────────────────────────────────────────────
    valid = [r for r in results if r.prediction not in ("Error",)]
    auth_votes  = sum(1 for r in valid if r.prediction == "Authentic")
    clone_votes = sum(1 for r in valid if r.prediction == "Cloned/Photocopy")

    if auth_votes > clone_votes:
        verdict     = "✅ Authentic QR Code"
        avg_conf    = round(np.mean([r.authentic_prob for r in valid]), 2)
    elif clone_votes > auth_votes:
        verdict     = "⚠️ Cloned / Photocopied QR Code"
        avg_conf    = round(np.mean([r.cloned_prob for r in valid]), 2)
    else:
        verdict     = "🔍 Inconclusive — Models Disagree"
        avg_conf    = round(np.mean([r.confidence for r in valid]), 2)

    return PredictionResponse(
        overall_verdict    = verdict,
        overall_confidence = avg_conf,
        authentic_votes    = auth_votes,
        cloned_votes       = clone_votes,
        model_results      = results,
        microdot_analysis  = microdot_info,
        filename           = file.filename or "uploaded_image",
    )


@app.post("/predict/{model_name}", tags=["Prediction"])
async def predict_single(model_name: str, file: UploadFile = File(...)):
    """Run inference with a single named model."""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    image_bytes = await file.read()
    try:
        tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    if model_name in loaded_models:
        pred, conf, auth_p, clone_p = run_inference(loaded_models[model_name], tensor)
        status = "loaded"
    else:
        pred, conf, auth_p, clone_p = demo_inference()
        status = "demo"

    return {
        "model_name":     model_name,
        "prediction":     pred,
        "confidence":     round(conf * 100, 2),
        "authentic_prob": round(auth_p * 100, 2),
        "cloned_prob":    round(clone_p * 100, 2),
        "status":         status,
    }
