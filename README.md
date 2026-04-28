# QR Code Authenticity Detector 🔍

Detects whether a QR code is **original** (contains covert micro-dots) or a **photocopy/clone** (micro-dots destroyed by the copy process).

---

## Architecture

```
qr_detector/
├── backend/
│   ├── main.py              ← FastAPI app (3 models, /predict endpoint)
│   └── requirements.txt
├── frontend/
│   ├── app.py               ← Streamlit UI
│   └── requirements.txt
├── models/                  ← Drop your .pth weight files here
│   ├── best_microdot_cnn_model.pth    (MicroDotCNN_UltraSimple)
│   ├── best_microdot_cnn_model2.pth   (MicroDotCNN_FiveLayer)
│   └── best_mobilenet_v3_model.pth    (MobileNetV3)
└── README.md
```

---

## Models

| Model | Architecture | Description |
|---|---|---|
| `MicroDotCNN_UltraSimple` | 3-block CNN | Fast, lightweight — catches obvious micro-dot patterns |
| `MicroDotCNN_FiveLayer` | 5-block CNN | High capacity — detects subtle 5–10% intensity dots |
| `MobileNetV3` | Transfer learning | Generalises well across print/scan conditions |

All three use **majority voting** to produce the final ensemble verdict.

---

## How Micro-dots Work

- **Original QR codes** have exactly **150 covert micro-dots** embedded in white regions
- Dot intensity: **229–242** on 0–255 scale = **5–10% black** (sub-perceptual to human eye)
- **Photocopiers/scanners** normalise white regions, destroying sub-threshold marks
- The CNNs are trained to detect the presence/absence of this statistical signature

---

## Setup & Run

### 1. Place model weights

Copy your trained `.pth` files into the `models/` folder:
```
models/best_microdot_cnn_model.pth
models/best_microdot_cnn_model2.pth
models/best_mobilenet_v3_model.pth
```

If no weights are found, the API runs in **DEMO mode** (random predictions) so you can still test the UI.

### 2. Install dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
pip install -r requirements.txt
```

### 3. Start the FastAPI backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 4. Start the Streamlit frontend

```bash
cd frontend
streamlit run app.py --server.port 8501
```

Open: http://localhost:8501

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info + loaded models |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List all models and their load status |
| `POST` | `/predict` | Classify QR code with all 3 models (ensemble) |
| `POST` | `/predict/{model_name}` | Classify with a single named model |

### Example: cURL

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@my_qr_code.png"
```

### Example: Python

```python
import requests

with open("my_qr_code.png", "rb") as f:
    r = requests.post("http://localhost:8000/predict",
                      files={"file": ("qr.png", f, "image/png")})
print(r.json())
```

---

## Response Schema

```json
{
  "overall_verdict": "✅ Authentic QR Code",
  "overall_confidence": 91.5,
  "authentic_votes": 3,
  "cloned_votes": 0,
  "model_results": [
    {
      "model_name": "MicroDotCNN_UltraSimple",
      "prediction": "Authentic",
      "confidence": 94.2,
      "authentic_prob": 94.2,
      "cloned_prob": 5.8,
      "description": "...",
      "status": "loaded"
    }
  ],
  "microdot_analysis": {
    "total_pixels": 50176,
    "white_pixels": 28400,
    "microdot_pixels": 147,
    "microdot_ratio": 0.005176,
    "expected_min_dots": 150,
    "intensity_range": "229–242 (5–10% black)"
  },
  "filename": "qr_sample.png"
}
```

---

## Training Details

The dataset was generated from original QR code images:
- **Class 0 (Cloned/Fake):** Gaussian blur + slight fade to simulate photocopy degradation
- **Class 1 (Authentic):** 150 micro-dots injected at 229–242 intensity in white regions
- Augmentation: random crop, rotation ±15°, perspective, brightness/contrast jitter, Gaussian noise
- Curriculum learning: hard mode from epoch 0 with learning-rate decay to 0.0001
- Early stopping: patience = 4 epochs on validation accuracy
