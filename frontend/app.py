"""
QR Code Authenticity Detector — Streamlit Frontend
"""

import streamlit as st
import requests
import json
import time
import base64
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="QR Authenticity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Global */
  [data-testid="stAppViewContainer"] { background: #0a0e1a; }
  [data-testid="stSidebar"]          { background: #0f1526; border-right: 1px solid #1e2d4a; }
  .block-container { padding-top: 1.5rem; }

  /* Cards */
  .glass-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(12px);
  }
  .verdict-authentic {
    background: linear-gradient(135deg, rgba(0,200,120,0.15), rgba(0,200,120,0.05));
    border: 1px solid rgba(0,200,120,0.4);
    border-radius: 16px; padding: 1.5rem; text-align: center; margin: 1rem 0;
  }
  .verdict-cloned {
    background: linear-gradient(135deg, rgba(255,80,80,0.15), rgba(255,80,80,0.05));
    border: 1px solid rgba(255,80,80,0.4);
    border-radius: 16px; padding: 1.5rem; text-align: center; margin: 1rem 0;
  }
  .verdict-inconclusive {
    background: linear-gradient(135deg, rgba(255,200,0,0.15), rgba(255,200,0,0.05));
    border: 1px solid rgba(255,200,0,0.4);
    border-radius: 16px; padding: 1.5rem; text-align: center; margin: 1rem 0;
  }
  .model-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
  }
  .metric-box {
    background: rgba(30,45,74,0.6);
    border-radius: 10px; padding: 0.8rem; text-align: center;
  }
  .tag-loaded   { background:#0d5c3a; color:#00e694; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
  .tag-demo     { background:#5c4a00; color:#ffd700; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
  .tag-error    { background:#5c0d0d; color:#ff6b6b; border-radius:6px; padding:2px 8px; font-size:0.75rem; }
  h1,h2,h3 { color: #e8eaf6; }
  p, li      { color: #9ca3af; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {}


def get_model_list() -> dict:
    try:
        r = requests.get(f"{API_URL}/models", timeout=3)
        return r.json()
    except Exception:
        return {}


def predict_image(image_bytes: bytes, filename: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/predict",
            files={"file": (filename, image_bytes, "image/png")},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API Error {r.status_code}: {r.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def confidence_color(confidence: float) -> str:
    if confidence >= 80:
        return "#00e694"
    elif confidence >= 60:
        return "#ffd700"
    else:
        return "#ff6b6b"


def make_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#9ca3af", "size": 13}},
        number={"suffix": "%", "font": {"color": "#e8eaf6", "size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 50],   "color": "rgba(255,255,255,0.03)"},
                {"range": [50, 100], "color": "rgba(255,255,255,0.06)"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=20, l=20, r=20),
        height=180,
    )
    return fig


def make_bar_chart(results: list[dict]) -> go.Figure:
    names  = [r["model_name"] for r in results]
    auth   = [r["authentic_prob"] for r in results]
    cloned = [r["cloned_prob"]    for r in results]

    fig = go.Figure(data=[
        go.Bar(name="Authentic",       x=names, y=auth,   marker_color="#00e694",
               text=[f"{v:.1f}%" for v in auth],   textposition="outside"),
        go.Bar(name="Cloned/Photocopy",x=names, y=cloned, marker_color="#ff6b6b",
               text=[f"{v:.1f}%" for v in cloned], textposition="outside"),
    ])
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#9ca3af"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
        yaxis={"gridcolor": "rgba(255,255,255,0.05)", "range": [0, 115]},
        margin=dict(t=20, b=20, l=20, r=20),
        height=260,
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 QR Auth Detector")
    st.markdown("---")

    # Backend status
    health = api_health()
    if health:
        st.success("🟢 Backend Online")
        loaded = health.get("models_available", [])
        if loaded:
            st.markdown(f"**Models loaded:** {len(loaded)}/3")
            for m in loaded:
                st.markdown(f"  ✅ `{m}`")
        else:
            st.warning("⚠️ Running in **DEMO mode** — no .pth weights found.")
    else:
        st.error("🔴 Backend Offline")
        st.markdown("""
**Start the API:**
```bash
cd backend
uvicorn main:app --reload
```
""")

    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
Original QR codes embed **150 covert micro-dots** at **5–10% black intensity** (229–242 on 0–255 scale) 
in white regions. Photocopiers and scanners destroy these sub-threshold marks.

**3 Models:**
- `MicroDotCNN_UltraSimple` — 3-block CNN
- `MicroDotCNN_FiveLayer` — 5-block CNN
- `MobileNetV3` — Transfer learning
""")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_raw = st.checkbox("Show raw API response", value=False)
    show_analysis = st.checkbox("Show pixel-level analysis", value=True)

    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem;color:#4a5568;'>v1.0 · Medishare QR Security</p>", unsafe_allow_html=True)


# ─── Main Layout ──────────────────────────────────────────────────────────────

st.markdown("# 🔍 QR Code Authenticity Detector")
st.markdown(
    "<p style='color:#6b7280;margin-top:-0.5rem;'>Upload a QR code image — all 3 models will classify it as <b style='color:#00e694'>Authentic</b> or <b style='color:#ff6b6b'>Cloned/Photocopy</b></p>",
    unsafe_allow_html=True,
)

# Upload zone
col_upload, col_info = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop or select a QR code image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help="Supported: PNG, JPG, JPEG, BMP, TIFF, WEBP",
        label_visibility="visible",
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"📎 {uploaded_file.name}  ({img.size[0]}×{img.size[1]} px)", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📡 What the models detect")
    st.markdown("""
**Authentic QR Code:**
- Contains ~150 faint micro-dots embedded in white zones
- Dots intensity: 229–242 (barely visible at 5–10% black)
- Destroyed by scanning/photocopying

**Cloned / Photocopied:**
- Missing or corrupted micro-dot pattern
- Only machine-detectable by trained CNNs
- Uniform white regions — no covert marks
""")
    if uploaded_file:
        st.markdown(f"**File:** `{uploaded_file.name}`")
        st.markdown(f"**Size:** `{len(uploaded_file.getvalue()) / 1024:.1f} KB`")
    st.markdown('</div>', unsafe_allow_html=True)

# Analyse button
if uploaded_file:
    st.markdown("")
    btn_col, _ = st.columns([1, 3])
    with btn_col:
        analyse = st.button("🔬 Analyse QR Code", use_container_width=True, type="primary")

    if analyse:
        with st.spinner("Running inference across all 3 models…"):
            image_bytes = uploaded_file.getvalue()
            t0 = time.time()
            response = predict_image(image_bytes, uploaded_file.name)
            elapsed = time.time() - t0

        if response:
            st.markdown(f"<p style='color:#4a5568;font-size:0.8rem;'>⏱ Inference completed in {elapsed:.2f}s</p>", unsafe_allow_html=True)

            # ── Verdict Banner ─────────────────────────────────────────────
            verdict = response["overall_verdict"]
            conf    = response["overall_confidence"]

            if "Authentic" in verdict:
                css_cls = "verdict-authentic"
                icon    = "✅"
            elif "Cloned" in verdict:
                css_cls = "verdict-cloned"
                icon    = "⚠️"
            else:
                css_cls = "verdict-inconclusive"
                icon    = "🔍"

            st.markdown(f"""
<div class="{css_cls}">
  <h2 style="margin:0;font-size:1.8rem;">{verdict}</h2>
  <p style="color:#9ca3af;margin:0.3rem 0 0;">
    Ensemble confidence: <b style="color:#e8eaf6;">{conf:.1f}%</b> &nbsp;·&nbsp;
    Authentic votes: <b style="color:#00e694;">{response['authentic_votes']}/3</b> &nbsp;·&nbsp;
    Cloned votes: <b style="color:#ff6b6b;">{response['cloned_votes']}/3</b>
  </p>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Per-model results ──────────────────────────────────────────
            st.markdown("### 🤖 Per-Model Results")

            tab1, tab2 = st.tabs(["  📊 Visual  ", "  🗂 Details  "])

            with tab1:
                # Gauge row
                g_cols = st.columns(3)
                for i, r in enumerate(response["model_results"]):
                    with g_cols[i]:
                        pred_is_auth = r["prediction"] == "Authentic"
                        gauge_val    = r["authentic_prob"] if pred_is_auth else r["cloned_prob"]
                        gauge_color  = "#00e694" if pred_is_auth else "#ff6b6b"
                        label        = f"{r['model_name']}<br>{'✅ Authentic' if pred_is_auth else '⚠️ Cloned'}"
                        fig = make_gauge(gauge_val, r["model_name"], gauge_color)
                        st.plotly_chart(fig, use_container_width=True, key=f"gauge_{i}")
                        pred_label = "✅ Authentic" if pred_is_auth else "⚠️ Cloned"
                        st.markdown(f"<p style='text-align:center;margin-top:-1rem;font-size:0.9rem;'>{pred_label} · {r['confidence']:.1f}% conf.</p>", unsafe_allow_html=True)

                # Bar chart
                st.markdown("#### Model Probability Comparison")
                st.plotly_chart(make_bar_chart(response["model_results"]), use_container_width=True, key="bar_chart")

            with tab2:
                for r in response["model_results"]:
                    tag_html = f"<span class='tag-{r['status']}'>{r['status'].upper()}</span>"
                    pred_color = "#00e694" if r["prediction"] == "Authentic" else "#ff6b6b"
                    st.markdown(f"""
<div class="model-card">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span style="font-weight:600;color:#e8eaf6;">{r['model_name']}</span>
    {tag_html}
  </div>
  <p style="font-size:0.8rem;color:#6b7280;margin:0.3rem 0;">{r['description']}</p>
  <div style="display:flex;gap:1rem;margin-top:0.6rem;">
    <div class="metric-box" style="flex:1;">
      <p style="margin:0;font-size:0.75rem;color:#6b7280;">Verdict</p>
      <p style="margin:0;font-weight:700;color:{pred_color};">{r['prediction']}</p>
    </div>
    <div class="metric-box" style="flex:1;">
      <p style="margin:0;font-size:0.75rem;color:#6b7280;">Confidence</p>
      <p style="margin:0;font-weight:700;color:#e8eaf6;">{r['confidence']:.2f}%</p>
    </div>
    <div class="metric-box" style="flex:1;">
      <p style="margin:0;font-size:0.75rem;color:#6b7280;">Auth Prob</p>
      <p style="margin:0;font-weight:700;color:#00e694;">{r['authentic_prob']:.2f}%</p>
    </div>
    <div class="metric-box" style="flex:1;">
      <p style="margin:0;font-size:0.75rem;color:#6b7280;">Clone Prob</p>
      <p style="margin:0;font-weight:700;color:#ff6b6b;">{r['cloned_prob']:.2f}%</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Micro-dot Pixel Analysis ───────────────────────────────────
            if show_analysis and "microdot_analysis" in response:
                st.markdown("---")
                st.markdown("### 🔬 Pixel-Level Micro-dot Analysis")
                ma = response["microdot_analysis"]

                if "error" not in ma:
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        st.markdown(f'<div class="metric-box"><p style="margin:0;font-size:0.75rem;color:#6b7280;">Total Pixels</p><p style="margin:0;font-weight:700;color:#e8eaf6;">{ma["total_pixels"]:,}</p></div>', unsafe_allow_html=True)
                    with mc2:
                        st.markdown(f'<div class="metric-box"><p style="margin:0;font-size:0.75rem;color:#6b7280;">White Pixels</p><p style="margin:0;font-weight:700;color:#e8eaf6;">{ma["white_pixels"]:,}</p></div>', unsafe_allow_html=True)
                    with mc3:
                        st.markdown(f'<div class="metric-box"><p style="margin:0;font-size:0.75rem;color:#e8eaf6;">Micro-dot Pixels</p><p style="margin:0;font-weight:700;color:#ffd700;">{ma["microdot_pixels"]:,}</p></div>', unsafe_allow_html=True)
                    with mc4:
                        ratio_pct = ma["microdot_ratio"] * 100
                        ratio_color = "#00e694" if ratio_pct > 0.05 else "#ff6b6b"
                        st.markdown(f'<div class="metric-box"><p style="margin:0;font-size:0.75rem;color:#6b7280;">Dot/White Ratio</p><p style="margin:0;font-weight:700;color:{ratio_color};">{ratio_pct:.4f}%</p></div>', unsafe_allow_html=True)

                    st.markdown(f"<p style='font-size:0.78rem;color:#4a5568;margin-top:0.5rem;'>Intensity range searched: <code>{ma['intensity_range']}</code>  ·  Expected: ≥{ma['expected_min_dots']} dots</p>", unsafe_allow_html=True)

            # ── Raw JSON ──────────────────────────────────────────────────
            if show_raw:
                st.markdown("---")
                st.markdown("### 📄 Raw API Response")
                st.json(response)

else:
    # Empty state
    st.markdown("""
<div class="glass-card" style="text-align:center;padding:3rem;">
  <div style="font-size:3rem;">📷</div>
  <h3 style="color:#6b7280;">Upload a QR Code to begin</h3>
  <p style="color:#4a5568;">Supports PNG · JPG · JPEG · BMP · TIFF · WEBP</p>
</div>
""", unsafe_allow_html=True)

    # Show model info even before upload
    st.markdown("### 🤖 Available Models")
    models_info = get_model_list()
    if models_info:
        cols = st.columns(3)
        for i, (name, info) in enumerate(models_info.items()):
            with cols[i]:
                status_icon  = "✅" if info.get("loaded") else "⚠️"
                status_label = "Loaded" if info.get("loaded") else "Demo Mode"
                st.markdown(f"""
<div class="model-card">
  <p style="font-weight:600;color:#e8eaf6;margin:0;">{status_icon} {name}</p>
  <p style="font-size:0.78rem;color:#6b7280;margin:0.4rem 0;">{info['description']}</p>
  <p style="font-size:0.72rem;color:#4a5568;margin:0;"><code>{info['weight_file']}</code></p>
  <p style="font-size:0.72rem;color:{'#00e694' if info.get('loaded') else '#ffd700'};margin-top:0.3rem;">{status_label}</p>
</div>
""", unsafe_allow_html=True)
