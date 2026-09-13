import os
import json
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from PIL import Image
import keras
import tensorflow as tf
import base64
import io
import json
import logging
import csv
from flask import Flask, request, jsonify, render_template_string, send_file
import cv2
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from iqa_module.pipeline import process_retinal_image
from grading_model import LesionAwareSeverityClassifier

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

CLASS_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
RISK_TIERS  = ["MONITOR", "MONITOR", "ENGAGE", "ACT NOW", "ACT NOW"]
RISK_COLORS = ["#10b981", "#10b981", "#f59e0b", "#d94f5c", "#d94f5c"]
BAR_COLORS  = ["#10b981", "#06b6d4", "#f59e0b", "#f97316", "#d94f5c"]

CLINICAL_DESC = [
    "No DR predicted by the model. Clinical review is recommended when appropriate.",
    "Mild DR predicted by the model. Clinical review is recommended when appropriate.",
    "Moderate DR predicted by the model. Review by a qualified ophthalmologist is recommended.",
    "Severe DR predicted by the model. Prompt review by a qualified ophthalmologist is recommended.",
    "Proliferative DR predicted by the model. Prompt specialist review is recommended."
]

AI_SUMMARIES = [
    {
        "summary": "Retinal imaging shows no evidence of diabetic retinopathy. Vascular architecture appears intact with no microaneurysms, exudates, or neovascularization detected.",
        "action": "Continue standard diabetes management. Schedule next retinal screening in 12 months.",
        "hcp": "Automated digital reminder via patient portal. No urgent HCP outreach required.",
        "channel": "Digital"
    },
    {
        "summary": "Early-stage microaneurysms identified in the peripheral retinal field. Consistent with mild non-proliferative diabetic retinopathy. No vision-threatening features present.",
        "action": "Optimise glycaemic and blood pressure control. Repeat retinal imaging in 6-12 months.",
        "hcp": "Flag for primary care physician review. Consider patient education on glycaemic targets.",
        "channel": "Digital + PCP Outreach"
    },
    {
        "summary": "Moderate non-proliferative diabetic retinopathy identified. Microaneurysms, retinal haemorrhages, and hard exudates detected in the central field. Progression risk elevated without intervention.",
        "action": "Ophthalmology referral within 30 days. Evaluate candidacy for anti-VEGF therapy.",
        "hcp": "Flag in CRM for HCP outreach via primary care provider. Consider therapy education for prescriber.",
        "channel": "CRM Flag + HCP Engagement"
    },
    {
        "summary": "Severe non-proliferative diabetic retinopathy detected. Extensive haemorrhages, venous beading, and intraretinal microvascular abnormalities across multiple quadrants. High risk of progression to PDR.",
        "action": "Urgent ophthalmology referral within 1 week. Consider pan-retinal photocoagulation or intravitreal injection.",
        "hcp": "Immediate field rep visit recommended. Prioritise specialist referral pathway and therapy initiation.",
        "channel": "Field Rep + Specialist Referral"
    },
    {
        "summary": "Proliferative diabetic retinopathy confirmed. Active neovascularization and vitreous haemorrhage signs detected. Highest-risk stage with imminent threat to vision.",
        "action": "Same-week specialist intervention required. Anti-VEGF therapy or surgical evaluation indicated.",
        "hcp": "Urgent field rep visit. Flag for immediate specialist coordination. Critical therapy initiation opportunity.",
        "channel": "Urgent Field Rep + Specialist"
    }
]

model = keras.models.load_model('models/diabetic_retinopathy_model.keras')
print("MODEL LAYERS:", [(i, l.name, type(l).__name__) for i, l in enumerate(model.layers)])

calibration_path = 'models/calibration.json'

if os.path.exists(calibration_path):
    with open(calibration_path, 'r') as f:
        calibration_data = json.load(f)
    temperature = calibration_data.get('temperature', 1.0)
else:
    temperature = 1.0
grading_classifier = LesionAwareSeverityClassifier(
    model=model,
    temperature=temperature,
    low_confidence_threshold=0.60
)
temperature_path = os.getenv('DR_TEMPERATURE_FILE', 'models/calibration.json')
configured_temperature = float(os.getenv('DR_TEMPERATURE', '1.0'))
if os.path.exists(temperature_path):
    with open(temperature_path, 'r', encoding='utf-8') as calibration_file:
        configured_temperature = float(json.load(calibration_file)['temperature'])
grading_model = LesionAwareSeverityClassifier(
  model,
  temperature=configured_temperature,
  low_confidence_threshold=float(os.getenv('DR_LOW_CONFIDENCE_THRESHOLD', '0.60')),
)

# --- Step 4: Load REAL validation metrics from Member 4's validation_sim results ---
# Selected run per validation_sim/config/selected_runs.json (gated run, IQA-accepted subset).
VALIDATION_METRICS = {
    'sensitivity': 'N/A',
    'specificity': 'N/A',
    'qwk': 'N/A',
    'n_validation': 'N/A',
}
try:
    _metrics_csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'validation_sim',
        'results', 'gated', '20260912_111805_221820', 'evaluation_metrics.csv'
    )
    with open(_metrics_csv_path, 'r', newline='', encoding='utf-8') as _f:
        _reader = csv.DictReader(_f)
        for _row in _reader:
            if _row.get('subset') == 'IQA Accepted Images (Valid Predictions)':
                VALIDATION_METRICS['sensitivity'] = f"{float(_row['sensitivity']) * 100:.1f}%"
                VALIDATION_METRICS['specificity'] = f"{float(_row['specificity']) * 100:.1f}%"
                VALIDATION_METRICS['qwk'] = f"{float(_row['qwk']):.3f}"
                VALIDATION_METRICS['n_validation'] = _row['count']
                break
    logging.info("Loaded real validation metrics: %s", VALIDATION_METRICS)
except Exception as _exc:
    logging.warning("Could not load validation_sim metrics, falling back to N/A: %s", _exc)

# Log model structure at startup to help diagnose Grad-CAM issues
logging.info("Top-level model layers:")
for i, l in enumerate(model.layers):
    logging.info("  [%d] %s (%s)", i, l.name, type(l).__name__)
    if hasattr(l, 'layers'):
        conv_names = [sl.name for sl in l.layers if 'conv' in sl.name.lower()]
        logging.info("      last 3 conv layers: %s", conv_names[-3:])


def _find_last_conv(layers):
    """Return the last Conv layer from a list, checking by type then name."""
    for l in reversed(layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            return l
    for l in reversed(layers):
        if 'conv' in l.name.lower():
            return l
    return None


def make_gradcam(img_array, model):
    try:
        # Find EfficientNetB3
        sub_model = next(
            (
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.Model)
                and "efficientnet" in layer.name.lower()
            ),
            None
        )

        if sub_model is None:
            logging.warning("Grad-CAM: EfficientNetB3 not found")
            return None

        # Find last convolution layer
        last_conv = _find_last_conv(sub_model.layers)

        if last_conv is None:
            logging.warning("Grad-CAM: no convolution layer found")
            return None

        logging.info(
            "Grad-CAM using sub-model=%s, conv=%s",
            sub_model.name,
            last_conv.name
        )

        # Create a model that gives:
        # 1. Last convolution feature maps
        # 2. EfficientNet output
        feature_model = tf.keras.Model(
            inputs=sub_model.input,
            outputs=[
                last_conv.output,
                sub_model.output
            ]
        )

        with tf.GradientTape() as tape:

            # EfficientNet forward pass
            conv_output, x = feature_model(
                img_array,
                training=False
            )

            # Continue through the outer model
            # after EfficientNet
            sub_model_index = model.layers.index(sub_model)

            for layer in model.layers[sub_model_index + 1:]:
                x = layer(x, training=False)

            predictions = x

            # Predicted class
            predicted_class = tf.argmax(
                predictions[0]
            )

            class_score = predictions[:, predicted_class]

        # Gradient of predicted class
        # with respect to convolution feature maps
        gradients = tape.gradient(
            class_score,
            conv_output
        )

        if gradients is None:
            logging.warning(
                "Grad-CAM: gradients are None"
            )
            return None

        # Average gradients over height and width
        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )

        # Weighted feature maps
        cam = tf.reduce_sum(
            conv_output *
            weights[:, tf.newaxis, tf.newaxis, :],
            axis=-1
        )[0]

        # Keep positive activations
        cam = tf.maximum(cam, 0)

        cam_max = tf.reduce_max(cam)

        if float(cam_max) == 0:
            logging.warning(
                "Grad-CAM: empty activation map"
            )
            return None

        cam = cam / cam_max
        cam = cam.numpy()

        # Resize heatmap
        cam_resized = np.array(
            Image.fromarray(
                (cam * 255).astype(np.uint8)
            ).resize(
                (224, 224),
                Image.BILINEAR
            ),
            dtype=np.float32
        ) / 255.0

        # Create heatmap
        hue = (1.0 - cam_resized) * 0.67

        hi = (hue * 6).astype(int) % 6
        f = hue * 6 - np.floor(hue * 6)

        ones = np.ones_like(f)
        zeros = np.zeros_like(f)

        r = np.select(
            [
                hi == 0,
                hi == 1,
                hi == 2,
                hi == 3,
                hi == 4,
                hi == 5
            ],
            [
                ones,
                1 - f,
                zeros,
                zeros,
                f,
                ones
            ]
        )

        g = np.select(
            [
                hi == 0,
                hi == 1,
                hi == 2,
                hi == 3,
                hi == 4,
                hi == 5
            ],
            [
                f,
                ones,
                ones,
                1 - f,
                zeros,
                zeros
            ]
        )

        b = np.select(
            [
                hi == 0,
                hi == 1,
                hi == 2,
                hi == 3,
                hi == 4,
                hi == 5
            ],
            [
                zeros,
                zeros,
                f,
                ones,
                ones,
                1 - f
            ]
        )

        heatmap = (
            np.stack(
                [r, g, b],
                axis=-1
            ) * 255
        ).astype(np.uint8)

        # Original image
        orig = (
            img_array[0] * 255
        ).astype(np.uint8)

        # Blend
        blended = (
            0.55 * orig +
            0.45 * heatmap
        ).astype(np.uint8)

        # Convert to Base64
        buf = io.BytesIO()

        Image.fromarray(
            blended
        ).save(
            buf,
            format="PNG"
        )

        return base64.b64encode(
            buf.getvalue()
        ).decode()

    except Exception as e:
        logging.warning(
            "Grad-CAM failed: %s",
            e
        )
        return None

def detect_lesions(img):
    """Classical CV lesion detector. img: RGB float(0-1) or uint8 array. Returns (overlay_uint8_rgb, lesion_count)."""
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, dark_thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_thresh = cv2.erode(dark_thresh, np.ones((3, 3), np.uint8), iterations=1)

    _, bright_thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright_thresh = cv2.erode(bright_thresh, np.ones((3, 3), np.uint8), iterations=1)

    overlay = img_uint8.copy()
    lesion_count = 0

    for mask, color in [(dark_thresh, (255, 0, 0)), (bright_thresh, (255, 255, 0))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if 5 < area < 300:
                (x, y), r = cv2.minEnclosingCircle(c)
                cv2.circle(overlay, (int(x), int(y)), int(r) + 2, color, 1)
                lesion_count += 1

    return overlay, lesion_count


def img_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Netra | DR Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&family=Nunito:wght@400;600;700;800&family=Carter+One&family=Slabo+13px&family=Fraunces:ital,wght@0,600;0,700;1,500;1,600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/ScrollTrigger.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #f6f8fb;
  --surface: #ffffff;
  --surface2: #f2f6f5;
  --surface3: #eef1f5;
  --border: #d6e0e4;
  --text: #172033;
  --muted: #8896a1;
  --accent: #0e9f92;
  --accent2: #087267;
}
html { font-size: 18px; }
body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; min-height: 100vh; }
/* HEADER */
.header {
  background: #ffffff;
  border-bottom: 1px solid #0e9f9244;
  padding: 0 40px; height: 68px;
  display: flex; align-items: center; gap: 14px;
}
.logo {
  width: 38px; height: 38px; border-radius: 9px;
  background: linear-gradient(135deg, #0e9f92, #087267);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; flex-shrink: 0;
}
.hnav { display: flex; gap: 28px; align-items: center; margin-left: 36px; }
.hnav a {
  font-family: 'Nunito', sans-serif; font-size: 0.82rem; font-weight: 700;
  color: var(--text); text-decoration: none; cursor: pointer; opacity: 0.72;
  transition: opacity 0.15s ease, color 0.15s ease;
}
.hnav a:hover { opacity: 1; color: var(--accent2); }
.hnav-cta {
  margin-left: auto; background: linear-gradient(135deg, var(--accent), var(--accent2));
  color: #ffffff; border: none; border-radius: 8px; padding: 9px 20px;
  font-family: 'Nunito', sans-serif; font-weight: 800; font-size: 0.8rem;
  cursor: pointer; letter-spacing: 0.01em; white-space: nowrap;
}
.hnav-cta:hover { filter: brightness(1.08); }
.upload-filename {
  margin-top: 14px; padding: 8px 14px; background: rgba(14,159,146,0.12);
  border: 1px solid rgba(14,159,146,0.3); border-radius: 8px; color: var(--accent2);
  font-weight: 700; font-size: 0.82rem; font-family: 'Barlow'; display: inline-block;
}
@media(max-width: 900px) { .hnav { display: none; } }
.htitle { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.02em; }
.hsub { font-size: 0.72rem; color: #087267; margin-top: 2px; }
.hpills { margin-left: auto; display: flex; gap: 6px; }
.hpill {
  background: #0e9f9222; border: 1px solid #0e9f9233;
  color: #087267; font-size: 0.65rem; font-weight: 600;
  padding: 3px 10px; border-radius: 20px;
}
/* LAYOUT */
.main { max-width: 1560px; margin: 0 auto; padding: 32px 56px 60px; }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 16px; padding: 22px;
}
.card-label {
  font-size: 0.67rem; font-weight: 700; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px;
}
/* UPLOAD STATE */
#uploadState { margin-bottom: 0; }
.upload-zone {
  border: 2px dashed var(--border); border-radius: 14px;
  padding: 52px 24px; text-align: center; cursor: pointer;
  transition: all 0.2s; position: relative; background: var(--surface2);
}
.upload-zone:hover { border-color: var(--accent); background: #0e9f920a; }
.upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
.upload-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }
.upload-sub { font-size: 0.8rem; color: var(--muted); line-height: 1.6; }
.upload-sub span { color: var(--accent2); font-weight: 600; }
.upload-hint { font-size: 0.72rem; color: var(--muted); margin-top: 8px; opacity: 0.7; }
.btn-primary {
  width: 100%; margin-top: 14px;
  background: linear-gradient(135deg, #0e9f92, #087267);
  border: none; border-radius: 11px; color: white;
  font-size: 0.88rem; font-weight: 700; padding: 14px;
  cursor: pointer; transition: all 0.2s;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  box-shadow: 0 4px 18px #0e9f9228;
  font-family: 'Inter', sans-serif;
}
.btn-primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 22px #0e9f9240; }
.btn-primary:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; transform: none; }
.btn-secondary {
  width: 100%; margin-top: 10px;
  background: transparent; border: 1px solid var(--border);
  border-radius: 10px; color: var(--muted);
  font-size: 0.78rem; font-weight: 500; padding: 10px;
  cursor: pointer; transition: all 0.2s;
  font-family: 'Inter', sans-serif;
}
.btn-secondary:hover { border-color: var(--accent); color: var(--accent2); }
/* RESULTS STATE */
#resultsState { display: none; }
/* SEVERITY SCALE */
.severity-scale {
  display: flex; margin-bottom: 18px; border-radius: 11px; overflow: hidden;
  border: 1px solid var(--border);
}
.sev-item {
  flex: 1; text-align: center; padding: 10px 6px;
  font-size: 0.63rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.05em; background: var(--surface); color: var(--muted);
  border-right: 1px solid var(--border); transition: all 0.3s;
}
.sev-item:last-child { border-right: none; }
.sev-dot { width: 5px; height: 5px; border-radius: 50%; margin: 0 auto 5px; background: currentColor; opacity: 0.5; }
.sev-item.sev-active { z-index: 1; }
/* TOP GRID */
.top-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 18px; }
@media(max-width: 640px) { .top-grid { grid-template-columns: 1fr; } }
/* IMAGE CARD */
.scan-img-wrap { border-radius: 11px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 10px; }
.scan-img-wrap img { width: 100%; display: block; max-height: 210px; object-fit: cover; }
/* RESULT CARD */
.risk-pill {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 6px 14px; border-radius: 8px;
  font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em;
  margin-bottom: 10px;
}
.risk-dot { width: 7px; height: 7px; border-radius: 50%; }
.result-diagnosis { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin-bottom: 7px; }
.result-conf-row { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
.result-conf-text { font-size: 0.78rem; color: var(--muted); white-space: nowrap; }
.conf-track { flex: 1; background: var(--surface2); border-radius: 100px; height: 5px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 100px; transition: width 0.9s cubic-bezier(0.4,0,0.2,1); }
.result-desc {
  font-size: 0.78rem; color: var(--muted); line-height: 1.65;
  padding: 10px 13px; background: var(--surface2);
  border-radius: 9px; border-left: 3px solid var(--accent); margin-bottom: 14px;
}
.divider { height: 1px; background: var(--border); margin: 13px 0; }
.scores-title { font-size: 0.65rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
.bar-row { margin-bottom: 8px; }
.bar-label { display: flex; justify-content: space-between; font-size: 0.72rem; margin-bottom: 4px; }
.bar-label span:last-child { color: var(--muted); }
.bar-track { background: var(--surface2); border-radius: 5px; height: 5px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 5px; transition: width 0.7s cubic-bezier(0.4,0,0.2,1); width: 0; }
/* GRADCAM */
.gradcam-card { margin-bottom: 18px; }
.gradcam-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin-top: 14px; }
@media(max-width: 560px) { .gradcam-grid { grid-template-columns: 1fr; } }
.gcam-wrap { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
.gcam-wrap img { width: 100%; display: block; max-height: 180px; object-fit: cover; }
.gcam-label { font-size: 0.68rem; color: var(--muted); text-align: center; margin-top: 8px; font-weight: 500; }
.gradcam-note {
  margin-top: 14px; padding: 11px 14px;
  background: #eef7f6; border: 1px solid #08726733;
  border-radius: 10px; font-size: 0.76rem; color: #087267; line-height: 1.65;
}
/* MODEL PERFORMANCE */
.perf-card { margin-bottom: 18px; }
.perf-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }
@media(max-width: 560px) { .perf-grid { grid-template-columns: repeat(2, 1fr); } }
.perf-metric { background: var(--surface2); border-radius: 10px; padding: 14px; text-align: center; }
.perf-val { font-size: 1.25rem; font-weight: 800; color: var(--accent2); margin-bottom: 3px; }
.perf-lbl { font-size: 0.63rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; }
/* AI CARD */
.ai-card { margin-bottom: 0; }
.ai-header { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.ai-icon-box {
  width: 34px; height: 34px; border-radius: 8px; flex-shrink: 0;
  background: #08726722; border: 1px solid #0e9f9233;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.75rem; font-weight: 800; color: #087267;
}
.ai-title { font-size: 0.88rem; font-weight: 700; color: #087267; }
.ai-sub { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }
.ai-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
@media(max-width: 640px) { .ai-grid { grid-template-columns: 1fr; } }
.ai-panel { background: var(--surface2); border-radius: 12px; padding: 16px; border: 1px solid var(--border); }
.ai-panel-title { font-size: 0.65rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 8px; }
.ai-panel-text { font-size: 0.78rem; color: var(--text); line-height: 1.65; }
.channel-tag {
  display: inline-block; margin-top: 10px;
  background: #0e9f9222; border: 1px solid #0e9f9233;
  color: #087267; font-size: 0.68rem; font-weight: 600;
  padding: 4px 10px; border-radius: 20px;
}
/* FADE IN */
.fade-in { animation: fadeIn 0.4s ease forwards; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.spinner { width: 15px; height: 15px; border: 2px solid rgba(255,255,255,0.25); border-top-color: white; border-radius: 50%; animation: spin 0.7s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.footer {
  margin-top: 56px; padding-top: 40px; border-top: 1px solid var(--border);
  padding-bottom: 0px;
}
.footer-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 40px; flex-wrap: wrap; margin-bottom: 28px; }
.footer-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.footer-brand .logo { width: 32px; height: 32px; font-size: 0.9rem; }
.footer-brand .htitle { font-size: 1.3rem; }
.footer-desc { color: var(--muted); font-size: 0.82rem; line-height: 1.65; max-width: 420px; font-family: 'Barlow'; }
.footer-links { display: flex; gap: 36px; flex-wrap: wrap; }
.footer-links-col { display: flex; flex-direction: column; gap: 10px; }
.footer-links-col .flabel { font-family: 'Nunito'; font-weight: 800; font-size: 0.68rem; letter-spacing: 0.08em; color: var(--accent2); text-transform: uppercase; margin-bottom: 4px; }
.footer-links-col a { color: var(--text); text-decoration: none; font-size: 0.82rem; font-family: 'Barlow'; opacity: 0.8; }
.footer-links-col a:hover { opacity: 1; color: var(--accent2); text-decoration: underline; }
.footer-bottom {
  display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
  border-top: 1px solid var(--border); padding: 18px 0 22px; color: var(--muted); font-size: 0.7rem; font-family: 'Nunito';
}
.footer-bottom b { color: var(--text); }
.footer-disclaimer { text-align: center; color: var(--muted); font-size: 0.66rem; padding: 14px 0 26px; font-family: 'Nunito'; letter-spacing: 0.01em; }
.footer a { color: var(--accent2); text-decoration: none; }
.footer a:hover { text-decoration: underline; }
/* ================== LIGHT THEME REFINEMENTS (teal palette) ================== */
body {
  font-family: 'Barlow', 'Inter', sans-serif;
  background-image: radial-gradient(ellipse at 50% 0%, rgba(14,159,146,0.07) 0%, transparent 75%);
  background-attachment: fixed;
}
.card { box-shadow: 0 8px 22px rgba(20,43,58,0.05); }
.htitle { font-family: 'Carter One', cursive; font-weight: 400; letter-spacing: 0.01em; font-size: 1.7rem; background: linear-gradient(135deg, var(--accent2), var(--accent)); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.hsub, .hpill, .card-label, .risk-pill, .scores-title, .perf-lbl, .ai-title, .ai-panel-title, .channel-tag, .gcam-label {
  font-family: 'Nunito', sans-serif;
}
.result-diagnosis { font-family: 'Slabo 13px', serif; font-weight: 800; }
.upload-title { font-family: 'Slabo 13px', serif; font-weight: 800; }

/* INTRO / LANDING PAGE */
#introState { max-width: 1560px; margin: 30px auto; text-align: center; padding: 0 0px; }
.intro-badge {
  display: inline-block; background: rgba(14,159,146,0.1); color: #087267;
  font-family: 'Nunito', sans-serif; font-weight: 800; font-size: 0.68rem;
  letter-spacing: 0.08em; padding: 6px 14px; border-radius: 20px; margin-bottom: 20px;
}
.intro-title { font-family: 'Carter One', cursive; font-weight: 400; font-size: 3rem; color: var(--text); margin-bottom: 12px; }
.intro-tagline { font-family: 'Barlow', sans-serif; font-size: 1rem; color: var(--muted); margin-bottom: 40px; line-height: 1.6; max-width: 480px; margin-left: auto; margin-right: auto; }
.intro-features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 40px; text-align: left; }
@media(max-width: 700px) { .intro-features { grid-template-columns: 1fr; } }
.intro-feature { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 20px; box-shadow: 0 6px 18px rgba(20,43,58,0.04); }
.intro-feature-icon {
  width: 30px; height: 30px; border-radius: 8px; background: var(--accent); color: white;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Nunito', sans-serif; font-weight: 800; font-size: 0.85rem; margin-bottom: 12px;
}
.intro-feature-title { font-family: 'Slabo 13px', serif; font-weight: 800; font-size: 0.95rem; margin-bottom: 6px; color: var(--text); }
.intro-feature-text { font-family: 'Barlow', sans-serif; font-size: 0.8rem; color: var(--muted); line-height: 1.55; }
.intro-cta { max-width: 260px; margin: 0 auto; }
.intro-disclaimer { margin-top: 20px; font-family: 'Nunito', sans-serif; font-size: 0.68rem; color: var(--muted); }
#uploadState { display: none; }

/* ================== LANDING PAGE (Netra) ================== */
.lp-serif { font-family: 'Fraunces', serif; }
.lp-hero { max-width: 1020px; margin: 30px auto 10px; text-align: center; position: relative; padding: 10px 10px 20px; }
.lp-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 30px;
  padding: 6px 14px; font-family: 'Nunito'; font-size: 0.65rem; font-weight: 800;
  letter-spacing: 0.06em; color: var(--accent2); margin-bottom: 22px;
  box-shadow: 0 4px 14px rgba(20,43,58,0.05);
}
.lp-badge .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.lp-hero h1 { font-size: 3.6rem; line-height: 1.08; margin: 0 0 22px; }
.lp-hero h1 .line1 { display: block; font-weight: 700; color: var(--text); }
.lp-hero h1 .line2 { display: block; font-style: italic; font-weight: 500; color: var(--accent2); }
.lp-hero p.lp-sub { max-width: 640px; margin: 0 auto 26px; color: var(--muted); font-size: 1rem; line-height: 1.65; font-family: 'Barlow'; }
.lp-ctas { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 6px; }
.lp-btn-secondary {
  background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 10px;
  padding: 13px 20px; font-size: 0.82rem; font-weight: 700; cursor: pointer; font-family: 'Nunito';
}
.lp-float { position: absolute; width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 18px; box-shadow: 0 8px 18px rgba(20,43,58,0.12); }
.lp-f1 { top: 6px; left: 2%; background: var(--accent); }
.lp-f2 { top: -2px; right: 4%; background: var(--text); }
.lp-f3 { bottom: 14px; right: -6px; background: #d9a441; }

.lp-stats { max-width: 760px; margin: 10px auto 54px; display: flex; justify-content: center; gap: 70px; }
.lp-stat { text-align: center; }
.lp-stat .num { font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.4rem; color: var(--text); }
.lp-stat .lbl { font-family: 'Nunito'; font-size: 0.6rem; font-weight: 800; letter-spacing: 0.06em; color: var(--muted); margin-top: 3px; text-transform: uppercase; }

.lp-section { margin-bottom: 64px; }
.lp-label { text-align: center; font-family: 'Nunito'; font-weight: 800; font-size: 0.75rem; letter-spacing: 0.1em; color: var(--accent2); text-transform: uppercase; margin-bottom: 12px; }
.lp-title { text-align: center; font-size: 2.1rem; font-weight: 700; margin: 0 0 10px; color: var(--text); font-family: 'Barlow'; }
.lp-desc { text-align: center; color: var(--muted); font-size: 0.82rem; max-width: 460px; margin: 0 auto 28px; line-height: 1.6; font-family: 'Barlow'; }

.lp-problem-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
@media(max-width:700px){ .lp-problem-grid { grid-template-columns: 1fr; } }
.lp-problem-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 30px 26px; border-top: 3px solid var(--border); }
.lp-problem-card.mid { border-top-color: #d9a441; }
.lp-problem-card.hi { background: #eef7f6; border-top-color: var(--accent); }
.lp-problem-icon { width: 54px; height: 54px; border-radius: 14px; background: var(--surface3); display: flex; align-items: center; justify-content: center; font-size: 24px; margin-bottom: 20px; }
.lp-problem-card.hi .lp-problem-icon { background: rgba(14,159,146,0.15); }
.lp-problem-card h3 { font-size: 1.3rem; margin: 0 0 10px; font-family: 'Barlow'; }
.lp-problem-card p { font-size: 0.74rem; color: var(--muted); line-height: 1.5; margin: 0; font-family: 'Barlow'; }
.lp-tag { font-family: 'Nunito'; font-size: 0.6rem; font-weight: 800; color: var(--accent2); letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px; display: block; }

.lp-pipe-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
@media(max-width:800px){ .lp-pipe-grid { grid-template-columns: 1fr 1fr; } }
.lp-pipe-card { background: var(--surface); border: 1px solid var(--border); border-top: 3px solid var(--accent); border-radius: 16px; padding: 22px; }
.lp-pipe-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.lp-pipe-num { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.2rem; color: #c7d2d6; }
.lp-pipe-badge { font-family: 'Nunito'; font-size: 0.55rem; font-weight: 800; color: var(--accent2); background: rgba(14,159,146,0.1); padding: 2px 7px; border-radius: 10px; letter-spacing: 0.05em; }
.lp-pipe-card h4 { font-size: 1.15rem; margin: 0 0 8px; font-family: 'Barlow'; }
.lp-pipe-card p { font-size: 0.72rem; color: var(--muted); line-height: 1.45; margin: 0; font-family: 'Barlow'; }

.lp-visual { width: 100%; height: 66px; border-radius: 8px; margin-bottom: 12px; position: relative; overflow: hidden; background: #0f1620; }
.lp-v-retina { position: absolute; inset: 0; border-radius: 50%; width: 54px; height: 54px; margin: auto; background: radial-gradient(circle at 40% 35%, #c0532b, #1a0805 78%); }
.lp-v-blur .lp-v-retina { filter: blur(2.5px); opacity: 0.6; }
.lp-v-blur .lp-v-badge { position: absolute; top: 6px; right: 8px; background: #d94f5c; color: white; font-size: 8px; font-weight: 800; font-family: 'Nunito'; padding: 2px 6px; border-radius: 5px; }
.lp-v-heat { position: absolute; inset: 0; border-radius: 50%; width: 54px; height: 54px; margin: auto; background: radial-gradient(circle at 45% 40%, #ff5a3c 0%, #ffce45 35%, #1a3a52 70%, #08131c 100%); }
.lp-v-lesion { position: absolute; inset: 0; }
.lp-v-lesion .dot { position: absolute; width: 5px; height: 5px; border-radius: 50%; }
.lp-v-report { display: flex; align-items: center; justify-content: center; background: #eef7f6; }
.lp-v-report .doc { width: 34px; height: 42px; background: white; border-radius: 3px; box-shadow: 0 3px 8px rgba(20,43,58,0.12); position: relative; }
.lp-v-report .doc::before, .lp-v-report .doc::after { content: ''; position: absolute; left: 6px; right: 6px; height: 2px; background: var(--border); border-radius: 2px; }
.lp-v-report .doc::before { top: 10px; }
.lp-v-report .doc::after { top: 16px; width: 60%; }
.lp-v-report .badge-ok { position: absolute; bottom: -4px; right: -4px; width: 17px; height: 17px; border-radius: 50%; background: var(--accent); color: white; display: flex; align-items: center; justify-content: center; font-size: 9px; }

.lp-cap-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
@media(max-width:700px){ .lp-cap-grid { grid-template-columns: 1fr; } }
.lp-cap-item { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 26px; transition: box-shadow 0.2s ease, transform 0.2s ease; }
.lp-cap-item:hover { box-shadow: 0 10px 22px rgba(20,43,58,0.08); transform: translateY(-2px); }
.lp-cap-icon { width: 52px; height: 52px; border-radius: 14px; background: rgba(14,159,146,0.12); color: var(--accent2); display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.3rem; margin-bottom: 20px; }
.lp-cap-item h4 { font-size: 1.25rem; margin: 0 0 10px; font-family: 'Barlow'; }
.lp-cap-item p { font-size: 0.72rem; color: var(--muted); line-height: 1.5; margin: 0; font-family: 'Barlow'; }

.lp-cta-band {
  padding: 26px; background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  display: flex; align-items: center; justify-content: space-between; gap: 20px; flex-wrap: wrap;
  margin-bottom: 30px;
}
.lp-cta-label { font-family: 'Nunito'; font-size: 0.62rem; font-weight: 800; letter-spacing: 0.07em; color: var(--accent2); text-transform: uppercase; margin-bottom: 6px; }
.lp-cta-band h3 { font-size: 1.05rem; margin: 0 0 4px; max-width: 320px; font-family: 'Barlow'; }
.lp-cta-band p { color: var(--muted); font-size: 0.76rem; margin: 0; max-width: 300px; font-family: 'Barlow'; }
.lp-btn-launch {
  background: var(--text); color: white; border: none; border-radius: 10px;
  padding: 13px 20px; font-size: 0.8rem; font-weight: 700; cursor: pointer; font-family: 'Nunito';
  display: flex; align-items: center; gap: 8px; white-space: nowrap;
}
.lp-btn-launch .arrow { width: 20px; height: 20px; border-radius: 50%; background: var(--accent); display: flex; align-items: center; justify-content: center; font-size: 10px; }
.lp-footnote { text-align: center; color: var(--muted); font-size: 0.65rem; font-family: 'Nunito'; font-weight: 700; margin-bottom: 10px; }
.lp-reveal { opacity: 0; transform: translateY(18px); }
</style>
</head>
<body>
<header class="header">
  <div class="logo">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M1 12C1 12 5 4 12 4C19 4 23 12 23 12C23 12 19 20 12 20C5 20 1 12 1 12Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="12" cy="12" r="3.5" stroke="white" stroke-width="2"/>
    </svg>
  </div>
  <div>
    <div class="htitle">Netra</div>
    <div class="hsub">AI-Assisted Diabetic Retinopathy Screening</div>
  </div>
  <nav class="hnav">
    <a onclick="window.scrollTo({top:0, behavior:'smooth'})">Home</a>
    <a onclick="document.getElementById('howNetraWorks').scrollIntoView({behavior:'smooth'})">How It Works</a>
    <a onclick="document.getElementById('capabilities').scrollIntoView({behavior:'smooth'})">Capabilities</a>
  </nav>
  <button class="hnav-cta" id="hnavCta" onclick="startApp()">Start Screening</button>
</header>
<main class="main">
  <!-- INTRO / LANDING STATE -->
  <div id="introState">

    <section class="lp-hero">
      <div class="lp-float lp-f1 lp-reveal">&#128737;&#65039;</div>
      <div class="lp-float lp-f2 lp-reveal">&#128269;</div>
      <div class="lp-float lp-f3 lp-reveal">&#128202;</div>
      <div class="lp-badge lp-reveal"><span class="dot"></span>AI-ASSISTED RETINAL SCREENING &middot; EFFICIENTNETB3</div>
      <h1 class="lp-reveal">
        <span class="lp-serif line1">See Diabetic Retinopathy</span>
        <span class="lp-serif line2">Before It Steals Sight.</span>
      </h1>
      <p class="lp-sub lp-reveal">Netra reads retinal fundus photographs the way a specialist would &mdash; grading severity, checking image quality first, and showing exactly what it saw so nothing is missed and nothing is hidden.</p>
      <div class="lp-ctas lp-reveal">
        <button class="btn-primary" style="width:auto; margin-top:0; padding:13px 22px;" onclick="startApp()">Start Screening &rarr;</button>
        <button class="lp-btn-secondary" onclick="document.getElementById('howNetraWorks').scrollIntoView({behavior:'smooth'})">See How It Works</button>
      </div>
    </section>

    <div class="lp-stats">
      <div class="lp-stat lp-reveal"><div class="num" data-count="96.7" data-suffix="%">0</div><div class="lbl">Sensitivity</div></div>
      <div class="lp-stat lp-reveal"><div class="num" data-count="3">0</div><div class="lbl">Screening Stages</div></div>
      <div class="lp-stat lp-reveal"><div class="num" data-count="94">0</div><div class="lbl">Validation Images</div></div>
    </div>

    <div class="lp-section">
      <div class="lp-label lp-reveal">THE SCREENING BLINDSPOT</div>
      <div class="lp-title lp-reveal">Manual Screening Doesn't Scale. Early DR Often Goes Unnoticed.</div>
      <div class="lp-problem-grid">
        <div class="lp-problem-card lp-reveal">
          <div class="lp-problem-icon">&#129658;&#8205;&#9877;&#65039;</div>
          <h3>Manual Grading</h3>
          <p>Relies on specialist availability &mdash; results can take days and vary between readers.</p>
        </div>
        <div class="lp-problem-card mid lp-reveal">
          <div class="lp-problem-icon">&#128065;&#65039;&#8205;&#128488;&#65039;</div>
          <h3>Silent Progression</h3>
          <p>Diabetic retinopathy often shows no symptoms until vision loss has already begun.</p>
        </div>
        <div class="lp-problem-card hi lp-reveal">
          <span class="lp-tag">The Critical Question</span>
          <h3>Does this retina show signs of DR right now?</h3>
          <p>Netra answers with a graded severity, visual evidence, and a clear referral recommendation.</p>
        </div>
      </div>
    </div>

    <div class="lp-section" id="howNetraWorks">
      <div class="lp-label lp-reveal">SCREENING PIPELINE</div>
      <div class="lp-title lp-reveal">How Netra Works</div>
      <div class="lp-desc lp-reveal">Four stages turn a single retinal photo into a clinically useful screening decision.</div>
      <div class="lp-pipe-grid">
        <div class="lp-pipe-card lp-reveal">
          <div class="lp-visual lp-v-blur"><div class="lp-v-retina"></div><div class="lp-v-badge">BLURRY</div></div>
          <div class="lp-pipe-top"><span class="lp-pipe-num">01</span><span class="lp-pipe-badge">ACTIVE</span></div>
          <h4>Quality Check</h4>
          <p>Automatically rejects blurry or poorly-lit scans before they reach the model.</p>
        </div>
        <div class="lp-pipe-card lp-reveal">
          <div class="lp-visual"><div class="lp-v-heat"></div></div>
          <div class="lp-pipe-top"><span class="lp-pipe-num">02</span><span class="lp-pipe-badge">ACTIVE</span></div>
          <h4>AI Grading</h4>
          <p>EfficientNetB3 grades severity across five ICDR classes, with calibrated confidence.</p>
        </div>
        <div class="lp-pipe-card lp-reveal">
          <div class="lp-visual">
            <div class="lp-v-retina"></div>
            <div class="lp-v-lesion">
              <div class="dot" style="top:16px; left:34px; background:#d94f5c;"></div>
              <div class="dot" style="top:30px; left:42px; background:#d94f5c;"></div>
              <div class="dot" style="top:38px; left:24px; background:#d9a441;"></div>
            </div>
          </div>
          <div class="lp-pipe-top"><span class="lp-pipe-num">03</span><span class="lp-pipe-badge">ACTIVE</span></div>
          <h4>Lesion Detection</h4>
          <p>Highlights lesion candidates and a Grad-CAM heatmap for visual interpretability.</p>
        </div>
        <div class="lp-pipe-card lp-reveal">
          <div class="lp-visual lp-v-report"><div class="doc"><div class="badge-ok">&#10003;</div></div></div>
          <div class="lp-pipe-top"><span class="lp-pipe-num">04</span><span class="lp-pipe-badge">ACTIVE</span></div>
          <h4>Clinical Report</h4>
          <p>Packages the result into a downloadable PDF with a referral recommendation.</p>
        </div>
      </div>
    </div>

    <div class="lp-section" id="capabilities">
      <div class="lp-label lp-reveal">SCREENING ENGINE</div>
      <div class="lp-title lp-reveal">Key Capabilities</div>
      <div class="lp-cap-grid">
        <div class="lp-cap-item lp-reveal"><div class="lp-cap-icon">&#10003;</div><h4>Automated Quality Gating</h4><p>Rejects unusable images before grading, so results are never built on bad data.</p></div>
        <div class="lp-cap-item lp-reveal"><div class="lp-cap-icon">&#9678;</div><h4>Explainable Grad-CAM Heatmaps</h4><p>Visualizes exactly which retinal regions drove the model's decision.</p></div>
        <div class="lp-cap-item lp-reveal"><div class="lp-cap-icon">&#9670;</div><h4>Lesion Candidate Overlay</h4><p>Classical CV highlights dark and bright lesion candidates for visual review.</p></div>
        <div class="lp-cap-item lp-reveal"><div class="lp-cap-icon">&#9873;</div><h4>Calibrated Confidence Flagging</h4><p>Low-confidence predictions are automatically routed for human review.</p></div>
        <div class="lp-cap-item lp-reveal"><div class="lp-cap-icon">&#9636;</div><h4>Honest Validation Reporting</h4><p>Real external validation metrics, including known limitations, shown transparently.</p></div>
        <div class="lp-cap-item lp-reveal"><div class="lp-cap-icon">&#8681;</div><h4>One-Click Clinical Report</h4><p>Generates a shareable PDF with grade, evidence, and referral guidance.</p></div>
      </div>
    </div>

    <div class="lp-cta-band lp-reveal">
      <div>
        <div class="lp-cta-label">Get Started</div>
        <h3 class="lp-serif">From a Photo to a Screening Decision.</h3>
        <p>Upload a retinal fundus image and get a graded, explainable result in seconds.</p>
      </div>
      <button class="lp-btn-launch" onclick="startApp()">Launch Screening Tool <span class="arrow">&rarr;</span></button>
    </div>

    <div class="lp-footnote">NETRA &middot; AI-ASSISTED DECISION SUPPORT &middot; NOT A DIAGNOSIS</div>
  </div>
  <!-- UPLOAD STATE -->
  <div id="uploadState">
    <div class="card" style="margin-bottom: 18px;">
      <div class="card-label">Upload Retinal Image</div>
      <div class="upload-zone" id="uploadZone">
        <input type="file" id="imageInput" accept="image/*">
        <div class="upload-title">Upload a retinal fundus image</div>
        <div class="upload-sub"><span>Click to browse</span> or drag and drop</div>
        <div class="upload-hint">PNG, JPG, JPEG supported</div>
        <div class="upload-filename" id="uploadFileName" style="display:none;"></div>
      </div>
      <button class="btn-primary" id="analyzeBtn" onclick="analyze()" disabled>
        <span id="btnText">Select an image to begin</span>
        <div class="spinner" id="spinner" style="display:none"></div>
      </button>
    </div>
  </div>
  <!-- QUALITY REJECT STATE -->
  <div id="qualityRejectState" class="card" style="display:none; border-color:#d94f5c;">
    <div class="card-label" style="color:#d94f5c;">Image Rejected — Quality Check Failed</div>
    <div style="font-size:0.85rem; color:var(--text); margin-bottom:10px;">
      Quality score: <span id="rejectScore" style="font-weight:700;"></span>
    </div>
    <div style="font-size:0.82rem; color:#fca5a5; margin-bottom:10px;">
      Reason: <span id="rejectReason"></span>
    </div>
    <div class="result-desc" style="border-left-color:#d94f5c;">
      <span id="rejectGuidance"></span>
    </div>
    <button class="btn-primary" style="margin-top:14px;" onclick="resetFromReject()">Recapture Image</button>
  </div>
  <!-- RESULTS STATE -->
  <div id="resultsState">
    <!-- Severity Scale -->
    <div class="severity-scale" id="severityScale">
      <div class="sev-item" id="sev0"><div class="sev-dot"></div>No DR</div>
      <div class="sev-item" id="sev1"><div class="sev-dot"></div>Mild</div>
      <div class="sev-item" id="sev2"><div class="sev-dot"></div>Moderate</div>
      <div class="sev-item" id="sev3"><div class="sev-dot"></div>Severe</div>
      <div class="sev-item" id="sev4"><div class="sev-dot"></div>Proliferative</div>
    </div>
    <!-- Top Grid: Scan + Result -->
    <div class="top-grid">
      <div class="card">
        <div class="card-label">Retinal Scan</div>
        <div class="scan-img-wrap">
          <img id="scanPreview" alt="Retinal scan">
        </div>
        <button class="btn-secondary" onclick="resetToUpload()">Upload another image</button>
      </div>
      <div class="card" id="resultCard">
        <div class="card-label">Classification Result</div>
        <div id="resultBody"></div>
      </div>
    </div>
    <!-- Grad-CAM -->
    <div class="card gradcam-card" id="gradcamCard" style="display:none;">
      <div class="card-label">Model Attention &mdash; Grad-CAM Visualization</div>
      <div class="gradcam-grid">
        <div>
          <div class="gcam-wrap"><img id="origImg" alt="Original scan"></div>
          <div class="gcam-label">Original Retinal Scan</div>
        </div>
        <div>
          <div class="gcam-wrap"><img id="camImg" alt="Attention heatmap"></div>
          <div class="gcam-label">AI Attention Heatmap &mdash; red zones indicate highest model focus</div>
        </div>
        <div>
          <div class="gcam-wrap"><img id="lesionImg" alt="Lesion overlay"></div>
          <div class="gcam-label" id="lesionLabel">Lesion Overlay</div>
        </div>
      </div>
      <div class="gradcam-note">
    Red zones indicate regions receiving higher attention from the model during classification. 
    This visualization provides an additional interpretability signal alongside the model prediction.
</div>
    </div>
    <!-- Model Performance -->
    <div class="card perf-card">
      <div class="card-label">Model Performance &mdash; External Validation (IDRiD, IQA-Accepted)</div>
      <div class="perf-grid">
        <div class="perf-metric"><div class="perf-val">{{ sensitivity }}</div><div class="perf-lbl">Sensitivity (Referable DR)</div></div>
        <div class="perf-metric"><div class="perf-val" style="color:#f59e0b">{{ specificity }}</div><div class="perf-lbl">Specificity</div></div>
        <div class="perf-metric"><div class="perf-val">{{ qwk }}</div><div class="perf-lbl">QWK (Cohen's Kappa)</div></div>
        <div class="perf-metric"><div class="perf-val">{{ n_validation }}</div><div class="perf-lbl">External Validation Images</div></div>
      </div>
      <div style="background:#0a1929;border:1px solid #3b82f644;border-radius:8px;padding:10px 14px;margin-top:12px;font-size:0.78rem;color:#93c5fd;line-height:1.4;">
        <b>Design philosophy &mdash; screening-first:</b> Like mammography and other high-stakes screening tools, this model is tuned to prioritize catching every possible case of DR ({{ sensitivity }} sensitivity) rather than minimizing false alarms. This intentionally trades off specificity ({{ specificity }}), so flagged cases are designed to route to a human ophthalmologist for confirmation &mdash; never as a standalone diagnosis. Threshold tuning to improve specificity is an active, ongoing area of the project (see validation_sim/README.md for full methodology).
      </div>
    </div>
    <!-- AI Clinical Intelligence -->
    <div class="card ai-card">
      <div class="ai-header">
        <div class="ai-icon-box">AI</div>
        <div>
          <div class="ai-title">AI-Generated Clinical Intelligence</div>
          <div class="ai-sub">Automated insight generation &middot; Pharma commercial decision support</div>
        </div>
      </div>
      <div class="ai-grid">
        <div class="ai-panel">
          <div class="ai-panel-title">Clinical Summary</div>
          <div class="ai-panel-text" id="aiSummary"></div>
        </div>
        <div class="ai-panel">
          <div class="ai-panel-title">Recommended Action</div>
          <div class="ai-panel-text" id="aiAction"></div>
        </div>
        <div class="ai-panel">
          <div class="ai-panel-title">HCP Engagement</div>
          <div class="ai-panel-text" id="aiHcp"></div>
          <div class="channel-tag" id="aiChannel"></div>
        </div>
      </div>
    </div>
  </div>
</main>
<footer class="footer">
  <div class="footer-top">
    <div style="max-width: 440px;">
      <div class="footer-brand">
        <div class="logo">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M1 12C1 12 5 4 12 4C19 4 23 12 23 12C23 12 19 20 12 20C5 20 1 12 1 12Z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="12" cy="12" r="3.5" stroke="white" stroke-width="2"/>
          </svg>
        </div>
        <div class="htitle">Netra</div>
      </div>
      <div class="footer-desc">Quality-gated, explainable diabetic retinopathy screening &mdash; built to catch bad images before they become bad diagnoses, and to show every lesion and confidence score behind each grade. Designed as decision support for real screening programs, not a replacement for an ophthalmologist.</div>
    </div>
    <div class="footer-links">
      <div class="footer-links-col">
        <div class="flabel">Project</div>
        <a onclick="window.scrollTo({top:0, behavior:'smooth'})">Home</a>
        <a onclick="document.getElementById('howNetraWorks').scrollIntoView({behavior:'smooth'})">How It Works</a>
        <a onclick="document.getElementById('capabilities').scrollIntoView({behavior:'smooth'})">Capabilities</a>
      </div>
      <div class="footer-links-col">
        <div class="flabel">References</div>
        <a href="https://github.com/priyankaraghunathan15/diabetic-retinopathy-detection" target="_blank">GitHub Repository</a>
        <a href="https://www.kaggle.com/competitions/aptos2019-blindness-detection" target="_blank">APTOS 2019 Dataset</a>
      </div>
    </div>
  </div>
  <div class="footer-bottom">
    <div>Built by <b>Team Netra</b></div>
    <div>&copy; 2026 Netra &middot; Built on EfficientNetB3, trained on APTOS 2019</div>
  </div>
  <div class="footer-disclaimer">NETRA &middot; AI-ASSISTED SCREENING DECISION SUPPORT &middot; NOT A DIAGNOSIS &middot; FOR DEMONSTRATION &amp; EDUCATIONAL PURPOSES ONLY</div>
</footer>
<script>
const LABELS      = ["No DR","Mild DR","Moderate DR","Severe DR","Proliferative DR"];
const TIERS       = ["MONITOR","MONITOR","ENGAGE","ACT NOW","ACT NOW"];
const TIER_COLORS = ["#10b981","#10b981","#f59e0b","#d94f5c","#d94f5c"];
const BAR_COLORS  = ["#10b981","#06b6d4","#f59e0b","#f97316","#d94f5c"];
document.getElementById('imageInput').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('scanPreview').src = ev.target.result;
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('btnText').textContent = 'Analyze Image';
    document.getElementById('uploadZone').style.borderColor = '#0e9f92';
    const fnEl = document.getElementById('uploadFileName');
    fnEl.textContent = '\u2713 ' + file.name;
    fnEl.style.display = 'inline-block';
  };
  reader.readAsDataURL(file);
});
function startApp() {
  document.getElementById('introState').style.display = 'none';
  document.getElementById('uploadState').style.display = 'block';
  updateNavCta(false);
  window.scrollTo({top:0, behavior:'smooth'});
}

function goHome() {
  document.getElementById('introState').style.display = 'block';
  document.getElementById('uploadState').style.display = 'none';
  document.getElementById('resultsState').style.display = 'none';
  document.getElementById('qualityRejectState').style.display = 'none';
  updateNavCta(true);
  window.scrollTo({top:0, behavior:'smooth'});
}

function updateNavCta(isHome) {
  const btn = document.getElementById('hnavCta');
  if (!btn) return;
  if (isHome) {
    btn.textContent = 'Start Screening';
    btn.onclick = startApp;
  } else {
    btn.textContent = '\u2190 Back to Home';
    btn.onclick = goHome;
  }
}

async function analyze() {
  const input = document.getElementById('imageInput');
  if (!input.files[0]) return;
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  document.getElementById('btnText').textContent = 'Checking image quality...';
  document.getElementById('spinner').style.display = 'block';

  try {
    const qFormData = new FormData();
    qFormData.append('file', input.files[0]);
    const qres = await fetch('/check_quality', { method: 'POST', body: qFormData });
    const qdata = await qres.json();

    if (!qdata.is_usable) {
      showQualityReject(qdata);
      btn.disabled = false;
      document.getElementById('btnText').textContent = 'Analyze Image';
      document.getElementById('spinner').style.display = 'none';
      return;
    }

    document.getElementById('btnText').textContent = 'Analyzing...';
    const formData = new FormData();
    formData.append('file', input.files[0]);
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    data.quality_score = qdata.quality_score;
    renderResults(data);
  } catch(e) {
    alert('Error analyzing image. Please try again.');
  }
  btn.disabled = false;
  document.getElementById('btnText').textContent = 'Analyze Image';
  document.getElementById('spinner').style.display = 'none';
}

function showQualityReject(qdata) {
  document.getElementById('uploadState').style.display = 'none';
  document.getElementById('qualityRejectState').style.display = 'block';
  document.getElementById('rejectReason').textContent = qdata.reason || 'Image quality insufficient for grading.';
  document.getElementById('rejectGuidance').textContent = qdata.clinical_guidance || 'Please recapture the image.';
  document.getElementById('rejectScore').textContent = (qdata.quality_score * 100).toFixed(0) + '/100';
}

function resetFromReject() {
  document.getElementById('qualityRejectState').style.display = 'none';
  resetToUpload();
}
async function downloadReport() {
  const data = window.currentReportData;
  if (!data) return;
  const btn = document.getElementById('downloadReportBtn');
  const originalText = btn.textContent;
  btn.textContent = 'Generating PDF...';
  btn.disabled = true;
  try {
    const res = await fetch('/generate_report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error('failed');
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'DR_Screening_Report.pdf';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (e) {
    alert('Could not generate report. Please try again.');
  }
  btn.textContent = originalText;
  btn.disabled = false;
}

function renderResults(data) {
  window.currentReportData = data;
  const i = data.predicted_class;
  const color = TIER_COLORS[i];
  document.getElementById('uploadState').style.display = 'none';
  document.getElementById('resultsState').style.display = 'block';
  document.getElementById('resultsState').classList.add('fade-in');
  const sevColors = ["#10b981","#10b981","#f59e0b","#d94f5c","#d94f5c"];
  const sevBg     = ["#052e16","#052e16","#1c1408","#1c0a0a","#1c0a0a"];
  for (let j = 0; j < 5; j++) {
    const el = document.getElementById('sev' + j);
    if (j === i) {
      el.style.background   = sevBg[i];
      el.style.color        = sevColors[i];
      el.style.borderBottom = '2px solid ' + sevColors[i];
      el.querySelector('.sev-dot').style.opacity = '1';
    } else {
      el.style.background   = 'var(--surface)';
      el.style.color        = 'var(--muted)';
      el.style.borderBottom = 'none';
    }
  }
  document.getElementById('resultBody').innerHTML = `
    ${data.confidence_flag === 'low' ? `
    <div style="background:#451a03;border:1px solid #f59e0b66;border-radius:8px;padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
      <span style="font-size:1.1rem;line-height:1;">⚠️</span>
      <span style="color:#fbbf24;font-size:0.82rem;font-weight:600;">Low confidence — please have this case reviewed by an ophthalmologist</span>
    </div>` : ''}
    <div class="risk-pill" style="background:${color}18;border:1px solid ${color}44">
      <div class="risk-dot" style="background:${color}"></div>
      <span style="color:${color}">${TIERS[i]}</span>
    </div>
    <div class="result-diagnosis" style="color:${color}">${LABELS[i]}</div>
    <div class="result-conf-row">
      <span class="result-conf-text">${(data.confidence*100).toFixed(1)}% confidence</span>
      <div class="conf-track">
        <div class="conf-fill" id="confFill" style="background:${color};width:0%"></div>
      </div>
    </div>
    <div style="font-size:0.72rem; color:var(--muted); margin-bottom:10px;">Input image quality score: <b style="color:var(--text)">${data.quality_score ? (data.quality_score*100).toFixed(0) : 'N/A'}/100</b></div>
    <div class="result-desc">${data.description}</div>
    <div class="divider"></div>
    <div class="scores-title">Confidence Scores</div>
    ${data.probabilities.map((p,j) => `
      <div class="bar-row">
        <div class="bar-label">
          <span>${LABELS[j]}</span>
          <span>${(p*100).toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" id="bar${j}" style="background:${BAR_COLORS[j]}"></div>
        </div>
      </div>`).join('')}
    <button id="downloadReportBtn" onclick="downloadReport()" style="margin-top:16px;width:100%;padding:12px;background:linear-gradient(135deg,#4f46e5,#087267);color:white;border:none;border-radius:10px;font-size:0.85rem;font-weight:600;cursor:pointer;">Download PDF Report</button>`;
  setTimeout(() => {
    document.getElementById('confFill').style.width = (data.confidence*100).toFixed(1) + '%';
    data.probabilities.forEach((p, j) => {
      const el = document.getElementById('bar' + j);
      if (el) el.style.width = (p*100).toFixed(1) + '%';
    });
  }, 80);
  if (data.gradcam) {
    document.getElementById('origImg').src = 'data:image/png;base64,' + data.original_image;
    document.getElementById('camImg').src  = 'data:image/png;base64,' + data.gradcam;
    if (data.lesion_overlay) {
      document.getElementById('lesionImg').src = 'data:image/png;base64,' + data.lesion_overlay;
      document.getElementById('lesionLabel').textContent = 'Lesion Overlay \u2014 ' + data.lesion_count + ' candidate regions flagged (red=dark lesion, yellow=bright lesion)';
    }
    document.getElementById('gradcamCard').style.display = 'block';
  }
  document.getElementById('aiSummary').textContent = data.ai_summary;
  document.getElementById('aiAction').textContent  = data.ai_action;
  document.getElementById('aiHcp').textContent     = data.ai_hcp;
  document.getElementById('aiChannel').textContent = data.ai_channel;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function resetToUpload() {
  document.getElementById('uploadState').style.display   = 'block';
  document.getElementById('resultsState').style.display  = 'none';
  document.getElementById('qualityRejectState').style.display = 'none';
  document.getElementById('gradcamCard').style.display   = 'none';
  document.getElementById('imageInput').value            = '';
  document.getElementById('analyzeBtn').disabled         = true;
  document.getElementById('btnText').textContent         = 'Select an image to begin';
  document.getElementById('uploadZone').style.borderColor = '';
}
</script>
<script>
  if (window.gsap && window.ScrollTrigger) {
    gsap.registerPlugin(ScrollTrigger);
    document.querySelectorAll('.lp-reveal').forEach(el => {
      gsap.to(el, {
        opacity: 1, y: 0, duration: 0.6, ease: 'power2.out',
        scrollTrigger: { trigger: el, start: 'top 90%' }
      });
    });
    document.querySelectorAll('.lp-stat .num').forEach(el => {
      const target = parseFloat(el.dataset.count);
      const suffix = el.dataset.suffix || '';
      ScrollTrigger.create({
        trigger: el, start: 'top 92%', once: true,
        onEnter: () => gsap.to({ v: 0 }, {
          v: target, duration: 1.1, ease: 'power2.out',
          onUpdate: function () { el.textContent = this.targets()[0].v.toFixed(target % 1 === 0 ? 0 : 1) + suffix; }
        })
      });
    });
  }
</script>
</body>
</html>"""


@app.route('/')
def index():
    return render_template_string(HTML, **VALIDATION_METRICS)


@app.route('/check_quality', methods=['POST'])
def check_quality():
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    result = process_retinal_image(image_np)

    response = {
        'status': result['status'],
        'is_usable': result['is_usable'],
        'quality_score': result['quality_score'],
        'reason': result['reason'],
        'clinical_guidance': result['clinical_guidance'],
    }

    if result['is_usable'] and result['enhanced_image'] is not None:
        enhanced_rgb = cv2.cvtColor(result['enhanced_image'], cv2.COLOR_BGR2RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb)
        response['enhanced_image_b64'] = img_to_b64(enhanced_pil)

    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file.stream).resize((224, 224)).convert('RGB')
    img_array = np.array(image).astype('float32') / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    






    enhanced_file = request.files.get('enhanced_image')
    enhanced_input = None
    if enhanced_file is not None and enhanced_file.filename:
        enhanced_image = Image.open(enhanced_file.stream).resize((224, 224)).convert('RGB')
        enhanced_input = np.expand_dims(np.array(enhanced_image).astype('float32') / 255.0, axis=0)

    lesion_evidence = None
    if request.form.get('lesion_evidence'):
        lesion_evidence = json.loads(request.form['lesion_evidence'])

    grading = grading_classifier.predict(
        img_input,
        enhanced_image=enhanced_input,
        lesion_evidence=lesion_evidence,
    )
    print("GRADING OUTPUT:", grading)

    predicted_class = grading['severity_grade']
    probs = np.asarray(grading['probabilities'], dtype=np.float32)
    summary = AI_SUMMARIES[predicted_class]

    orig_b64    = img_to_b64(image)
    gradcam_b64 = make_gradcam(img_input, model)

    lesion_overlay_img, lesion_count = detect_lesions(img_array)
    lesion_b64 = img_to_b64(Image.fromarray(lesion_overlay_img))

    return jsonify({
        'predicted_class': predicted_class,
        'severity_grade': grading['severity_grade'],
        'label':           CLASS_LABELS[predicted_class],
        'confidence':      grading['calibrated_confidence'],
        'raw_confidence':  grading['raw_confidence'],
        'calibrated_confidence': grading['calibrated_confidence'],
        'confidence_flag': grading['confidence_flag'],
        'probabilities':   [float(p) for p in probs],
        'description':     CLINICAL_DESC[predicted_class],
        'ai_summary':      summary['summary'],
        'ai_action':       summary['action'],
        'ai_hcp':          summary['hcp'],
        'ai_channel':      summary['channel'],
        'original_image':  orig_b64,
        'gradcam':         gradcam_b64,
        'lesion_overlay':  lesion_b64,
        'lesion_count':    lesion_count
    })


@app.route('/generate_report', methods=['POST'])
def generate_report():
    import base64, io, datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

    data = request.get_json(force=True) or {}

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch,
                             leftMargin=0.6*inch, rightMargin=0.6*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleCustom', parent=styles['Title'], fontSize=18, spaceAfter=4)
    sub_style = ParagraphStyle('SubCustom', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=14)
    h2 = ParagraphStyle('H2Custom', parent=styles['Heading2'], fontSize=12, spaceBefore=10, spaceAfter=6)
    body = styles['Normal']
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)

    elements = []
    report_id = "DR-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    elements.append(Paragraph("Diabetic Retinopathy Screening Report", title_style))
    elements.append(Paragraph(
        f"Report ID: {report_id} &nbsp;|&nbsp; Generated: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}",
        sub_style
    ))

    label = data.get('label', 'N/A')
    confidence = data.get('confidence') or 0
    quality_score = data.get('quality_score')
    lesion_count = data.get('lesion_count', 'N/A')
    confidence_flag = data.get('confidence_flag', 'ok')
    description = data.get('description', '')

    summary_data = [
        ["Predicted Grade", label],
        ["Model Confidence", f"{confidence*100:.1f}%"],
        ["Input Image Quality Score", f"{quality_score*100:.0f}/100" if quality_score is not None else "N/A"],
        ["Candidate Lesion Regions Flagged", str(lesion_count)],
    ]
    t = Table(summary_data, colWidths=[2.4*inch, 3.4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    if confidence_flag == 'low':
        elements.append(Paragraph(
            '<b><font color="#b45309">Low Confidence Result:</font></b> Calibrated confidence is below the '
            'reliability threshold. Manual review by an ophthalmologist is strongly recommended before any '
            'clinical decision.', body))
        elements.append(Spacer(1, 8))

    elements.append(Paragraph("Clinical Evidence Summary", h2))
    elements.append(Paragraph(description or "No description available.", body))
    elements.append(Spacer(1, 10))

    def b64_to_rlimage(b64_str, size=1.7 * inch):
        if not b64_str:
            return None
        try:
            img_bytes = base64.b64decode(b64_str)
            return RLImage(io.BytesIO(img_bytes), width=size, height=size)
        except Exception:
            return None

    img_cells, label_cells = [], []
    for key, lbl in [('original_image', 'Original'), ('gradcam', 'Grad-CAM'), ('lesion_overlay', 'Lesion Overlay')]:
        rl_img = b64_to_rlimage(data.get(key))
        if rl_img:
            img_cells.append(rl_img)
            label_cells.append(Paragraph(lbl, ParagraphStyle('imglbl', parent=styles['Normal'],
                                                               fontSize=8, alignment=1)))
    if img_cells:
        elements.append(Paragraph("Imaging", h2))
        img_table = Table([img_cells, label_cells])
        img_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        elements.append(img_table)
        elements.append(Spacer(1, 10))

    elements.append(Paragraph("Referral Recommendation", h2))
    referral_text = data.get('ai_action') or "Consult an ophthalmologist to confirm this AI-assisted finding."
    elements.append(Paragraph(referral_text, body))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph(
        "Disclaimer: This report was generated by an AI-assisted screening decision support tool. "
        "It is NOT a medical diagnosis. All findings must be confirmed by a qualified ophthalmologist "
        "before any clinical decision is made. This system is intentionally tuned to prioritize sensitivity "
        "(catching true cases) over specificity, so positive or uncertain findings are designed to be "
        "routed to human review rather than acted on directly.",
        disclaimer_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, mimetype='application/pdf', as_attachment=True,
                      download_name=f'{report_id}.pdf')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
