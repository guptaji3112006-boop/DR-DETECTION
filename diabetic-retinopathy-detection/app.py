import os
import json

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from PIL import Image
import keras
import tensorflow as tf
import base64
import io
import json
import logging
import csv
from flask import Flask, request, jsonify, render_template, send_file
import cv2
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from iqa_module.pipeline import process_retinal_image
from grading_model import LesionAwareSeverityClassifier

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

CLASS_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
RISK_TIERS = ["MONITOR", "MONITOR", "ENGAGE", "ACT NOW", "ACT NOW"]
RISK_COLORS = ["#10b981", "#10b981", "#f59e0b", "#d94f5c", "#d94f5c"]
BAR_COLORS = ["#10b981", "#06b6d4", "#f59e0b", "#f97316", "#d94f5c"]

CLINICAL_DESC = [
    "No DR predicted by the model. Clinical review is recommended when appropriate.",
    "Mild DR predicted by the model. Clinical review is recommended when appropriate.",
    "Moderate DR predicted by the model. Review by a qualified ophthalmologist is recommended.",
    "Severe DR predicted by the model. Prompt review by a qualified ophthalmologist is recommended.",
    "Proliferative DR predicted by the model. Prompt specialist review is recommended.",
]

AI_SUMMARIES = [
    {
        "summary": "Retinal imaging shows no evidence of diabetic retinopathy. Vascular architecture appears intact with no microaneurysms, exudates, or neovascularization detected.",
        "action": "Continue standard diabetes management. Schedule next retinal screening in 12 months.",
        "hcp": "Automated digital reminder via patient portal. No urgent HCP outreach required.",
        "channel": "Digital",
    },
    {
        "summary": "Early-stage microaneurysms identified in the peripheral retinal field. Consistent with mild non-proliferative diabetic retinopathy. No vision-threatening features present.",
        "action": "Optimise glycaemic and blood pressure control. Repeat retinal imaging in 6-12 months.",
        "hcp": "Flag for primary care physician review. Consider patient education on glycaemic targets.",
        "channel": "Digital + PCP Outreach",
    },
    {
        "summary": "Moderate non-proliferative diabetic retinopathy identified. Microaneurysms, retinal haemorrhages, and hard exudates detected in the central field. Progression risk elevated without intervention.",
        "action": "Ophthalmology referral within 30 days. Evaluate candidacy for anti-VEGF therapy.",
        "hcp": "Flag in CRM for HCP outreach via primary care provider. Consider therapy education for prescriber.",
        "channel": "CRM Flag + HCP Engagement",
    },
    {
        "summary": "Severe non-proliferative diabetic retinopathy detected. Extensive haemorrhages, venous beading, and intraretinal microvascular abnormalities across multiple quadrants. High risk of progression to PDR.",
        "action": "Urgent ophthalmology referral within 1 week. Consider pan-retinal photocoagulation or intravitreal injection.",
        "hcp": "Immediate field rep visit recommended. Prioritise specialist referral pathway and therapy initiation.",
        "channel": "Field Rep + Specialist Referral",
    },
    {
        "summary": "Proliferative diabetic retinopathy confirmed. Active neovascularization and vitreous haemorrhage signs detected. Highest-risk stage with imminent threat to vision.",
        "action": "Same-week specialist intervention required. Anti-VEGF therapy or surgical evaluation indicated.",
        "hcp": "Urgent field rep visit. Flag for immediate specialist coordination. Critical therapy initiation opportunity.",
        "channel": "Urgent Field Rep + Specialist",
    },
]

model = keras.models.load_model("models/diabetic_retinopathy_model.keras")
print(
    "MODEL LAYERS:", [(i, l.name, type(l).__name__) for i, l in enumerate(model.layers)]
)

calibration_path = "models/calibration.json"

if os.path.exists(calibration_path):
    with open(calibration_path, "r") as f:
        calibration_data = json.load(f)
    temperature = calibration_data.get("temperature", 1.0)
else:
    temperature = 1.0
grading_classifier = LesionAwareSeverityClassifier(
    model=model, temperature=temperature, low_confidence_threshold=0.60
)
temperature_path = os.getenv("DR_TEMPERATURE_FILE", "models/calibration.json")
configured_temperature = float(os.getenv("DR_TEMPERATURE", "1.0"))
if os.path.exists(temperature_path):
    with open(temperature_path, "r", encoding="utf-8") as calibration_file:
        configured_temperature = float(json.load(calibration_file)["temperature"])
grading_model = LesionAwareSeverityClassifier(
    model,
    temperature=configured_temperature,
    low_confidence_threshold=float(os.getenv("DR_LOW_CONFIDENCE_THRESHOLD", "0.60")),
)

# --- Step 4: Load REAL validation metrics from Member 4's validation_sim results ---
# Selected run per validation_sim/config/selected_runs.json (gated run, IQA-accepted subset).
VALIDATION_METRICS = {
    "sensitivity": "N/A",
    "specificity": "N/A",
    "qwk": "N/A",
    "n_validation": "N/A",
}
try:
    _metrics_csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "validation_sim",
        "results",
        "gated",
        "20260912_111805_221820",
        "evaluation_metrics.csv",
    )
    with open(_metrics_csv_path, "r", newline="", encoding="utf-8") as _f:
        _reader = csv.DictReader(_f)
        for _row in _reader:
            if _row.get("subset") == "IQA Accepted Images (Valid Predictions)":
                VALIDATION_METRICS["sensitivity"] = (
                    f"{float(_row['sensitivity']) * 100:.1f}%"
                )
                VALIDATION_METRICS["specificity"] = (
                    f"{float(_row['specificity']) * 100:.1f}%"
                )
                VALIDATION_METRICS["qwk"] = f"{float(_row['qwk']):.3f}"
                VALIDATION_METRICS["n_validation"] = _row["count"]
                break
    logging.info("Loaded real validation metrics: %s", VALIDATION_METRICS)
except Exception as _exc:
    logging.warning(
        "Could not load validation_sim metrics, falling back to N/A: %s", _exc
    )

# Log model structure at startup to help diagnose Grad-CAM issues
logging.info("Top-level model layers:")
for i, l in enumerate(model.layers):
    logging.info("  [%d] %s (%s)", i, l.name, type(l).__name__)
    if hasattr(l, "layers"):
        conv_names = [sl.name for sl in l.layers if "conv" in sl.name.lower()]
        logging.info("      last 3 conv layers: %s", conv_names[-3:])


def _find_last_conv(layers):
    """Return the last Conv layer from a list, checking by type then name."""
    for l in reversed(layers):
        if isinstance(l, tf.keras.layers.Conv2D):
            return l
    for l in reversed(layers):
        if "conv" in l.name.lower():
            return l
    return None


def make_gradcam(img_array, model, predicted_class, **kwargs):
    try:
        # Find EfficientNetB3
        sub_model = next(
            (
                layer
                for layer in model.layers
                if isinstance(layer, tf.keras.Model)
                and "efficientnet" in layer.name.lower()
            ),
            None,
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
            "Grad-CAM using sub-model=%s, conv=%s", sub_model.name, last_conv.name
        )

        # Create a model that gives:
        # 1. Last convolution feature maps
        # 2. EfficientNet output
        feature_model = tf.keras.Model(
            inputs=sub_model.input, outputs=[last_conv.output, sub_model.output]
        )

        with tf.GradientTape() as tape:
            # EfficientNet forward pass
            conv_output, x = feature_model(img_array, training=False)

            # Continue through the outer model
            # after EfficientNet
            sub_model_index = model.layers.index(sub_model)

            for layer in model.layers[sub_model_index + 1 :]:
                x = layer(x, training=False)

            predictions = x

            class_score = predictions[:, predicted_class]

        # Gradient of predicted class
        # with respect to convolution feature maps
        gradients = tape.gradient(class_score, conv_output)

        if gradients is None or not tf.reduce_all(tf.math.is_finite(gradients)):
            logging.warning("Grad-CAM: gradients are None or non-finite")
            return None

        # Average gradients over height and width
        weights = tf.reduce_mean(gradients, axis=(1, 2))

        # Weighted feature maps
        cam = tf.reduce_sum(
            conv_output * weights[:, tf.newaxis, tf.newaxis, :], axis=-1
        )[0]

        # Keep positive activations
        cam = tf.maximum(cam, 0)

        cam_max = tf.reduce_max(cam)

        if float(cam_max) == 0:
            logging.warning("Grad-CAM: empty activation map")
            return None

        cam = cam / cam_max
        cam = cam.numpy()

        # Resize heatmap
        cam_resized = (
            np.array(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (224, 224), Image.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )

        # Create heatmap
        hue = (1.0 - cam_resized) * 0.67

        hi = (hue * 6).astype(int) % 6
        f = hue * 6 - np.floor(hue * 6)

        ones = np.ones_like(f)
        zeros = np.zeros_like(f)

        r = np.select(
            [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
            [ones, 1 - f, zeros, zeros, f, ones],
        )

        g = np.select(
            [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
            [f, ones, ones, 1 - f, zeros, zeros],
        )

        b = np.select(
            [hi == 0, hi == 1, hi == 2, hi == 3, hi == 4, hi == 5],
            [zeros, zeros, f, ones, ones, 1 - f],
        )

        heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

        # Original image
        orig = (img_array[0] * 255).astype(np.uint8)

        import cv2

        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        fov_mask = gray > 5

        activation_outside = np.sum(cam_resized[~fov_mask])
        logging.info(f"Grad-CAM activation outside FOV: {activation_outside:.4f}")

        blended = orig.copy()
        blended[fov_mask] = (0.55 * orig[fov_mask] + 0.45 * heatmap[fov_mask]).astype(
            np.uint8
        )

        # Resize blended image back to original aspect ratio
        if "target_size" in kwargs:
            target_size = kwargs["target_size"]
            blended_pil = Image.fromarray(blended).resize(
                target_size, Image.Resampling.LANCZOS
            )
        else:
            blended_pil = Image.fromarray(blended)

        # Convert to Base64
        buf = io.BytesIO()

        blended_pil.save(buf, format="PNG")

        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        logging.warning("Grad-CAM failed: %s", e)
        return None


def detect_lesions(img):
    """Classical CV lesion detector. img: RGB float(0-1) or uint8 array. Returns (overlay_uint8_rgb, lesion_count)."""
    try:
        img_uint8 = (
            (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        )
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # 1. FOV Masking
        _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        fov_mask = cv2.erode(
            fov_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )

        # 2. Morphological filtering (Top-Hat for bright lesions, Bottom-Hat for dark lesions)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        bottomhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # 3. Thresholding the morphological results
        _, bright_thresh = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
        _, dark_thresh = cv2.threshold(bottomhat, 30, 255, cv2.THRESH_BINARY)

        # Apply FOV mask to eliminate artifacts outside the retina
        bright_thresh = cv2.bitwise_and(bright_thresh, fov_mask)
        dark_thresh = cv2.bitwise_and(dark_thresh, fov_mask)

        overlay = img_uint8.copy()
        lesion_count = 0

        # Find contours
        for mask, color in [(dark_thresh, (255, 0, 0)), (bright_thresh, (255, 255, 0))]:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                area = cv2.contourArea(c)
                # Area >= 4 pixels to capture small microaneurysms while ignoring 1-2 pixel noise
                if 4 <= area < 1000:
                    (x, y), r = cv2.minEnclosingCircle(c)
                    cv2.circle(overlay, (int(x), int(y)), int(r) + 2, color, 1)
                    lesion_count += 1

        return overlay, lesion_count
    except Exception as e:
        logging.warning("Lesion detection failed: %s", e)
        return None, -1


def img_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


HTML = """
"""


@app.route("/")
def index():
    return render_template("index.html", **VALIDATION_METRICS)


@app.route("/check_quality", methods=["POST"])
def check_quality():
    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    result = process_retinal_image(image_np)

    response = {
        "status": result["status"],
        "is_usable": result["is_usable"],
        "quality_score": result["quality_score"],
        "reason": result["reason"],
        "clinical_guidance": result["clinical_guidance"],
    }

    if result["is_usable"] and result["enhanced_image"] is not None:
        enhanced_rgb = cv2.cvtColor(result["enhanced_image"], cv2.COLOR_BGR2RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb)
        response["enhanced_image_b64"] = img_to_b64(enhanced_pil)

    return jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    orig_image = Image.open(file.stream).convert("RGB")
    orig_w, orig_h = orig_image.size
    max_dim = 800
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        lesion_size = (int(orig_w * scale), int(orig_h * scale))
        lesion_image = orig_image.resize(lesion_size, Image.Resampling.LANCZOS)
    else:
        lesion_image = orig_image.copy()

    lesion_img_array = np.array(lesion_image)

    model_image = orig_image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(model_image).astype("float32") / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    enhanced_file = request.files.get("enhanced_image")
    enhanced_input = None
    if enhanced_file is not None and enhanced_file.filename:
        enhanced_image = (
            Image.open(enhanced_file.stream).resize((224, 224)).convert("RGB")
        )
        enhanced_input = np.expand_dims(
            np.array(enhanced_image).astype("float32") / 255.0, axis=0
        )

    lesion_evidence = None
    if request.form.get("lesion_evidence"):
        lesion_evidence = json.loads(request.form["lesion_evidence"])

    grading = grading_classifier.predict(
        img_input,
        enhanced_image=enhanced_input,
        lesion_evidence=lesion_evidence,
    )
    print("GRADING OUTPUT:", grading)

    predicted_class = grading["severity_grade"]
    probs = np.asarray(grading["probabilities"], dtype=np.float32)
    summary = AI_SUMMARIES[predicted_class]

    orig_b64 = img_to_b64(lesion_image)
    gradcam_b64 = make_gradcam(
        img_input, model, predicted_class, target_size=lesion_image.size
    )

    lesion_result = detect_lesions(lesion_img_array)
    if lesion_result[0] is not None:
        lesion_overlay_img, lesion_count = lesion_result
        lesion_b64 = img_to_b64(Image.fromarray(lesion_overlay_img))
    else:
        lesion_b64 = None
        lesion_count = -1

    return jsonify(
        {
            "predicted_class": predicted_class,
            "severity_grade": grading["severity_grade"],
            "label": CLASS_LABELS[predicted_class],
            "gradcam_target_class": CLASS_LABELS[predicted_class],
            "confidence": grading["calibrated_confidence"],
            "raw_confidence": grading["raw_confidence"],
            "calibrated_confidence": grading["calibrated_confidence"],
            "confidence_flag": grading["confidence_flag"],
            "probabilities": [float(p) for p in probs],
            "description": CLINICAL_DESC[predicted_class],
            "ai_summary": summary["summary"],
            "ai_action": summary["action"],
            "ai_hcp": summary["hcp"],
            "ai_channel": summary["channel"],
            "original_image": orig_b64,
            "gradcam": gradcam_b64,
            "lesion_overlay": lesion_b64,
            "lesion_count": lesion_count,
        }
    )


@app.route("/generate_report", methods=["POST"])
def generate_report():
    import base64, io, datetime
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
    )

    data = request.get_json(force=True) or {}

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=4
    )
    sub_style = ParagraphStyle(
        "SubCustom",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=14,
    )
    h2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6
    )
    body = styles["Normal"]
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.grey
    )

    elements = []
    report_id = "DR-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    elements.append(Paragraph("Diabetic Retinopathy Screening Report", title_style))
    elements.append(
        Paragraph(
            f"Report ID: {report_id} &nbsp;|&nbsp; Generated: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}",
            sub_style,
        )
    )

    label = data.get("label", "N/A")
    confidence = data.get("confidence") or 0
    quality_score = data.get("quality_score")
    lesion_count = data.get("lesion_count", "N/A")
    lesion_count_str = str(lesion_count)
    if lesion_count == -1:
        lesion_count_str = "Lesion detection unavailable"
    elif lesion_count == 0:
        lesion_count_str = "0 (Does not rule out DR)"

    confidence_flag = data.get("confidence_flag", "ok")
    description = data.get("description", "")

    summary_data = [
        ["Predicted Grade", label],
        ["Model Confidence", f"{confidence * 100:.1f}%"],
        [
            "Input Image Quality Score",
            f"{quality_score * 100:.0f}/100" if quality_score is not None else "N/A",
        ],
        ["Candidate Lesion Regions Flagged", lesion_count_str],
    ]
    t = Table(summary_data, colWidths=[2.4 * inch, 3.4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ]
        )
    )
    elements.append(t)
    elements.append(Spacer(1, 12))

    if confidence_flag == "low":
        elements.append(
            Paragraph(
                '<b><font color="#b45309">Low Confidence Result:</font></b> Calibrated confidence is below the '
                "reliability threshold. Manual review by an ophthalmologist is strongly recommended before any "
                "clinical decision.",
                body,
            )
        )
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

    orig_img = b64_to_rlimage(data.get("original_image"))
    if orig_img:
        img_cells.append(orig_img)
        label_cells.append(
            Paragraph(
                "Original",
                ParagraphStyle(
                    "imglbl", parent=styles["Normal"], fontSize=8, alignment=1
                ),
            )
        )

    gradcam_img = b64_to_rlimage(data.get("gradcam"))
    target_class = data.get("gradcam_target_class", "N/A")
    # Simple color legend using HTML-like text
    lbl_text = f"Grad-CAM (Target: {target_class})<br/>Low -<font color='blue'>■</font><font color='cyan'>■</font><font color='green'>■</font><font color='yellow'>■</font><font color='red'>■</font>- High"

    if gradcam_img:
        img_cells.append(gradcam_img)
    else:
        img_cells.append(
            Paragraph(
                "<br/><br/><br/>Grad-CAM<br/>unavailable",
                ParagraphStyle(
                    "unavailable",
                    parent=styles["Normal"],
                    fontSize=8,
                    alignment=1,
                    textColor=colors.grey,
                ),
            )
        )

    label_cells.append(
        Paragraph(
            lbl_text,
            ParagraphStyle("imglbl", parent=styles["Normal"], fontSize=8, alignment=1),
        )
    )

    lesion_img = b64_to_rlimage(data.get("lesion_overlay"))
    if lesion_img:
        img_cells.append(lesion_img)
        label_cells.append(
            Paragraph(
                "Lesion Overlay",
                ParagraphStyle(
                    "imglbl", parent=styles["Normal"], fontSize=8, alignment=1
                ),
            )
        )

    if img_cells:
        elements.append(Paragraph("Imaging", h2))
        img_table = Table([img_cells, label_cells])
        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        elements.append(img_table)
        elements.append(Spacer(1, 10))

    elements.append(Paragraph("Referral Recommendation", h2))
    referral_text = (
        data.get("ai_action")
        or "Consult an ophthalmologist to confirm this AI-assisted finding."
    )
    elements.append(Paragraph(referral_text, body))
    elements.append(Spacer(1, 16))

    elements.append(
        Paragraph(
            "Disclaimer: This report was generated by an AI-assisted screening decision support tool. "
            "It is NOT a medical diagnosis. All findings must be confirmed by a qualified ophthalmologist "
            "before any clinical decision is made. This system is intentionally tuned to prioritize sensitivity "
            "(catching true cases) over specificity, so positive or uncertain findings are designed to be "
            "routed to human review rather than acted on directly.",
            disclaimer_style,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{report_id}.pdf",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
