# Retinal Image Quality Assessment (IQA) & Enhancement Module
**Project:** Build With Bharat 2.0 — AI Diabetic Retinopathy Triage System (`DR-DETECTION`)
**Role:** Member 1 (Frontline Input Pipeline & Image Quality Lead)

---

## 📌 Executive Summary
In rural and remote screening camps, up to 20-30% of fundus photographs are degraded due to cataracts, poor pupil dilation, patient movement, or incorrect illumination. Feeding degraded images directly into deep learning models leads to false negatives and catastrophic diagnostic errors.

The `iqa_module` serves as the frontline gatekeeper:
1. Validates input format and standardizes resolution ($512 \times 512$).
2. Automatically isolates the retinal Field of View (FOV), filtering out 30%+ black background noise.
3. Quantifies tissue-level Sharpness, Illumination, and Contrast.
4. Generates an empirical Quality Score ($0.0 - 1.0$) with automated `PASS`/`REJECT` triage.
5. Applies clinical LAB-color CLAHE enhancement to highlight microaneurysms and exudates for Member 2's classifier.

---

## 🔬 Calibrated Clinical Metrics (IDRiD Benchmark)

| Metric | Clinical Basis | Acceptance Threshold | Rejection Action |
| :--- | :--- | :--- | :--- |
| **FOV Coverage** | Retinal area visibility | $\ge 50\%$ | Re-align camera / re-center pupil |
| **Sharpness** | Laplacian energy on retinal tissue | $\ge 80.0$ | Stabilize camera / ask patient to focus |
| **Illumination** | Green/Gray retinal mean intensity | $60.0 - 180.0$ | Adjust flash or ambient lighting |
| **Contrast** | Standard deviation of retinal pixels | $\ge 12.0$ | Flag for potential media opacity (cataract) |
| **Quality Score** | Weighted composite score | $\ge 0.60$ | Overall sub-clinical capture $\rightarrow$ Recapture |

---

## 💻 Teammate Interface Contract

```python
from iqa_module import process_retinal_image

result = process_retinal_image("path/to/retina.jpg")

if result["is_usable"]:
    # Member 2 (Model): Pass clean, enhanced 512x512 image
    model_input = result["processed_image"]
else:
    # Member 5 (UI): Display actionable clinical guidance
    print(f"Recapture Guidance: {result['clinical_guidance']}")