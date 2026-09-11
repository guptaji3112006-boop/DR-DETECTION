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

## 💻 Output Contract & Teammate Interface

```python
from iqa_module import process_retinal_image

result = process_retinal_image("path/to/retina.jpg")

# Expected Output Contract:
# {
#     "quality_score": 0.925,                  # float: 0.00 to 1.00
#     "status": "PASS",                        # str: "PASS" or "REJECT"
#     "reason": "Good clinical quality...",    # str: clinical assessment summary
#     "enhanced_image": np.ndarray or None,    # 512x512 enhanced BGR image (None if REJECT)
#     "is_usable": True,                       # bool: True if PASS, False if REJECT
#     "clinical_guidance": "...",              # str: Actionable instruction for camp worker
#     "metrics": { ... }                       # dict: tissue-level measurements
# }

if result["is_usable"]:
    # Member 2 (Disease Grading Classifier):
    model_input = result["enhanced_image"]
else:
    # Member 5 (Tele-Ophthalmology UI):
    # Trigger Reject-and-Recapture loop with specific guidance
    print(f"Status: {result['status']}")
    print(f"Recapture Instructions: {result['clinical_guidance']}")
```

---

## 🛡️ Feature 9: Safety & Clinical Limitations

> [!IMPORTANT]
> **Screening & Triage Positioning — Not an Autonomous Diagnostic Device**

1. **Intended Clinical Context**:
   - The `iqa_module` and downstream triage pipeline are engineered as an **assistive Clinical Decision Support System (CDSS)** for frontline screening in primary health centers (PHCs), mobile eye camps, and vision centers.
   - It is designed to be operated by trained healthcare workers (e.g., ASHA/ANM workers, ophthalmic technicians) to flag patients requiring urgent specialist evaluation.
   - **This software does not make definitive medical diagnoses.** Diagnostic confirmation and treatment prescription remain the exclusive prerogative of a licensed ophthalmologist or retina specialist.

2. **Reject-and-Recapture Safety Mechanism**:
   - If an image fails the quality gate (`REJECT`), it is intentionally barred from entering the diagnostic model (`enhanced_image` is returned as `None`).
   - This prevents **catastrophic false negatives** where cataracts, severe motion blur, or corneal glare obscure sight-threatening proliferative diabetic retinopathy (PDR) or diabetic macular edema (DME).
   - If an eye consistently fails IQA after 3 recapture attempts due to physiological opacities (e.g., dense cataract or small pupil), the protocol requires a **direct clinical referral** rather than assuming the eye is healthy.

3. **Enhancement Boundary Safeguards**:
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) is strictly clamped (`clipLimit=2.0`) in the LAB luminance space. This prevents over-amplification of noise that could otherwise mimic microaneurysms or small hemorrhages.
   - Circular Field of View (FOV) masking isolates retinal parenchyma and suppresses peripheral camera lens reflection artifacts.

---

## 📚 References & Scientific Prior Art

* **EyeQ Dataset & Multi-Color Space Fusion Network (MCF-Net)**:
  * Fu et al., *"Evaluation of Retinal Image Quality Assessment Networks in Different Color-Spaces"*, MICCAI / arXiv.
  * Repository: [HzFu/EyeQ](https://github.com/HzFu/EyeQ) — Quality grading standards (Good, Usable, Reject) across 28,792 fundus photographs.
* **Retinal Degradation Modeling & Correction**:
  * Fu et al., *"Modeling and Enhancing Low-quality Retinal Fundus Images"*, IEEE TMI.
  * Repository: [HzFu/EyeQ_enhancement](https://github.com/HzFu/EyeQ_enhancement) — Low-quality fundus restoration, illumination normalization, and lesion preservation principles.
* **Clinical Calibration Benchmark**:
  * Indian Diabetic Retinopathy Image Dataset (IDRiD), Zenodo Open Access. Empirical threshold calibration on verified clinical fundus acquisitions.