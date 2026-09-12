import os
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model and data...")
model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetic_retinopathy_model.keras'),
    compile=False
)
images = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'sample_images.npy'))
labels = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'sample_labels.npy')).astype(int)
print(f"Loaded {len(images)} samples.")

def accuracy(imgs):
    preds = model.predict(imgs, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    return float(np.mean(pred_classes == labels))

def preprocess_clahe(imgs):
    out = []
    for img in imgs:
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
        out.append(enhanced)
    return np.array(out)

stages = []

# Stage 1: bare model, raw images
stages.append({"stage": "Bare model", "accuracy": accuracy(images)})

# Stage 2: + preprocessing (CLAHE contrast enhancement)
enhanced_images = preprocess_clahe(images)
stages.append({"stage": "+ Preprocessing (CLAHE)", "accuracy": accuracy(enhanced_images)})

# Stage 3+: no separate IQA/lesion-feature models available yet, so these
# stages reuse the enhanced-image result as a documented placeholder until
# Member 1 (IQA) and lesion-feature integration are wired in.
stages.append({"stage": "+ IQA filtering (pending integration)", "accuracy": stages[-1]["accuracy"]})
stages.append({"stage": "+ Lesion features (pending integration)", "accuracy": stages[-1]["accuracy"]})
stages.append({"stage": "Full pipeline (current)", "accuracy": stages[-1]["accuracy"]})

df = pd.DataFrame(stages)
csv_path = os.path.join(OUT_DIR, "..", "ablation_results.csv")
df.to_csv(csv_path, index=False)
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#080b12')
ax.set_facecolor('#080b12')
bars = ax.bar(df["stage"], df["accuracy"], color="#4fd1c5")
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy", color="white")
ax.set_title("Ablation Study: Accuracy by Pipeline Stage", color="white", fontsize=13, pad=15)
ax.tick_params(colors="white", rotation=15, labelsize=8)
for spine in ax.spines.values():
    spine.set_color("white")
for bar, acc in zip(bars, df["accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.0%}", ha='center', color="white", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ablation_chart.png"), dpi=120, bbox_inches='tight', facecolor='#080b12')
plt.close()

print("\nDone. Real accuracy for stages 1-2; stages 3-5 pending IQA/lesion integration.")
