"""
Ablation study: measures accuracy at each pipeline stage.
Uses dummy/placeholder accuracy values until the real trained model
and full test dataset are available -- swap in real evaluation logic
in the marked section below once ready.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(OUT_DIR, exist_ok=True)

# ── STAGES ───────────────────────────────────────────────────────────
# Replace these placeholder numbers with real accuracy once you can
# run each stage against the labeled test set.
stages = [
    {"stage": "Bare model",              "accuracy": 0.62},
    {"stage": "+ Preprocessing",          "accuracy": 0.68},
    {"stage": "+ IQA filtering",          "accuracy": 0.74},
    {"stage": "+ Lesion features",        "accuracy": 0.81},
    {"stage": "Full pipeline",            "accuracy": 0.87},
]

df = pd.DataFrame(stages)
csv_path = os.path.join(OUT_DIR, "..", "ablation_results.csv")
df.to_csv(csv_path, index=False)
print(f"Saved table: {csv_path}")
print(df.to_string(index=False))

# ── CHART ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#080b12')
ax.set_facecolor('#080b12')

bars = ax.bar(df["stage"], df["accuracy"], color="#4fd1c5")
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy", color="white")
ax.set_title("Ablation Study: Accuracy by Pipeline Stage", color="white", fontsize=13, pad=15)
ax.tick_params(colors="white", rotation=15)
for spine in ax.spines.values():
    spine.set_color("white")

for bar, acc in zip(bars, df["accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.0%}",
            ha='center', color="white", fontsize=10)

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, "ablation_chart.png")
plt.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='#080b12')
plt.close()
print(f"Saved chart: {chart_path}")

print("\nDone. NOTE: accuracy values are placeholders -- replace with real")
print("evaluation results once the trained model + labeled test set are ready.")
