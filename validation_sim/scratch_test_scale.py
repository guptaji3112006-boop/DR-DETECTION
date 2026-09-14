import json
import numpy as np
from pathlib import Path
from PIL import Image
import keras

model_path = Path(r"diabetic-retinopathy-detection\models\diabetic_retinopathy_model.keras")
model = keras.models.load_model(model_path, compile=False)

manifest = Path(r"validation_sim\results\test_manifest.csv")
import pandas as pd
df = pd.read_csv(manifest)

tp = 0
tn = 0
fp = 0
fn = 0

print("Testing without / 255.0...")
for idx, row in df.iterrows():
    image_path = Path("external_data") / row['relative_path']
    true_grade = row['true_grade']
    
    with Image.open(image_path) as image:
        image = image.resize((224, 224)).convert("RGB")
        # Removing the / 255.0!
        array = np.asarray(image, dtype=np.float32)
        model_input = np.expand_dims(array, axis=0)
        
    output = np.asarray(model.predict(model_input, verbose=0))
    probabilities = output[0]
    predicted_grade = int(np.argmax(probabilities))
    
    # Binary classification (>=2 is referable)
    is_referable_true = true_grade >= 2
    is_referable_pred = predicted_grade >= 2
    
    if is_referable_true and is_referable_pred: tp += 1
    elif not is_referable_true and not is_referable_pred: tn += 1
    elif not is_referable_true and is_referable_pred: fp += 1
    elif is_referable_true and not is_referable_pred: fn += 1

print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
if (tp+fn) > 0: print(f"Sensitivity: {tp/(tp+fn):.2%}")
if (tn+fp) > 0: print(f"Specificity: {tn/(tn+fp):.2%}")
