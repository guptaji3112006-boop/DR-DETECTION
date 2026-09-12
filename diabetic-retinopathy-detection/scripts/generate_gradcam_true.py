import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
FILE_NAMES  = ['gradcam_true_nodr', 'gradcam_true_mild', 'gradcam_true_moderate',
               'gradcam_true_severe', 'gradcam_true_proliferative']
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')

print("Loading model...")
model = tf.keras.models.load_model(
    os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetic_retinopathy_model.keras'),
    compile=False
)
print("Model loaded.")

print("Loading sample data...")
images = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'sample_images.npy'))
labels = np.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'sample_labels.npy'))
print(f"Loaded {len(images)} samples.")

# Reuse the embedded efficientnetb3 sub-model's own input/output directly
base = model.get_layer('efficientnetb3')
x = base.output
for layer in model.layers[1:]:
    x = layer(x)
grad_model = tf.keras.Model(base.input, [base.output, x])

def make_gradcam_heatmap(img_array, pred_index=None):
    inp = tf.convert_to_tensor(img_array[np.newaxis], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(inp)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), float(tf.nn.softmax(preds[0])[pred_index])

os.makedirs(OUT_DIR, exist_ok=True)

for cls_idx in range(5):
    idxs = np.where(labels.astype(int) == cls_idx)[0]
    if len(idxs) == 0:
        print(f"WARNING: no samples for class {cls_idx} ({CLASS_NAMES[cls_idx]}), skipping.")
        continue

    img = images[idxs[0]].astype(np.float32)
    heatmap, pred_cls, confidence = make_gradcam_heatmap(img)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('#080b12')

    axes[0].imshow(img)
    axes[0].set_title('Original Retinal Scan', color='white', fontsize=12, pad=10)
    axes[0].axis('off')

    axes[1].imshow(img)
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.45)
    axes[1].set_title(
        f'Grad-CAM\n{CLASS_NAMES[pred_cls]} - {confidence:.0%} confidence',
        color='white', fontsize=12, pad=10
    )
    axes[1].axis('off')

    plt.tight_layout(pad=1.5)
    out_path = os.path.join(OUT_DIR, f'{FILE_NAMES[cls_idx]}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#080b12')
    plt.close()

    print(f"[{cls_idx}] {CLASS_NAMES[cls_idx]:15s} -> pred: {CLASS_NAMES[pred_cls]:15s} "
          f"conf: {confidence:.1%}  saved: {out_path}")

print("\nDone. True Grad-CAM images saved to images/")
