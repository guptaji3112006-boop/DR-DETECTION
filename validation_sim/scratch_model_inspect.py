import keras
import sys

model_path = r"diabetic-retinopathy-detection\models\diabetic_retinopathy_model.keras"
model = keras.models.load_model(model_path, compile=False)

print(model.summary())

# print details of the first few layers
for i, layer in enumerate(model.layers[:10]):
    print(f"Layer {i}: {layer.name}, {layer.__class__.__name__}")
    
# If it's a Sequential or Functional that wraps a base model
# let's try to find if there's a Rescaling or built-in preprocessing layer
for layer in model.layers:
    if 'rescaling' in layer.name.lower() or 'normalization' in layer.name.lower() or 'preprocess' in layer.name.lower():
        print(f"Found preprocessing layer: {layer.name}, {layer.__class__.__name__}")
        
    if isinstance(layer, keras.Model) or hasattr(layer, 'layers'):
        for sublayer in layer.layers[:10]:
            if 'rescaling' in sublayer.name.lower() or 'normalization' in sublayer.name.lower() or 'preprocess' in sublayer.name.lower():
                print(f"Found nested preprocessing layer: {sublayer.name}, {sublayer.__class__.__name__}")

