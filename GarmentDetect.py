import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained ResNet50 model
model = keras.applications.resnet50.ResNet50(weights='imagenet')

# Load and preprocess the input image
def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Classify the image
def classify_cloth_type(img_path):
    img = load_image(img_path)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=3)[0]  # Top 3 predictions
    cloth_types = [label for (_, label, _) in decoded_preds]
    return cloth_types

# Path to the input image
image_path = '1440-X-2160-51-600x900.jpg'

# Classify the cloth type
predicted_types = classify_cloth_type(image_path)

# Print the predicted cloth types
print("Predicted cloth types:")
for cloth_type in predicted_types:
    print(cloth_type)
