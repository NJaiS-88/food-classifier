import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("my_model.keras")

# Define labels
labels = {0: "Pizza", 1: "Softdrink", 2: "Burger"}

# Title
st.title("Food Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and convert to RGB (drop alpha channel if present)
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess the image (adjust size to match your model input)
    img = image.resize((128, 128))  # change size if your model expects different input
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Show result
    st.write(f"Prediction: {predicted_class} → {labels[predicted_class]}")

    # Optional: show confidence scores
    st.write("Confidence scores:")
    for i, score in enumerate(prediction[0]):
        st.write(f"{labels[i]}: {score:.2f}")
