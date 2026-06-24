import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Class labels
CLASS_LABELS = {
    0: "Pizza",
    1: "Soft Drink",
    2: "Burger",
}

CLASS_EMOJIS = {
    0: "🍕",
    1: "🥤",
    2: "🍔",
}

# Set page config
st.set_page_config(page_title="Food Classifier", page_icon="🍔")

st.title("🍔 Food Classifier App")
st.write("Upload an image of food to see what the model predicts!")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_model (2).keras")
    return model

with st.spinner("Loading model..."):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")

    try:
        # Preprocess image exactly like Colab
        target_size = (128, 128)  # must match training size
        img = image.load_img(uploaded_file, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0   # normalize
        img_array = np.expand_dims(img_array, axis=0) # add batch dimension

        # Predict
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # Display results
        label = CLASS_LABELS.get(int(predicted_class_idx), f"Unknown ({predicted_class_idx})")
        emoji = CLASS_EMOJIS.get(int(predicted_class_idx), "❓")
        st.success(f"{emoji} Prediction: **{label}** (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
