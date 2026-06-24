import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Food Classifier", page_icon="🍔")

st.title("🍔 Food Classifier App")
st.write("Upload an image of food to see what the model predicts!")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.keras')
    return model

with st.spinner('Loading model...'):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Classifying...")
    
    # Preprocess image
    # Note: Resize dimensions should match what the model was trained on.
    # Defaulting to 224x224 as it is common.
    target_size = (128, 128) 
    
    try:
        img_resized = image.resize(target_size)
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        # Add scaling if the model requires it
        # img_array = img_array / 255.0

        predictions = model.predict(img_array)
        
        # Assuming classification model
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Display results
        st.success(f"Prediction: Class {predicted_class_idx} (Confidence: {confidence:.2f})")
        st.info("Tip: Update app.py with your specific class labels for more descriptive output.")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
