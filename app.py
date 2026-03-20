import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)
# Use the converted Keras 3 compatible model
MODEL_PATH = "brain_tumor_model.keras"
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = (224, 224)
# -----------------------------
# Custom layers needed by the saved model
# -----------------------------
@tf.keras.utils.register_keras_serializable()
def convert_to_rgb(image):
    return tf.image.grayscale_to_rgb(image)
@tf.keras.utils.register_keras_serializable()
class TrueDivide(tf.keras.layers.Layer):
    def call(self, x):
        return x / 127.5
@tf.keras.utils.register_keras_serializable()
class Subtract(tf.keras.layers.Layer):
    def call(self, x):
        return x - 1.0
# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    # We use compile=False because we only need the model for inference
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "convert_to_rgb": convert_to_rgb,
            "TrueDivide": TrueDivide,
            "Subtract": Subtract,
        },
        compile=False,
    )
    return model
# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("L")   # grayscale
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    
    # IMPORTANT: Do NOT normalize here by 255.0!
    # The model handles normalization internally via TrueDivide and Subtract layers.
    
    img_array = np.expand_dims(img_array, axis=-1)   # (224, 224, 1)
    img_array = np.expand_dims(img_array, axis=0)    # (1, 224, 224, 1)
    return img, img_array
# -----------------------------
# Prediction
# -----------------------------
def predict_image(model, img_array):
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index]) * 100
    return predicted_class, confidence, predictions
# -----------------------------
# UI
# -----------------------------
st.title("🧠 Brain Tumor MRI Image Classification")
st.write(
    "Upload a brain MRI image to predict the tumor type and view confidence scores."
)
with st.spinner("Loading AI model... Please wait."):
    model = load_model()
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    display_img, input_array = preprocess_image(uploaded_file)
    st.image(display_img, caption="Uploaded MRI Image", use_container_width=True)
    with st.spinner("Analyzing MRI image..."):
        predicted_class, confidence, predictions = predict_image(model, input_array)
    st.subheader("Prediction Result")
    st.success(f"Predicted Tumor Type: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
    st.subheader("Confidence Scores")
    for class_name, score in zip(CLASS_NAMES, predictions):
        st.write(f"**{class_name}**: {score * 100:.2f}%")
    st.subheader("Confidence Chart")
    chart_data = {
        "Class": CLASS_NAMES,
        "Confidence": [float(score) * 100 for score in predictions]
    }
    st.bar_chart(chart_data, x="Class", y="Confidence")
st.markdown("---")
st.caption("This project is for educational use and not for clinical diagnosis.")